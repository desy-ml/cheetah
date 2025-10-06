from matplotlib import pyplot as plt
import torch
from cheetah.accelerator.element import Element
from cheetah.particles import Beam
from cheetah.particles.particle_beam import ParticleBeam
from cheetah.utils import UniqueNameGenerator, verify_device_and_dtype


generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Patch(Element):
    """
    Patch element that shifts the reference orbit and time. Note that this element does not
    support batching for the offset, time offset, pitch, tilt, E_tot_offset, and E_tot_set parameters.

    :param offset:  Exit face offset in (x,y,z) from Entrance in meters.
    :param time_offset: Reference time offset in seconds.
    :param pitch: Exit face orientation (x,y,z) from Entrance in radians.
    :param tilt: Tilt angle in the x-y plane in radians.
    :param E_tot_offset: Energy offset in (eV).
    :param E_tot_set: Energy setpoint in (eV).
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment.

    """

    def __init__(
        self,
        offset: torch.Tensor | None = None,
        time_offset: torch.Tensor | None = None,
        pitch: torch.Tensor | None = None,
        tilt: torch.Tensor | None = None,
        E_tot_offset: torch.Tensor | None = None,
        E_tot_set: torch.Tensor | None = None,
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        device, dtype = verify_device_and_dtype(
            [offset, time_offset, pitch, tilt, E_tot_offset, E_tot_set], device, dtype
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.register_buffer_or_parameter(
            "offset",
            torch.as_tensor(offset if offset is not None else 0.0, **factory_kwargs),
        )
        self.register_buffer_or_parameter(
            "time_offset",
            torch.as_tensor(
                time_offset if time_offset is not None else 0.0, **factory_kwargs
            ),
        )
        self.register_buffer_or_parameter(
            "pitch",
            torch.as_tensor(pitch if pitch is not None else 0.0, **factory_kwargs),
        )
        self.register_buffer_or_parameter(
            "tilt", torch.as_tensor(tilt if tilt is not None else 0.0, **factory_kwargs)
        )
        self.register_buffer_or_parameter(
            "E_tot_offset",
            torch.as_tensor(
                E_tot_offset if E_tot_offset is not None else 0.0, **factory_kwargs
            ),
        )
        self.register_buffer_or_parameter(
            "E_tot_set",
            torch.as_tensor(
                E_tot_set if E_tot_set is not None else 0.0, **factory_kwargs
            ),
        )

    @property
    def length(self) -> torch.Tensor:
        rotation_matrix = self.rotation_matrix()
        return (
            rotation_matrix[-1, 0] * self.offset[0]
            + rotation_matrix[-1, 1] * self.offset[1]
            + rotation_matrix[-1, 2] * self.offset[2]
        )

    def transform_particles(self, incoming: ParticleBeam) -> Beam:
        particles = incoming.particles

        # position coordinates
        entrance_position = particles[..., :-1:2]
        # momentum coordinates
        rel_p = particles[..., 5] + 1.0  # convert delta to p/p0
        entrance_momentum = torch.vstack(
            [
                particles[..., 1],
                particles[..., 3],
                torch.sqrt(
                    (rel_p) ** 2 - particles[..., 1] ** 2 - particles[..., 3] ** 2
                ),
            ]
        ).T

        # compute the exit positions and momentum - note these computations follow bmad coordinates
        rotation_matrix = self.rotation_matrix()
        exit_positions = (rotation_matrix.T @ (entrance_position - self.offset).T).T
        exit_momentum = (rotation_matrix.T @ entrance_momentum.T).T

        # track particles to the end of the patch
        exit_positions[..., 0] += (
            -exit_positions[..., 2] * exit_momentum[..., 0] / exit_momentum[..., 2]
        )
        exit_positions[..., 1] += (
            -exit_positions[..., 2] * exit_momentum[..., 1] / exit_momentum[..., 2]
        )
        exit_positions[..., 2] += (
            -exit_positions[..., 2] * rel_p / exit_momentum[..., 2]
        )

        # Interleave last dimensions of exit_positions and exit_momentum
        exit_particles = torch.empty(
            (
                *exit_positions.shape[:-1],
                exit_positions.shape[-1] + exit_momentum.shape[-1] + 1,
            ),
            device=exit_positions.device,
            dtype=exit_positions.dtype,
        )
        exit_particles[..., :-1:2] = exit_positions
        exit_particles[..., 1:-1:2] = exit_momentum
        exit_particles[..., -1] = particles[..., -1]

        # convert momentum back to delta
        exit_particles[..., -2] -= 1.0

        return ParticleBeam(
            particles=exit_particles,
            energy=incoming.energy + self.E_tot_offset,
            s=self.length + incoming.s,
        )

    def rotation_matrix(self) -> torch.Tensor:
        """
        Returns the rotation matrix for the patch element based on its pitch and tilt.
        """
        pitch = self.pitch
        tilt = self.tilt

        rotation_y = torch.tensor(
            [
                [torch.cos(pitch[0]), 0.0, torch.sin(pitch[0])],
                [0.0, 1.0, 0.0],
                [-torch.sin(pitch[0]), 0.0, torch.cos(pitch[0])],
            ],
            device=pitch.device,
            dtype=pitch.dtype,
        )

        rotation_neg_x = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, torch.cos(pitch[1]), torch.sin(pitch[1])],
                [0.0, -torch.sin(pitch[1]), torch.cos(pitch[1])],
            ],
            device=pitch.device,
            dtype=pitch.dtype,
        )

        rotation_z = torch.tensor(
            [
                [torch.cos(tilt), -torch.sin(tilt), 0.0],
                [torch.sin(tilt), torch.cos(tilt), 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=tilt.device,
            dtype=tilt.dtype,
        )

        return rotation_y @ rotation_neg_x @ rotation_z

    def track(self, incoming: Beam) -> Beam:
        if isinstance(incoming, ParticleBeam):
            return self.transform_particles(incoming)
        else:
            raise TypeError("Patch element currently only supports ParticleBeam input.")

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        pass  # Patch element does not have a visual representation

    def is_skippable(self) -> bool:
        return False  # Patch elements cannot be skipped
