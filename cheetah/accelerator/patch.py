import torch
from matplotlib import pyplot as plt
from scipy.constants import speed_of_light

from cheetah.accelerator.element import Element
from cheetah.particles import Beam
from cheetah.particles.particle_beam import ParticleBeam
from cheetah.utils import UniqueNameGenerator

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Patch(Element):
    """
    Patch element that shifts the reference orbit and time. Note that this element does
    not support batching for the `offset`, `time_offset`, `pitch`, `tilt`,
    `energy_offset`, and `energy_setpoint` parameters.

    :param offset: Exit face offset in (x, y, z) from the entrance in meters.
    :param time_offset: Reference time offset in seconds.
    :param pitch: Exit face orientation (x, y, z) from the entrance in radians.
    :param tilt: Tilt angle in the x-y plane in radians.
    :param energy_offset: Energy offset in eV.
    :param energy_setpoint: Energy setpoint in eV.
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
        energy_offset: torch.Tensor | None = None,
        energy_setpoint: torch.Tensor | None = None,
        drift_to_exit: bool = True,
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.drift_to_exit = drift_to_exit

        self.register_buffer_or_parameter(
            "offset",
            torch.as_tensor(
                offset if offset is not None else [0.0, 0.0, 0.0], **factory_kwargs
            ),
        )
        self.register_buffer_or_parameter(
            "time_offset",
            torch.as_tensor(
                time_offset if time_offset is not None else 0.0, **factory_kwargs
            ),
        )
        self.register_buffer_or_parameter(
            "pitch",
            torch.as_tensor(
                pitch if pitch is not None else [0.0, 0.0], **factory_kwargs
            ),
        )
        self.register_buffer_or_parameter(
            "tilt", torch.as_tensor(tilt if tilt is not None else 0.0, **factory_kwargs)
        )
        self.register_buffer_or_parameter(
            "energy_offset",
            torch.as_tensor(
                energy_offset if energy_offset is not None else 0.0, **factory_kwargs
            ),
        )
        self.register_buffer_or_parameter(
            "energy_setpoint",
            torch.as_tensor(
                energy_setpoint if energy_setpoint is not None else 0.0,
                **factory_kwargs,
            ),
        )

    def track(self, incoming: Beam) -> Beam:
        if isinstance(incoming, ParticleBeam):
            return self.transform_particles(incoming)
        else:
            raise TypeError("Patch element currently only supports ParticleBeam input.")

    def transform_particles(self, incoming: ParticleBeam) -> Beam:
        outgoing_particles = incoming.particles.clone()

        # Momentum coordinates
        rel_p = incoming.p + 1.0  # Convert delta to p / p0
        p_vec = torch.stack(
            [
                incoming.px,
                incoming.py,
                (
                    rel_p.pow(2)
                    - incoming.particles[..., 1].pow(2)
                    - incoming.particles[..., 3].pow(2)
                ).sqrt(),
            ],
            dim=-1,
        )

        # Compute the exit positions and momentum
        # NOTE: These computations follow Bmad coordinates
        rotation_matrix_inverse = self._rotation_matrix().inverse()
        r_vec = torch.stack(
            (
                incoming.x - self.offset[0],
                incoming.y - self.offset[1],
                -self.offset[2].expand(incoming.x.shape),
            ),
            dim=-1,
        )
        r_vec = (rotation_matrix_inverse @ r_vec.mT).mT
        p_vec = (rotation_matrix_inverse @ p_vec.mT).mT

        outgoing_particles = torch.stack(
            [
                r_vec[..., 0],
                p_vec[..., 0],
                r_vec[..., 1],
                p_vec[..., 1],
                incoming.particles[..., 4] + self.time_offset * speed_of_light,
                incoming.particles[..., 5],
                incoming.particles[..., 6],
            ]
        )

        # Track particles to the end of the patch
        if self.drift_to_exit:
            outgoing_particles[..., [0, 2]] = outgoing_particles[..., [0, 2]] - (
                r_vec[..., 2] * p_vec[..., [0, 1]] / p_vec[..., 2]
            )
            outgoing_particles[..., 4] = (
                outgoing_particles[..., 4]
                + r_vec[..., 2] * rel_p / p_vec[..., 2]
                + self.length
            )

        return ParticleBeam(
            particles=outgoing_particles,
            energy=incoming.energy + self.energy_offset,
            particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities,
            s=incoming.s + self.length,
            species=incoming.species,
        )

    def _rotation_matrix(self) -> torch.Tensor:
        """
        Computes the rotation matrix for the patch element based on its pitch and tilt.
        """
        factory_kwargs = {"device": self.pitch.device, "dtype": self.pitch.dtype}

        rotation_y = torch.tensor(
            [
                [self.pitch[0].cos(), 0.0, self.pitch[0].sin()],
                [0.0, 1.0, 0.0],
                [-self.pitch[0].sin(), 0.0, self.pitch[0].cos()],
            ],
            **factory_kwargs,
        )
        rotation_neg_x = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, self.pitch[1].cos(), self.pitch[1].sin()],
                [0.0, -self.pitch[1].sin(), self.pitch[1].cos()],
            ],
            **factory_kwargs,
        )
        rotation_z = torch.tensor(
            [
                [self.tilt.cos(), -self.tilt.sin(), 0.0],
                [self.tilt.sin(), self.tilt.cos(), 0.0],
                [0.0, 0.0, 1.0],
            ],
            **factory_kwargs,
        )

        return rotation_y @ rotation_neg_x @ rotation_z

    @property
    def length(self) -> torch.Tensor:
        return (self._rotation_matrix().inverse()[-1, 0:3] * self.offset).sum()

    @property
    def is_skippable(self) -> bool:
        return False  # Patch elements cannot be skipped

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        pass  # Patch element does not have a visual representation
