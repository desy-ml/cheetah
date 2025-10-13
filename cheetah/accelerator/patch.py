from matplotlib import pyplot as plt
import torch

from scipy.constants import speed_of_light

from cheetah.accelerator.element import Element
from cheetah.particles import Beam
from cheetah.particles.particle_beam import ParticleBeam
from cheetah.utils import UniqueNameGenerator

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
        drift_to_exit: bool = True,
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.register_buffer_or_parameter(
            "offset",
            torch.as_tensor(offset if offset is not None else [0.0, 0.0, 0.0], **factory_kwargs),
        )
        self.register_buffer_or_parameter(
            "time_offset",
            torch.as_tensor(
                time_offset if time_offset is not None else 0.0, **factory_kwargs
            ),
        )
        self.register_buffer_or_parameter(
            "pitch",
            torch.as_tensor(pitch if pitch is not None else [0.0, 0.0], **factory_kwargs),
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
        self.drift_to_exit = drift_to_exit

    @property
    def length(self) -> torch.Tensor:
        rotation_matrix_inv = self.rotation_matrix().inverse()
        return (
            rotation_matrix_inv[-1, 0] * self.offset[0]
            + rotation_matrix_inv[-1, 1] * self.offset[1]
            + rotation_matrix_inv[-1, 2] * self.offset[2]
        )

    def transform_particles(self, incoming: ParticleBeam) -> Beam:
        particles = incoming.particles
        final_particles = particles.clone()

        # position coordinates
        entrance_position = particles[..., :-1:2]
        # momentum coordinates
        rel_p = particles[..., -2] + 1.0  # convert delta to p/p0
        p_vec = torch.stack(
            [
                particles[..., 1],
                particles[..., 3],
                torch.sqrt(
                    (rel_p) ** 2 - particles[..., 1] ** 2 - particles[..., 3] ** 2
                ),
            ], dim=-1
        )

        # compute the exit positions and momentum - note these computations follow bmad coordinates
        rotation_matrix_inv = self.rotation_matrix().inverse()    
        r_vec = torch.stack((
            entrance_position[...,0] - self.offset[0],
            entrance_position[...,1] - self.offset[1],
            - self.offset[2].expand(entrance_position[...,0].shape)
        ), dim=-1)
        r_vec = (rotation_matrix_inv @ r_vec.transpose(-1,-2)).transpose(-1,-2)
        p_vec = (rotation_matrix_inv @ p_vec.transpose(-1,-2)).transpose(-1,-2)

        final_particles[..., 4] = final_particles[..., 4] + self.time_offset * speed_of_light  # time offset update

        # set final momenta
        final_particles[..., 1] = p_vec[..., 0]
        final_particles[..., 3] = p_vec[..., 1]

        # set final positions
        final_particles[..., 0] = r_vec[..., 0]
        final_particles[..., 2] = r_vec[..., 1]

        # track particles to the end of the patch
        if self.drift_to_exit:
            final_particles[..., 0] = final_particles[..., 0] - (
                r_vec[..., 2] * p_vec[..., 0] / p_vec[..., 2]
            )
            final_particles[..., 2] = final_particles[..., 2] - (
                r_vec[..., 2] * p_vec[..., 1] / p_vec[..., 2]
            )
            final_particles[..., 4] = final_particles[..., 4] + r_vec[..., 2] * rel_p / p_vec[..., 2] + self.length


        # convert momentum back to delta
        return ParticleBeam(
            particles=final_particles,
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

    @property
    def is_skippable(self) -> bool:
        return False  # Patch elements cannot be skipped
