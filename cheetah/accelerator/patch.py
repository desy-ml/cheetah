import torch
from matplotlib import pyplot as plt
from scipy.constants import speed_of_light

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParameterBeam, ParticleBeam, Species
from cheetah.track_methods import patch_rotation_matrix, patch_transfer_map
from cheetah.utils import UniqueNameGenerator

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Patch(Element):
    """
    Patch element that shifts the reference orbit and time.

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
        metadata: dict | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            name=name, sanitize_name=sanitize_name, metadata=metadata, **factory_kwargs
        )

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
            return self._transform_particles(incoming)
        elif isinstance(incoming, ParameterBeam):
            return self._transform_parameters(incoming)

    def _transform_particles(self, incoming: ParticleBeam) -> Beam:
        # Momentum coordinates
        rel_p = incoming.p + 1.0  # Convert delta to p / p0
        pz_sq = rel_p.square() - incoming.px.square() - incoming.py.square()
        pz = torch.clamp(pz_sq, min=0.0).sqrt()
        particle_momenta = torch.stack(
            [
                incoming.px,
                incoming.py,
                pz,
            ],
            dim=-1,
        )

        R = self._rotation_matrix()
        offset_x = self.offset[..., 0].unsqueeze(-1)
        offset_y = self.offset[..., 1].unsqueeze(-1)
        offset_z = self.offset[..., 2].unsqueeze(-1)

        dx = incoming.x - offset_x
        dy = incoming.y - offset_y
        dz_in = -offset_z.expand_as(dx)
        shifted_positions = torch.stack([dx, dy, dz_in], dim=-1)

        # Coordinate transformation to exit frame: (r_entrance - offset) @ R
        particle_positions = shifted_positions @ R
        particle_momenta = particle_momenta @ R

        batch_shape = particle_positions.shape[:-2]
        num_particles = particle_positions.shape[-2]
        outgoing_particles = incoming.particles.expand(
            *batch_shape, num_particles, 7
        ).clone()

        outgoing_particles[..., [0, 2]] = particle_positions[..., [0, 1]]
        outgoing_particles[..., [1, 3]] = particle_momenta[..., [0, 1]]
        time_offset_z = (self.time_offset * speed_of_light).unsqueeze(-1)
        outgoing_particles[..., 4] = incoming.particles[..., 4] + time_offset_z

        # Track particles to the end of the patch
        if self.drift_to_exit:
            dz_exit = particle_positions[..., 2:3]
            pz_exit = particle_momenta[..., 2:3]
            outgoing_particles[..., [0, 2]] -= (
                dz_exit * particle_momenta[..., [0, 1]] / pz_exit
            )
            len_patch = (
                self.length.unsqueeze(-1)
                if self.length.ndim > 0
                else self.length
            )
            outgoing_particles[..., 4] += (
                dz_exit.squeeze(-1) * rel_p / pz_exit.squeeze(-1) + len_patch
            )

        return ParticleBeam(
            particles=outgoing_particles,
            energy=incoming.energy + self.energy_offset,
            particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities,
            s=incoming.s + self.length,
            species=incoming.species,
        )

    def first_order_transfer_map(
        self,
        energy: torch.Tensor | None = None,
        species: Species | None = None,
    ) -> torch.Tensor:
        """
        Compute the first-order transfer map for the patch element.

        :param energy: Energy of the beam in eV.
        :param species: Particle species of the beam.
        :return: First-order 7x7 transfer map.
        """
        return patch_transfer_map(
            offset=self.offset,
            pitch=self.pitch,
            tilt=self.tilt,
            time_offset=self.time_offset,
            drift_to_exit=self.drift_to_exit,
        )

    def _transform_parameters(self, incoming: ParameterBeam) -> ParameterBeam:
        """Transform a `ParameterBeam` through the `Patch` element."""
        tm = self.first_order_transfer_map(incoming.energy, incoming.species)
        new_mu = (tm @ incoming.mu.unsqueeze(-1)).squeeze(-1)
        new_cov = tm @ incoming.cov @ tm.mT

        return ParameterBeam(
            mu=new_mu,
            cov=new_cov,
            energy=incoming.energy + self.energy_offset,
            total_charge=incoming.total_charge,
            s=incoming.s + self.length,
            species=incoming.species,
        )

    def _rotation_matrix(self) -> torch.Tensor:
        """
        Compute the rotation matrix for the patch element based on its pitch and tilt.
        """
        return patch_rotation_matrix(self.pitch, self.tilt)

    @property
    def length(self) -> torch.Tensor:
        return (self._rotation_matrix()[..., :, 2] * self.offset).sum(dim=-1)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "offset",
            "time_offset",
            "pitch",
            "tilt",
            "energy_offset",
            "energy_setpoint",
            "drift_to_exit",
        ]

    @property
    def is_skippable(self) -> bool:
        return False  # Patch elements cannot be skipped

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        pass  # Patch element does not have a visual representation
