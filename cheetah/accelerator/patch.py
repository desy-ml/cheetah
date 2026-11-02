import torch
from matplotlib import pyplot as plt
from scipy.constants import speed_of_light

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParameterBeam, ParticleBeam, Species
from cheetah.utils import cache_transfer_map, compute_relativistic_factors


class Patch(Element):
    """
    `Patch` element that shifts the reference orbit and time.

    :param position_offset: Exit face spatial displacement `(x, y, z)` from the entrance
        in meters.
    :param pitch: Exit face pitch angles `(x_pitch, y_pitch)` in radians.
    :param tilt: Tilt angle in the x-y plane in radians.
    :param time_offset: Reference time offset in seconds.
    :param energy_offset: Reference energy offset in eV.
    :param drift_to_exit: Whether to drift particles to the exit face plane.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment. If `None` (default), a warning is raised for
        invalid names. Set to `True` to sanitise, or `False` to silence the warning.
    :param metadata: Dictionary of arbitrary, serialisable annotations attached to the
        element.
    :param device: Device on which to create the element's tensors.
    :param dtype: Data type of the element's tensors.
    """

    def __init__(
        self,
        position_offset: torch.Tensor | None = None,
        pitch: torch.Tensor | None = None,
        tilt: torch.Tensor | None = None,
        time_offset: torch.Tensor | None = None,
        energy_offset: torch.Tensor | None = None,
        drift_to_exit: bool = True,
        name: str | None = None,
        sanitize_name: bool | None = None,
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
            "position_offset",
            (
                position_offset
                if position_offset is not None
                else torch.tensor([0.0, 0.0, 0.0], **factory_kwargs)
            ),
        )
        self.register_buffer_or_parameter(
            "pitch",
            (
                pitch
                if pitch is not None
                else torch.tensor([0.0, 0.0], **factory_kwargs)
            ),
        )
        self.register_buffer_or_parameter(
            "tilt", tilt if tilt is not None else torch.tensor(0.0, **factory_kwargs)
        )
        self.register_buffer_or_parameter(
            "time_offset",
            (
                time_offset
                if time_offset is not None
                else torch.tensor(0.0, **factory_kwargs)
            ),
        )
        self.register_buffer_or_parameter(
            "energy_offset",
            (
                energy_offset
                if energy_offset is not None
                else torch.tensor(0.0, **factory_kwargs)
            ),
        )

    @property
    def is_active(self) -> bool:
        """Whether the `Patch` has any non-zero offsets or rotation angles."""
        return (
            (self.position_offset != 0.0).any()
            or (self.pitch != 0.0).any()
            or (self.tilt != 0.0).any()
            or (self.time_offset != 0.0).any()
            or (self.energy_offset != 0.0).any()
        ).item()

    @property
    def is_skippable(self) -> bool:
        """Whether the `Patch` can be skipped during tracking."""
        return not self.is_active

    @property
    def length(self) -> torch.Tensor:
        return (self._rotation_matrix()[..., :, 2] * self.position_offset).sum(dim=-1)

    @property
    def vector_shape(self) -> torch.Size:
        """
        Broadcasting vector shape of the `Patch` element across all its features.
        """
        return torch.broadcast_shapes(
            self.position_offset.shape[:-1],
            self.pitch.shape[:-1],
            self.tilt.shape,
            self.time_offset.shape,
            self.energy_offset.shape,
        )

    @cache_transfer_map
    def first_order_transfer_map(
        self,
        energy: torch.Tensor,
        species: Species,
    ) -> torch.Tensor:
        """
        Compute the first-order transfer map for the `Patch` element.

        :param energy: Incoming reference energy in eV.
        :param species: Particle species of the beam.
        :return: First-order 7x7 transfer map.
        """
        vector_shape = torch.broadcast_shapes(self.vector_shape, energy.shape)
        factory_kwargs = {
            "device": self.position_offset.device,
            "dtype": self.position_offset.dtype,
        }

        rotation_matrix_inverse = self._rotation_matrix().mT

        reference_exit = -(
            rotation_matrix_inverse @ self.position_offset.unsqueeze(-1)
        ).squeeze(-1)
        reference_x_exit = reference_exit[..., 0]
        reference_y_exit = reference_exit[..., 1]
        reference_z = reference_exit[..., 2]

        reference_momentum = rotation_matrix_inverse[..., :, 2]
        reference_px = reference_momentum[..., 0]
        reference_py = reference_momentum[..., 1]
        reference_pz = reference_momentum[..., 2]

        transfer_map = torch.eye(7, **factory_kwargs).repeat(*vector_shape, 1, 1)

        transfer_map[..., 1, 1] = rotation_matrix_inverse[..., 0, 0]
        transfer_map[..., 1, 3] = rotation_matrix_inverse[..., 0, 1]
        transfer_map[..., 1, 5] = rotation_matrix_inverse[..., 0, 2]

        transfer_map[..., 3, 1] = rotation_matrix_inverse[..., 1, 0]
        transfer_map[..., 3, 3] = rotation_matrix_inverse[..., 1, 1]
        transfer_map[..., 3, 5] = rotation_matrix_inverse[..., 1, 2]

        transfer_map[..., 5, 5] = 1.0
        transfer_map[..., 1, 6] = reference_px
        transfer_map[..., 3, 6] = reference_py

        if self.drift_to_exit:
            reference_pz_squared = reference_pz.square()

            for column, delta_r_index, delta_p_index, delta_delta in [
                (0, 0, None, 0.0),
                (1, None, 0, 0.0),
                (2, 1, None, 0.0),
                (3, None, 1, 0.0),
                (4, None, None, 0.0),
                (5, None, 2, 1.0),
            ]:
                delta_r = (
                    rotation_matrix_inverse[..., :, delta_r_index]
                    if delta_r_index is not None
                    else torch.zeros_like(reference_momentum)
                )
                delta_p = (
                    rotation_matrix_inverse[..., :, delta_p_index]
                    if delta_p_index is not None
                    else torch.zeros_like(reference_momentum)
                )

                term_rx = (
                    delta_r[..., 0] - (reference_px / reference_pz) * delta_r[..., 2]
                )
                term_px = -reference_z * (
                    delta_p[..., 0] / reference_pz
                    - reference_px * delta_p[..., 2] / reference_pz_squared
                )
                transfer_map[..., 0, column] = term_rx + term_px

                term_ry = (
                    delta_r[..., 1] - (reference_py / reference_pz) * delta_r[..., 2]
                )
                term_py = -reference_z * (
                    delta_p[..., 1] / reference_pz
                    - reference_py * delta_p[..., 2] / reference_pz_squared
                )
                transfer_map[..., 2, column] = term_ry + term_py

                delta_tau = 1.0 if column == 4 else 0.0
                term_tau = (
                    delta_tau
                    + delta_r[..., 2] / reference_pz
                    + reference_z
                    * (
                        delta_delta / reference_pz
                        - delta_p[..., 2] / reference_pz_squared
                    )
                )
                transfer_map[..., 4, column] = term_tau

            transfer_map[..., 0, 6] = (
                reference_x_exit - reference_z * reference_px / reference_pz
            )
            transfer_map[..., 2, 6] = (
                reference_y_exit - reference_z * reference_py / reference_pz
            )
            transfer_map[..., 4, 6] = (
                self.time_offset * speed_of_light
                + reference_z / reference_pz
                + self.length
            )
        else:
            transfer_map[..., 0, 0] = rotation_matrix_inverse[..., 0, 0]
            transfer_map[..., 0, 2] = rotation_matrix_inverse[..., 0, 1]
            transfer_map[..., 2, 0] = rotation_matrix_inverse[..., 1, 0]
            transfer_map[..., 2, 2] = rotation_matrix_inverse[..., 1, 1]
            transfer_map[..., 0, 6] = reference_x_exit
            transfer_map[..., 2, 6] = reference_y_exit
            transfer_map[..., 4, 6] = self.time_offset * speed_of_light

        if (self.energy_offset != 0.0).any():
            outgoing_energy = energy + self.energy_offset
            _, _, incoming_beta = compute_relativistic_factors(energy, species.mass_eV)
            _, _, outgoing_beta = compute_relativistic_factors(
                outgoing_energy, species.mass_eV
            )
            momentum_ratio = (energy * incoming_beta) / (
                outgoing_energy * outgoing_beta
            )
            momentum_ratio_matrix = momentum_ratio[..., None, None]
            transfer_map[..., [1, 3], :] = (
                transfer_map[..., [1, 3], :] * momentum_ratio_matrix
            )
            transfer_map[..., 5, 5] = momentum_ratio
            transfer_map[..., 5, 6] = momentum_ratio - 1.0

        return transfer_map

    def track(self, incoming: Beam) -> Beam:
        """
        Track particles through the `Patch` element.

        :param incoming: Beam entering the element.
        :return: Beam exiting the element.
        """
        if isinstance(incoming, ParticleBeam):
            return self._track_particle_beam(incoming)
        elif isinstance(incoming, ParameterBeam):
            return self._track_parameter_beam(incoming)
        else:
            raise TypeError(f"Parameter incoming is of invalid type {type(incoming)}")

    def _track_particle_beam(self, incoming: ParticleBeam) -> ParticleBeam:
        """Track a `ParticleBeam` through the `Patch` element."""
        # Momentum coordinates
        relative_momentum = incoming.p + 1.0  # Convert delta to p / p0
        pz = (
            relative_momentum.square() - incoming.px.square() - incoming.py.square()
        ).sqrt()
        particle_momenta = torch.stack([incoming.px, incoming.py, pz], dim=-1)

        rotation_matrix = self._rotation_matrix()
        shifted_positions = torch.stack(
            [incoming.x, incoming.y, torch.zeros_like(incoming.x)], dim=-1
        ) - self.position_offset.unsqueeze(-2)

        # Transform coordinates to exit frame: (r_entrance - offset) @ rotation_matrix
        particle_positions = shifted_positions @ rotation_matrix
        particle_momenta = particle_momenta @ rotation_matrix

        vector_shape = torch.broadcast_shapes(
            self.vector_shape, incoming.particles.shape[:-2], incoming.energy.shape
        )
        outgoing_particles = incoming.particles.expand(
            *vector_shape, incoming.num_particles, 7
        ).clone()

        outgoing_particles[..., [0, 2]] = particle_positions[..., [0, 1]]
        outgoing_particles[..., [1, 3]] = particle_momenta[..., [0, 1]]
        time_offset_z = (self.time_offset * speed_of_light).unsqueeze(-1)
        outgoing_particles[..., 4] = incoming.tau + time_offset_z

        length = self.length
        # Track particles to the end of the patch
        if self.drift_to_exit:
            delta_z_exit = particle_positions[..., 2:3]
            pz_exit = particle_momenta[..., 2:3]
            drift_factor = delta_z_exit / pz_exit
            outgoing_particles[..., [0, 2]] -= (
                drift_factor * particle_momenta[..., [0, 1]]
            )
            patch_length = length.unsqueeze(-1) if length.ndim > 0 else length
            outgoing_particles[..., 4] += (
                drift_factor.squeeze(-1) * relative_momentum + patch_length
            )

        outgoing_energy = (
            incoming.energy + self.energy_offset
            if (self.energy_offset != 0.0).any()
            else incoming.energy
        )
        if (self.energy_offset != 0.0).any():
            _, _, incoming_beta = compute_relativistic_factors(
                incoming.energy, incoming.species.mass_eV
            )
            _, _, outgoing_beta = compute_relativistic_factors(
                outgoing_energy, incoming.species.mass_eV
            )
            momentum_ratio = (incoming.energy * incoming_beta) / (
                outgoing_energy * outgoing_beta
            )
            momentum_ratio_particles = momentum_ratio[..., None, None]
            outgoing_particles[..., [1, 3]] = (
                outgoing_particles[..., [1, 3]] * momentum_ratio_particles
            )
            outgoing_particles[..., 5] = outgoing_particles[
                ..., 5
            ] * momentum_ratio_particles.squeeze(-1) + (
                momentum_ratio_particles.squeeze(-1) - 1.0
            )

        return ParticleBeam(
            particles=outgoing_particles,
            energy=outgoing_energy,
            particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities,
            s=incoming.s + length,
            species=incoming.species.clone(),
        )

    def _track_parameter_beam(self, incoming: ParameterBeam) -> ParameterBeam:
        """Track a `ParameterBeam` through the `Patch` element."""
        transfer_map = self.first_order_transfer_map(incoming.energy, incoming.species)
        outgoing_mu = (transfer_map @ incoming.mu.unsqueeze(-1)).squeeze(-1)
        outgoing_cov = transfer_map @ incoming.cov @ transfer_map.mT

        outgoing_energy = (
            incoming.energy + self.energy_offset
            if (self.energy_offset != 0.0).any()
            else incoming.energy
        )

        return ParameterBeam(
            mu=outgoing_mu,
            cov=outgoing_cov,
            energy=outgoing_energy,
            total_charge=incoming.total_charge,
            s=incoming.s + self.length,
            species=incoming.species.clone(),
        )

    def _rotation_matrix(self) -> torch.Tensor:
        """
        Compute the rotation matrix for the `Patch` element based on its pitch and tilt.
        """
        vector_shape = torch.broadcast_shapes(self.pitch.shape[:-1], self.tilt.shape)

        cos_x = self.pitch[..., 0].cos()
        sin_x = self.pitch[..., 0].sin()
        cos_y = self.pitch[..., 1].cos()
        sin_y = self.pitch[..., 1].sin()
        cos_z = self.tilt.cos()
        sin_z = self.tilt.sin()

        rotation_matrix = self.pitch.new_zeros((*vector_shape, 3, 3))
        rotation_matrix[..., 0, 0] = cos_x * cos_z - sin_x * sin_y * sin_z
        rotation_matrix[..., 0, 1] = -cos_x * sin_z - sin_x * sin_y * cos_z
        rotation_matrix[..., 0, 2] = sin_x * cos_y

        rotation_matrix[..., 1, 0] = cos_y * sin_z
        rotation_matrix[..., 1, 1] = cos_y * cos_z
        rotation_matrix[..., 1, 2] = sin_y

        rotation_matrix[..., 2, 0] = -sin_x * cos_z - cos_x * sin_y * sin_z
        rotation_matrix[..., 2, 1] = sin_x * sin_z - cos_x * sin_y * cos_z
        rotation_matrix[..., 2, 2] = cos_x * cos_y

        return rotation_matrix

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        """
        Plot a representation of this element into a `matplotlib` Axes at position `s`.
        """
        ax = ax or plt.subplot(111)
        return ax

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "position_offset",
            "pitch",
            "tilt",
            "time_offset",
            "energy_offset",
            "drift_to_exit",
        ]
