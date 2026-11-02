import torch
from matplotlib import pyplot as plt
from scipy.constants import speed_of_light

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParameterBeam, ParticleBeam, Species
from cheetah.utils import cache_transfer_map, compute_relativistic_factors


class Patch(Element):
    """
    `Patch` element that shifts the reference orbit and time.

    :param offset: Exit face offset `(x, y, z)` from the entrance in meters.
    :param time_offset: Reference time offset in seconds.
    :param pitch: Exit face pitch angles `(x_pitch, y_pitch)` in radians.
    :param tilt: Tilt angle in the x-y plane in radians.
    :param energy_offset: Reference energy offset in eV.
    :param energy_setpoint: Target reference energy setpoint in eV.
    :param drift_to_exit: Whether to drift particles to the exit face plane.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment.
    :param metadata: Dictionary of arbitrary, serialisable annotations attached to the
        element.
    :param device: Device on which to create the element's tensors.
    :param dtype: Data type of the element's tensors.
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
        relative_momentum = incoming.p + 1.0  # Convert delta to p / p0
        pz_squared = (
            relative_momentum.square() - incoming.px.square() - incoming.py.square()
        )
        pz = pz_squared.clamp(min=0.0).sqrt()
        particle_momenta = torch.stack(
            [
                incoming.px,
                incoming.py,
                pz,
            ],
            dim=-1,
        )

        rotation_matrix = self._rotation_matrix()
        offset_x = self.offset[..., 0].unsqueeze(-1)
        offset_y = self.offset[..., 1].unsqueeze(-1)
        offset_z = self.offset[..., 2].unsqueeze(-1)

        delta_x = incoming.x - offset_x
        delta_y = incoming.y - offset_y
        delta_z_entrance = -offset_z.expand_as(delta_x)
        shifted_positions = torch.stack([delta_x, delta_y, delta_z_entrance], dim=-1)

        # Transform coordinates to exit frame: (r_entrance - offset) @ rotation_matrix
        particle_positions = shifted_positions @ rotation_matrix
        particle_momenta = particle_momenta @ rotation_matrix

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
            delta_z_exit = particle_positions[..., 2:3]
            pz_exit = particle_momenta[..., 2:3]
            outgoing_particles[..., [0, 2]] -= (
                delta_z_exit * particle_momenta[..., [0, 1]] / pz_exit
            )
            patch_length = (
                self.length.unsqueeze(-1) if self.length.ndim > 0 else self.length
            )
            outgoing_particles[..., 4] += (
                delta_z_exit.squeeze(-1) * relative_momentum / pz_exit.squeeze(-1)
                + patch_length
            )

        if (self.energy_setpoint > 0.0).any() or (self.energy_offset != 0.0).any():
            outgoing_energy = self.energy_setpoint.where(
                self.energy_setpoint > 0.0,
                incoming.energy + self.energy_offset,
            )
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
        else:
            outgoing_energy = incoming.energy

        return ParticleBeam(
            particles=outgoing_particles,
            energy=outgoing_energy,
            particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities,
            s=incoming.s + self.length,
            species=incoming.species,
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
        broadcast_batch = torch.broadcast_shapes(
            self.offset.shape[:-1],
            self.pitch.shape[:-1],
            self.tilt.shape,
            self.time_offset.shape,
        )
        factory_kwargs = {"device": self.offset.device, "dtype": self.offset.dtype}

        offset = self.offset
        time_offset = self.time_offset
        if broadcast_batch:
            offset = offset.expand(*broadcast_batch, 3)
            time_offset = time_offset.expand(broadcast_batch)

        rotation_matrix = self._rotation_matrix()
        rotation_matrix_inverse = rotation_matrix.mT

        reference_momentum = rotation_matrix_inverse[..., :, 2]
        reference_px = reference_momentum[..., 0]
        reference_py = reference_momentum[..., 1]
        reference_pz = reference_momentum[..., 2]

        reference_z = -(
            rotation_matrix_inverse[..., 2, 0] * offset[..., 0]
            + rotation_matrix_inverse[..., 2, 1] * offset[..., 1]
            + rotation_matrix_inverse[..., 2, 2] * offset[..., 2]
        )
        reference_x_exit = -(
            rotation_matrix_inverse[..., 0, 0] * offset[..., 0]
            + rotation_matrix_inverse[..., 0, 1] * offset[..., 1]
            + rotation_matrix_inverse[..., 0, 2] * offset[..., 2]
        )
        reference_y_exit = -(
            rotation_matrix_inverse[..., 1, 0] * offset[..., 0]
            + rotation_matrix_inverse[..., 1, 1] * offset[..., 1]
            + rotation_matrix_inverse[..., 1, 2] * offset[..., 2]
        )

        transfer_map = torch.eye(7, **factory_kwargs).repeat(*broadcast_batch, 1, 1)

        transfer_map[..., 1, 1] = rotation_matrix_inverse[..., 0, 0]
        transfer_map[..., 1, 3] = rotation_matrix_inverse[..., 0, 1]
        transfer_map[..., 1, 5] = rotation_matrix_inverse[..., 0, 2]

        transfer_map[..., 3, 1] = rotation_matrix_inverse[..., 1, 0]
        transfer_map[..., 3, 3] = rotation_matrix_inverse[..., 1, 1]
        transfer_map[..., 3, 5] = rotation_matrix_inverse[..., 1, 2]

        transfer_map[..., 5, 5] = 1.0

        if self.drift_to_exit:
            reference_pz_safe = reference_pz.clamp(min=1e-12)
            reference_pz_squared = reference_pz_safe.square()

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
                    delta_r[..., 0]
                    - (reference_px / reference_pz_safe) * delta_r[..., 2]
                )
                term_px = -reference_z * (
                    delta_p[..., 0] / reference_pz_safe
                    - reference_px * delta_p[..., 2] / reference_pz_squared
                )
                transfer_map[..., 0, column] = term_rx + term_px

                term_ry = (
                    delta_r[..., 1]
                    - (reference_py / reference_pz_safe) * delta_r[..., 2]
                )
                term_py = -reference_z * (
                    delta_p[..., 1] / reference_pz_safe
                    - reference_py * delta_p[..., 2] / reference_pz_squared
                )
                transfer_map[..., 2, column] = term_ry + term_py

                delta_tau = 1.0 if column == 4 else 0.0
                term_tau = (
                    delta_tau
                    + delta_r[..., 2] / reference_pz_safe
                    + reference_z
                    * (
                        delta_delta / reference_pz_safe
                        - delta_p[..., 2] / reference_pz_squared
                    )
                )
                transfer_map[..., 4, column] = term_tau

            transfer_map[..., 0, 6] = (
                reference_x_exit - reference_z * reference_px / reference_pz_safe
            )
            transfer_map[..., 1, 6] = reference_px
            transfer_map[..., 2, 6] = (
                reference_y_exit - reference_z * reference_py / reference_pz_safe
            )
            transfer_map[..., 3, 6] = reference_py
            transfer_map[..., 4, 6] = (
                time_offset * speed_of_light
                + reference_z / reference_pz_safe
                + self.length
            )
        else:
            transfer_map[..., 0, 0] = rotation_matrix_inverse[..., 0, 0]
            transfer_map[..., 0, 2] = rotation_matrix_inverse[..., 0, 1]
            transfer_map[..., 2, 0] = rotation_matrix_inverse[..., 1, 0]
            transfer_map[..., 2, 2] = rotation_matrix_inverse[..., 1, 1]
            transfer_map[..., 0, 6] = reference_x_exit
            transfer_map[..., 1, 6] = reference_px
            transfer_map[..., 2, 6] = reference_y_exit
            transfer_map[..., 3, 6] = reference_py
            transfer_map[..., 4, 6] = time_offset * speed_of_light

        if (self.energy_setpoint > 0.0).any() or (self.energy_offset != 0.0).any():
            outgoing_energy = self.energy_setpoint.where(
                self.energy_setpoint > 0.0,
                energy + self.energy_offset,
            )
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

    def _transform_parameters(self, incoming: ParameterBeam) -> ParameterBeam:
        """Transform a `ParameterBeam` through the `Patch` element."""
        transfer_map = self.first_order_transfer_map(incoming.energy, incoming.species)
        outgoing_mu = (transfer_map @ incoming.mu.unsqueeze(-1)).squeeze(-1)
        outgoing_cov = transfer_map @ incoming.cov @ transfer_map.mT

        if (self.energy_setpoint > 0.0).any() or (self.energy_offset != 0.0).any():
            outgoing_energy = self.energy_setpoint.where(
                self.energy_setpoint > 0.0,
                incoming.energy + self.energy_offset,
            )
        else:
            outgoing_energy = incoming.energy

        return ParameterBeam(
            mu=outgoing_mu,
            cov=outgoing_cov,
            energy=outgoing_energy,
            total_charge=incoming.total_charge,
            s=incoming.s + self.length,
            species=incoming.species,
        )

    def _rotation_matrix(self) -> torch.Tensor:
        """
        Compute the rotation matrix for the `Patch` element based on its pitch and tilt.
        """
        broadcast_batch = torch.broadcast_shapes(self.pitch.shape[:-1], self.tilt.shape)

        pitch = self.pitch
        tilt = self.tilt
        if broadcast_batch:
            pitch = pitch.expand(*broadcast_batch, 2)
            tilt = tilt.expand(broadcast_batch)

        rotation_y = pitch.new_zeros((*broadcast_batch, 3, 3))
        rotation_y[..., 0, 0] = pitch[..., 0].cos()
        rotation_y[..., 0, 2] = pitch[..., 0].sin()
        rotation_y[..., 1, 1] = 1.0
        rotation_y[..., 2, 0] = -pitch[..., 0].sin()
        rotation_y[..., 2, 2] = pitch[..., 0].cos()

        rotation_negative_x = pitch.new_zeros((*broadcast_batch, 3, 3))
        rotation_negative_x[..., 0, 0] = 1.0
        rotation_negative_x[..., 1, 1] = pitch[..., 1].cos()
        rotation_negative_x[..., 1, 2] = pitch[..., 1].sin()
        rotation_negative_x[..., 2, 1] = -pitch[..., 1].sin()
        rotation_negative_x[..., 2, 2] = pitch[..., 1].cos()

        rotation_z = tilt.new_zeros((*broadcast_batch, 3, 3))
        rotation_z[..., 0, 0] = tilt.cos()
        rotation_z[..., 0, 1] = -tilt.sin()
        rotation_z[..., 1, 0] = tilt.sin()
        rotation_z[..., 1, 1] = tilt.cos()
        rotation_z[..., 2, 2] = 1.0

        return rotation_y @ rotation_negative_x @ rotation_z

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
        ax = ax or plt.subplot(111)
        return ax
