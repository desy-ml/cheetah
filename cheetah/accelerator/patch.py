import torch
from matplotlib import pyplot as plt
from scipy.constants import speed_of_light

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParameterBeam, ParticleBeam, Species
from cheetah.utils import cache_transfer_map, compute_relativistic_factors


class Patch(Element):
    """
    Patch element that shifts the reference orbit and time.

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
                self.length.unsqueeze(-1) if self.length.ndim > 0 else self.length
            )
            outgoing_particles[..., 4] += (
                dz_exit.squeeze(-1) * rel_p / pz_exit.squeeze(-1) + len_patch
            )

        if torch.any(self.energy_setpoint > 0.0) or torch.any(
            self.energy_offset != 0.0
        ):
            outgoing_energy = torch.where(
                self.energy_setpoint > 0.0,
                self.energy_setpoint,
                incoming.energy + self.energy_offset,
            )
            _, _, beta0 = compute_relativistic_factors(
                incoming.energy, incoming.species.mass_eV
            )
            _, _, beta1 = compute_relativistic_factors(
                outgoing_energy, incoming.species.mass_eV
            )
            p_ratio = (incoming.energy * beta0) / (outgoing_energy * beta1)
            p_ratio_particles = p_ratio[..., None, None]
            outgoing_particles[..., [1, 3]] = (
                outgoing_particles[..., [1, 3]] * p_ratio_particles
            )
            outgoing_particles[..., 5] = outgoing_particles[
                ..., 5
            ] * p_ratio_particles.squeeze(-1) + (p_ratio_particles.squeeze(-1) - 1.0)
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
        Compute the first-order transfer map for the patch element.

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

        R = self._rotation_matrix()
        RT = R.mT

        p0 = RT[..., :, 2]
        px0, py0, pz0 = p0[..., 0], p0[..., 1], p0[..., 2]

        z0 = -(
            RT[..., 2, 0] * offset[..., 0]
            + RT[..., 2, 1] * offset[..., 1]
            + RT[..., 2, 2] * offset[..., 2]
        )
        x_exit0 = -(
            RT[..., 0, 0] * offset[..., 0]
            + RT[..., 0, 1] * offset[..., 1]
            + RT[..., 0, 2] * offset[..., 2]
        )
        y_exit0 = -(
            RT[..., 1, 0] * offset[..., 0]
            + RT[..., 1, 1] * offset[..., 1]
            + RT[..., 1, 2] * offset[..., 2]
        )

        tm = torch.eye(7, **factory_kwargs).repeat(*broadcast_batch, 1, 1)

        tm[..., 1, 1] = RT[..., 0, 0]
        tm[..., 1, 3] = RT[..., 0, 1]
        tm[..., 1, 5] = RT[..., 0, 2]

        tm[..., 3, 1] = RT[..., 1, 0]
        tm[..., 3, 3] = RT[..., 1, 1]
        tm[..., 3, 5] = RT[..., 1, 2]

        tm[..., 5, 5] = 1.0

        if self.drift_to_exit:
            pz0_safe = pz0.clamp(min=1e-12)
            pz0_sq = pz0_safe.square()

            for col, dr_idx, dp_idx, ddelta in [
                (0, 0, None, 0.0),
                (1, None, 0, 0.0),
                (2, 1, None, 0.0),
                (3, None, 1, 0.0),
                (4, None, None, 0.0),
                (5, None, 2, 1.0),
            ]:
                dr = RT[..., :, dr_idx] if dr_idx is not None else torch.zeros_like(p0)
                dp = RT[..., :, dp_idx] if dp_idx is not None else torch.zeros_like(p0)

                term_rx = dr[..., 0] - (px0 / pz0_safe) * dr[..., 2]
                term_px = -z0 * (dp[..., 0] / pz0_safe - px0 * dp[..., 2] / pz0_sq)
                tm[..., 0, col] = term_rx + term_px

                term_ry = dr[..., 1] - (py0 / pz0_safe) * dr[..., 2]
                term_py = -z0 * (dp[..., 1] / pz0_safe - py0 * dp[..., 2] / pz0_sq)
                tm[..., 2, col] = term_ry + term_py

                dtau = 1.0 if col == 4 else 0.0
                term_tau = (
                    dtau
                    + dr[..., 2] / pz0_safe
                    + z0 * (ddelta / pz0_safe - dp[..., 2] / pz0_sq)
                )
                tm[..., 4, col] = term_tau

            tm[..., 0, 6] = x_exit0 - z0 * px0 / pz0_safe
            tm[..., 1, 6] = px0
            tm[..., 2, 6] = y_exit0 - z0 * py0 / pz0_safe
            tm[..., 3, 6] = py0
            tm[..., 4, 6] = time_offset * speed_of_light + z0 / pz0_safe + self.length
        else:
            tm[..., 0, 0] = RT[..., 0, 0]
            tm[..., 0, 2] = RT[..., 0, 1]
            tm[..., 2, 0] = RT[..., 1, 0]
            tm[..., 2, 2] = RT[..., 1, 1]
            tm[..., 0, 6] = x_exit0
            tm[..., 1, 6] = px0
            tm[..., 2, 6] = y_exit0
            tm[..., 3, 6] = py0
            tm[..., 4, 6] = time_offset * speed_of_light

        if torch.any(self.energy_setpoint > 0.0) or torch.any(
            self.energy_offset != 0.0
        ):
            outgoing_energy = torch.where(
                self.energy_setpoint > 0.0,
                self.energy_setpoint,
                energy + self.energy_offset,
            )
            _, _, beta0 = compute_relativistic_factors(energy, species.mass_eV)
            _, _, beta1 = compute_relativistic_factors(outgoing_energy, species.mass_eV)
            p_ratio = (energy * beta0) / (outgoing_energy * beta1)
            p_ratio_matrix = p_ratio[..., None, None]
            tm[..., [1, 3], :] = tm[..., [1, 3], :] * p_ratio_matrix
            tm[..., 5, 5] = p_ratio
            tm[..., 5, 6] = p_ratio - 1.0

        return tm

    def _transform_parameters(self, incoming: ParameterBeam) -> ParameterBeam:
        """Transform a `ParameterBeam` through the `Patch` element."""
        tm = self.first_order_transfer_map(incoming.energy, incoming.species)
        new_mu = (tm @ incoming.mu.unsqueeze(-1)).squeeze(-1)
        new_cov = tm @ incoming.cov @ tm.mT

        if torch.any(self.energy_setpoint > 0.0) or torch.any(
            self.energy_offset != 0.0
        ):
            outgoing_energy = torch.where(
                self.energy_setpoint > 0.0,
                self.energy_setpoint,
                incoming.energy + self.energy_offset,
            )
        else:
            outgoing_energy = incoming.energy

        return ParameterBeam(
            mu=new_mu,
            cov=new_cov,
            energy=outgoing_energy,
            total_charge=incoming.total_charge,
            s=incoming.s + self.length,
            species=incoming.species,
        )

    def _rotation_matrix(self) -> torch.Tensor:
        """
        Compute the rotation matrix for the patch element based on its pitch and tilt.
        """
        broadcast_batch = torch.broadcast_shapes(self.pitch.shape[:-1], self.tilt.shape)
        factory_kwargs = {"device": self.pitch.device, "dtype": self.pitch.dtype}

        pitch = self.pitch
        tilt = self.tilt
        if broadcast_batch:
            pitch = pitch.expand(*broadcast_batch, 2)
            tilt = tilt.expand(broadcast_batch)

        rotation_y = torch.zeros((*broadcast_batch, 3, 3), **factory_kwargs)
        rotation_y[..., 0, 0] = pitch[..., 0].cos()
        rotation_y[..., 0, 2] = pitch[..., 0].sin()
        rotation_y[..., 1, 1] = 1.0
        rotation_y[..., 2, 0] = -pitch[..., 0].sin()
        rotation_y[..., 2, 2] = pitch[..., 0].cos()

        rotation_neg_x = torch.zeros((*broadcast_batch, 3, 3), **factory_kwargs)
        rotation_neg_x[..., 0, 0] = 1.0
        rotation_neg_x[..., 1, 1] = pitch[..., 1].cos()
        rotation_neg_x[..., 1, 2] = pitch[..., 1].sin()
        rotation_neg_x[..., 2, 1] = -pitch[..., 1].sin()
        rotation_neg_x[..., 2, 2] = pitch[..., 1].cos()

        rotation_z = torch.zeros((*broadcast_batch, 3, 3), **factory_kwargs)
        rotation_z[..., 0, 0] = tilt.cos()
        rotation_z[..., 0, 1] = -tilt.sin()
        rotation_z[..., 1, 0] = tilt.sin()
        rotation_z[..., 1, 1] = tilt.cos()
        rotation_z[..., 2, 2] = 1.0

        return rotation_y @ rotation_neg_x @ rotation_z

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
