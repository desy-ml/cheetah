import torch
from matplotlib import pyplot as plt
from scipy.constants import speed_of_light

from cheetah.accelerator.element import Element
from cheetah.particles import Beam
from cheetah.particles.parameter_beam import ParameterBeam
from cheetah.particles.particle_beam import ParticleBeam
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

    def _transform_parameters(self, incoming: ParameterBeam) -> ParameterBeam:
        """
        Transform a ParameterBeam through the Patch element.

        The Patch applies linear transformations to the 7D phase space vector.
        For Gaussian beams, the mean and covariance transform as:
            mu' = A @ mu + b
            cov' = A @ cov @ A^T
        """
        # Get the rotation matrix (applies to positions and momenta)
        rotation_matrix = self._rotation_matrix()  # Shape: (..., 3, 3)

        # Build the full 7x7 transformation matrix for phase space
        # The 7th dimension (always 1) doesn't transform
        batch_shape = rotation_matrix.shape[:-2]  # Get batch shape from rotation matrix
        A = torch.eye(7, dtype=incoming.mu.dtype, device=incoming.mu.device)
        A = A.expand(*batch_shape, 7, 7).clone()  # Shape: (..., 7, 7)
        position_idx = torch.tensor([0, 2, 4], device=incoming.mu.device)  # x, y, tau
        momentum_idx = torch.tensor([1, 3, 5], device=incoming.mu.device)  # px, py, p
        for i in range(len(position_idx)):
            for j in range(len(momentum_idx)):
                A[..., position_idx[i], position_idx[j]] = rotation_matrix[..., i, j]
                A[..., momentum_idx[i], momentum_idx[j]] = rotation_matrix[..., i, j]

        # Build the offset vector [dx, dpx, dy, dpy, d_tau, dp, 0]
        b = torch.zeros(7, dtype=incoming.mu.dtype, device=incoming.mu.device)
        b = b.expand(*batch_shape, 7).clone()
        b[..., 0] = -self.offset[0]  # x offset
        b[..., 2] = -self.offset[1]  # y offset
        b[..., 4] = (
            self.time_offset * speed_of_light
        )  # time offset to longitudinal offset
        # Energy offset: convert to dimensionless momentum offset
        b[..., 5] = self.energy_offset / incoming.energy

        # Transform mean and covariance using linear transformation
        mu_transformed = torch.matmul(A, incoming.mu.unsqueeze(-1)).squeeze(-1) + b
        cov_transformed = torch.matmul(
            torch.matmul(A, incoming.cov), A.transpose(-2, -1)
        )

        # Handle drift to exit (longitudinal drift)
        if self.drift_to_exit:
            # Add the drift length as an offset in tau (longitudinal position)
            mu_transformed[..., 4] = mu_transformed[..., 4] + self.length

        return ParameterBeam(
            mu=mu_transformed,
            cov=cov_transformed,
            energy=incoming.energy + self.energy_offset,
            total_charge=incoming.total_charge,
            s=incoming.s + self.length,
            species=incoming.species,
        )

    def _rotation_matrix(self) -> torch.Tensor:
        """
        Computes the rotation matrix for the patch element based on its pitch and tilt.
        """
        rotation_y = self.pitch.new_zeros((*self.pitch.shape[:-1], 3, 3))
        rotation_y[..., 0, 0] = self.pitch[..., 0].cos()
        rotation_y[..., 0, 2] = self.pitch[..., 0].sin()
        rotation_y[..., 1, 1] = 1.0
        rotation_y[..., 2, 0] = -self.pitch[..., 0].sin()
        rotation_y[..., 2, 2] = self.pitch[..., 0].cos()

        rotation_neg_x = self.pitch.new_zeros((*self.pitch.shape[:-1], 3, 3))
        rotation_neg_x[..., 0, 0] = 1.0
        rotation_neg_x[..., 1, 1] = self.pitch[..., 1].cos()
        rotation_neg_x[..., 1, 2] = self.pitch[..., 1].sin()
        rotation_neg_x[..., 2, 1] = -self.pitch[..., 1].sin()
        rotation_neg_x[..., 2, 2] = self.pitch[..., 1].cos()

        rotation_z = self.pitch.new_zeros((*self.tilt.shape, 3, 3))
        rotation_z[..., 0, 0] = self.tilt.cos()
        rotation_z[..., 0, 1] = -self.tilt.sin()
        rotation_z[..., 1, 0] = self.tilt.sin()
        rotation_z[..., 1, 1] = self.tilt.cos()
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
        pass  # Patch element does not have a visual representation
