import matplotlib.pyplot as plt
import torch
from scipy.constants import elementary_charge, epsilon_0, speed_of_light

from cheetah.accelerator.element import Element
from cheetah.particles import ParticleBeam
from cheetah.utils import verify_device_and_dtype


class SpaceChargeKick(Element):
    """
    Applies the effect of space charge over a length `effect_length`, on the
    **momentum** (i.e. divergence and energy spread) of the beam. The positions are
    unmodified; this is meant to be combined with another lattice element (e.g. `Drift`)
    that does modify the positions, but does not take into account space charge. The
    integrated Green function method
    (https://journals.aps.org/prab/abstract/10.1103/PhysRevSTAB.9.044204) is used to
    compute the effect of space charge. This is similar to the method used in Ocelot.
    The main difference is that it solves the Poisson equation in the beam frame, while
    here we solve a modified Poisson equation in the laboratory frame
    (https://pubs.aip.org/aip/pop/article-abstract/15/5/056701/1016636/Simulation-of-beams-or-plasmas-crossing-at).
    The two methods are in principle equivalent.

    Overview of the method:
     - Compute the beam charge density on a grid.
     - Convolve the charge density with a Green function (the integrated green function)
       to find the potential `phi` on the grid. The convolution uses the Hockney method
       for open boundaries (allocate 2x larger arrays and perform convolution using
       FFTs).
     - Compute the corresponding electromagnetic fields and Lorentz force on the grid.
     - Interpolate the Lorentz force to the particles and update their momentum.

    :param effect_length: Length over which the effect is applied in meters.
    :param grid_shape: Number of grid points in (x, y, tau) directions.
    :param grid_extent_x: Dimensions of the grid on which to compute space-charge, as
        multiples of sigma of the beam in the x direction (dimensionless).
    :param grid_extent_y: Dimensions of the grid on which to compute space-charge, as
        multiples of sigma of the beam in the y direction (dimensionless).
    :param grid_extent_tau: Dimensions of the grid on which to compute space-charge, as
        multiples of sigma of the beam in the tau direction (dimensionless).
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python
        variable name. This is needed if you want to use the `segment.element_name`
        syntax to access the element in a segment.
    """

    def __init__(
        self,
        effect_length: torch.Tensor,
        grid_shape: tuple[int, int, int] = (32, 32, 32),
        # TODO: Simplify these to a single tensor?
        grid_extent_x: torch.Tensor | None = None,
        grid_extent_y: torch.Tensor | None = None,
        grid_extent_tau: torch.Tensor | None = None,
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        device, dtype = verify_device_and_dtype(
            [effect_length, grid_extent_x, grid_extent_y, grid_extent_tau],
            device,
            dtype,
        )
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.grid_shape = grid_shape

        self.register_buffer_or_parameter(
            "effect_length", torch.as_tensor(effect_length, **factory_kwargs)
        )
        # In multiples of sigma
        self.register_buffer_or_parameter(
            "grid_extent_x",
            torch.as_tensor(
                grid_extent_x if grid_extent_x is not None else 3.0, **factory_kwargs
            ),
        )
        self.register_buffer_or_parameter(
            "grid_extent_y",
            torch.as_tensor(
                grid_extent_y if grid_extent_y is not None else 3.0, **factory_kwargs
            ),
        )
        self.register_buffer_or_parameter(
            "grid_extent_tau",
            torch.as_tensor(
                grid_extent_tau if grid_extent_tau is not None else 3.0,
                **factory_kwargs,
            ),
        )

    def _deposit_charge_on_grid(
        self,
        beam: ParticleBeam,
        xp_coordinates: torch.Tensor,
        cell_size: torch.Tensor,
        grid_dimensions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Deposits the charge density of the beam onto a grid, using the
        Cloud-In-Cell (CIC) method. Returns a grid of charge density in C/m^3.
        """
        charge = torch.zeros(
            beam.particles.shape[:-2] + self.grid_shape,
            device=beam.particles.device,
            dtype=beam.particles.dtype,
        )

        # Compute inverse cell size (to avoid multiple divisions later on)
        inv_cell_size = 1 / cell_size

        # Get particle positions
        particle_positions = xp_coordinates[..., [0, 2, 4]]
        normalized_positions = (
            particle_positions + grid_dimensions.unsqueeze(-2)
        ) * inv_cell_size.unsqueeze(-2)

        # Find indices of the lower corners of the cells containing the particles
        cell_indices = torch.floor(normalized_positions).type(torch.int)

        # Calculate the weights for all surrounding cells
        offsets = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            device=cell_indices.device,
        )
        surrounding_indices = cell_indices.unsqueeze(-2) + offsets.unsqueeze(-3)
        # Shape: (..., num_particles, 8, 3)
        weights = 1 - torch.abs(
            normalized_positions.unsqueeze(-2) - surrounding_indices
        )
        # Shape: (.., num_particles, 8, 3)
        cell_weights = weights.prod(dim=-1)  # Shape: (.., num_particles, 8)

        # Add the charge contributions to the cells
        # Shape: (..., 8 * num_particles)
        idx_vector = (
            torch.arange(cell_indices.shape[0], device=cell_indices.device)
            .repeat(8 * beam.particles.shape[-2], 1)
            .T
        )
        idx_x = surrounding_indices[..., 0].flatten(start_dim=-2)
        idx_y = surrounding_indices[..., 1].flatten(start_dim=-2)
        idx_tau = surrounding_indices[..., 2].flatten(start_dim=-2)

        # Check that particles are inside the grid
        valid_mask = (
            (idx_x >= 0)
            & (idx_x < self.grid_shape[0])
            & (idx_y >= 0)
            & (idx_y < self.grid_shape[1])
            & (idx_tau >= 0)
            & (idx_tau < self.grid_shape[2])
        )

        # Accumulate the charge contributions
        survived_particle_charges = beam.particle_charges * beam.survival_probabilities
        repeated_charges = survived_particle_charges.repeat_interleave(
            repeats=8, dim=-1
        )  # Shape:(..., 8 * num_particles)
        values = (cell_weights.flatten(start_dim=-2) * repeated_charges)[valid_mask]
        charge.index_put_(
            (
                idx_vector[valid_mask],
                idx_x[valid_mask],
                idx_y[valid_mask],
                idx_tau[valid_mask],
            ),
            values,
            accumulate=True,
        )

        # Normalize by the cell volume
        inv_cell_volume = (
            inv_cell_size[..., 0] * inv_cell_size[..., 1] * inv_cell_size[..., 2]
        )

        return charge * inv_cell_volume[..., None, None, None]

    def _integrated_potential(
        self, x: torch.Tensor, y: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the integrate potential as in
        https://journals.aps.org/prab/abstract/10.1103/PhysRevSTAB.10.129901
        The formula used here is slightly different than the one used in
        the above paper, but is equivalent (up to integration constants),
        and is more robust to numerical errors.
        """

        r = torch.sqrt(x**2 + y**2 + tau**2)
        integrated_potential = (
            -0.5 * tau**2 * torch.atan(x * y / (tau * r))
            - 0.5 * y**2 * torch.atan(x * tau / (y * r))
            - 0.5 * x**2 * torch.atan(y * tau / (x * r))
            + y * tau * torch.asinh(x / torch.sqrt(y**2 + tau**2))
            + x * tau * torch.asinh(y / torch.sqrt(x**2 + tau**2))
            + x * y * torch.asinh(tau / torch.sqrt(x**2 + y**2))
        )
        return integrated_potential

    def _array_rho(
        self,
        beam: ParticleBeam,
        xp_coordinates: torch.Tensor,
        cell_size: torch.Tensor,
        grid_dimensions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Allocates a 2x larger array in all dimensions (to perform Hockney's method), and
        copies the charge density in one of the "quadrants".
        """
        charge_density = self._deposit_charge_on_grid(
            beam, xp_coordinates, cell_size, grid_dimensions
        )
        new_dims = tuple(2 * dim for dim in self.grid_shape)

        # Create a new tensor with the doubled dimensions, filled with zeros
        new_charge_density = torch.zeros(
            beam.particles.shape[:-2] + new_dims,
            device=beam.particles.device,
            dtype=beam.particles.dtype,
        )

        # Copy the original charge_density values to the beginning of the new tensor
        new_charge_density[
            ...,
            : charge_density.shape[-3],
            : charge_density.shape[-2],
            : charge_density.shape[-1],
        ] = charge_density

        return new_charge_density

    def _integrated_green_function(
        self, beam: ParticleBeam, cell_size: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the Integrated Green Function (IGF) in the 2x larger array,
        as needed for the Hockney method.
        """
        dx, dy, dtau = (
            cell_size[..., 0],
            cell_size[..., 1],
            cell_size[..., 2] * beam.relativistic_gamma,
            # The longitudinal dimension is scaled by gamma, since we are solving a
            # modified Poisson equation in the lab frame (see docstring of the class)
        )
        num_grid_points_x, num_grid_points_y, num_grid_points_tau = self.grid_shape

        # Create coordinate grids
        x = torch.arange(num_grid_points_x, device=beam.particles.device)
        y = torch.arange(num_grid_points_y, device=beam.particles.device)
        tau = torch.arange(num_grid_points_tau, device=beam.particles.device)
        ix_grid, iy_grid, itau_grid = torch.meshgrid(x, y, tau, indexing="ij")
        x_grid = (
            ix_grid[None, :, :, :] * dx[..., None, None, None]
        )  # Shape: [..., nx, ny, nz]
        y_grid = (
            iy_grid[None, :, :, :] * dy[..., None, None, None]
        )  # Shape: [..., nx, ny, nz]
        tau_grid = (
            itau_grid[None, :, :, :] * dtau[..., None, None, None]
        )  # Shape: [..., nx, ny, nz]

        # Compute the Green's function values
        G_values = (
            self._integrated_potential(
                x_grid + 0.5 * dx[..., None, None, None],
                y_grid + 0.5 * dy[..., None, None, None],
                tau_grid + 0.5 * dtau[..., None, None, None],
            )
            - self._integrated_potential(
                x_grid - 0.5 * dx[..., None, None, None],
                y_grid + 0.5 * dy[..., None, None, None],
                tau_grid + 0.5 * dtau[..., None, None, None],
            )
            - self._integrated_potential(
                x_grid + 0.5 * dx[..., None, None, None],
                y_grid - 0.5 * dy[..., None, None, None],
                tau_grid + 0.5 * dtau[..., None, None, None],
            )
            - self._integrated_potential(
                x_grid + 0.5 * dx[..., None, None, None],
                y_grid + 0.5 * dy[..., None, None, None],
                tau_grid - 0.5 * dtau[..., None, None, None],
            )
            + self._integrated_potential(
                x_grid + 0.5 * dx[..., None, None, None],
                y_grid - 0.5 * dy[..., None, None, None],
                tau_grid - 0.5 * dtau[..., None, None, None],
            )
            + self._integrated_potential(
                x_grid - 0.5 * dx[..., None, None, None],
                y_grid + 0.5 * dy[..., None, None, None],
                tau_grid - 0.5 * dtau[..., None, None, None],
            )
            + self._integrated_potential(
                x_grid - 0.5 * dx[..., None, None, None],
                y_grid - 0.5 * dy[..., None, None, None],
                tau_grid + 0.5 * dtau[..., None, None, None],
            )
            - self._integrated_potential(
                x_grid - 0.5 * dx[..., None, None, None],
                y_grid - 0.5 * dy[..., None, None, None],
                tau_grid - 0.5 * dtau[..., None, None, None],
            )
        )

        # Initialize the grid with double dimensions
        green_func_values = torch.zeros(
            (
                *beam.particles.shape[:-2],
                2 * num_grid_points_x,
                2 * num_grid_points_y,
                2 * num_grid_points_tau,
            ),
            device=beam.particles.device,
            dtype=beam.particles.dtype,
        )

        # Fill the grid with G_values and its periodic copies
        green_func_values[
            ..., :num_grid_points_x, :num_grid_points_y, :num_grid_points_tau
        ] = G_values
        green_func_values[
            ..., num_grid_points_x + 1 :, :num_grid_points_y, :num_grid_points_tau
        ] = G_values[..., 1:, :, :].flip(
            dims=[-3]
        )  # Reverse x, excluding the first element
        green_func_values[
            ..., :num_grid_points_x, num_grid_points_y + 1 :, :num_grid_points_tau
        ] = G_values[..., :, 1:, :].flip(
            dims=[-2]
        )  # Reverse y, excluding the first element
        green_func_values[
            ..., :num_grid_points_x, :num_grid_points_y, num_grid_points_tau + 1 :
        ] = G_values[..., :, :, 1:].flip(
            dims=[-1]
        )  # Reverse s, excluding the first element
        green_func_values[
            ..., num_grid_points_x + 1 :, num_grid_points_y + 1 :, :num_grid_points_tau
        ] = G_values[..., 1:, 1:, :].flip(
            dims=[-3, -2]
        )  # Reverse the x and y dimensions
        green_func_values[
            ..., :num_grid_points_x, num_grid_points_y + 1 :, num_grid_points_tau + 1 :
        ] = G_values[..., :, 1:, 1:].flip(
            dims=[-2, -1]
        )  # Reverse the y and s dimensions
        green_func_values[
            ..., num_grid_points_x + 1 :, :num_grid_points_y, num_grid_points_tau + 1 :
        ] = G_values[..., 1:, :, 1:].flip(
            dims=[-3, -1]
        )  # Reverse the x and s dimensions
        green_func_values[
            ...,
            num_grid_points_x + 1 :,
            num_grid_points_y + 1 :,
            num_grid_points_tau + 1 :,
        ] = G_values[..., 1:, 1:, 1:].flip(
            dims=[-3, -2, -1]
        )  # Reverse all dimensions

        return green_func_values

    def _solve_poisson_equation(
        self,
        beam: ParticleBeam,
        xp_coordinates: torch.Tensor,
        cell_size,
        grid_dimensions,
    ) -> torch.Tensor:  # Works only for ParticleBeam at this stage
        """
        Solves the Poisson equation for the given charge density, using FFT convolution.
        """
        charge_density = self._array_rho(
            beam, xp_coordinates, cell_size, grid_dimensions
        )
        charge_density_ft = torch.fft.rfftn(charge_density, dim=[1, 2, 3])
        integrated_green_function = self._integrated_green_function(beam, cell_size)
        integrated_green_function_ft = torch.fft.rfftn(
            integrated_green_function, dim=[1, 2, 3]
        )
        potential_ft = charge_density_ft * integrated_green_function_ft
        potential = (1 / (4 * torch.pi * epsilon_0)) * torch.fft.irfftn(
            potential_ft, dim=[1, 2, 3]
        ).real

        # Return the physical potential
        return potential[
            ...,
            : charge_density.shape[-3] // 2,
            : charge_density.shape[-2] // 2,
            : charge_density.shape[-1] // 2,
        ]

    def _E_plus_vB_field(
        self,
        beam: ParticleBeam,
        xp_coordinates: torch.Tensor,
        cell_size: torch.Tensor,
        grid_dimensions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the force field from the potential and the particle positions and
        velocities, as in https://doi.org/10.1063/1.2837054.
        """
        inv_cell_size = 1 / cell_size
        igamma2 = torch.zeros_like(beam.relativistic_gamma)
        igamma2[beam.relativistic_gamma != 0] = (
            1 / beam.relativistic_gamma[beam.relativistic_gamma != 0] ** 2
        )
        potential = self._solve_poisson_equation(
            beam, xp_coordinates, cell_size, grid_dimensions
        )

        grad_x = torch.zeros_like(potential)
        grad_y = torch.zeros_like(potential)
        grad_tau = torch.zeros_like(potential)

        # Compute the gradients of the potential, using central differences, with 0
        # boundary conditions
        grad_x[..., 1:-1, :, :] = (
            potential[..., 2:, :, :] - potential[..., :-2, :, :]
        ) * (0.5 * inv_cell_size[..., 0, None, None, None])
        grad_y[..., :, 1:-1, :] = (
            potential[..., :, 2:, :] - potential[..., :, :-2, :]
        ) * (0.5 * inv_cell_size[..., 1, None, None, None])
        grad_tau[..., :, :, 1:-1] = (
            potential[..., :, :, 2:] - potential[..., :, :, :-2]
        ) * (0.5 * inv_cell_size[..., 2, None, None, None])

        # Scale the gradients with lorentz factor
        grad_x = -igamma2[..., None, None, None] * grad_x
        grad_y = -igamma2[..., None, None, None] * grad_y
        grad_tau = -igamma2[..., None, None, None] * grad_tau

        return grad_x, grad_y, grad_tau

    def _compute_forces(
        self,
        beam: ParticleBeam,
        xp_coordinates: torch.Tensor,
        cell_size: torch.Tensor,
        grid_dimensions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Interpolates the space charge force from the grid onto the macroparticles.
        Reciprocal function of _deposit_charge_on_grid. `beam` needs to have a flattened
        vector shape.
        """
        grad_x, grad_y, grad_z = self._E_plus_vB_field(
            beam, xp_coordinates, cell_size, grid_dimensions
        )
        grid_shape = self.grid_shape
        interpolated_forces = torch.zeros(
            (*beam.particles.shape[:-1], 3),
            device=beam.particles.device,
            dtype=beam.particles.dtype,
        )  # (..., num_particles, 3)

        # Get particle positions
        particle_positions = xp_coordinates[..., [0, 2, 4]]
        normalized_positions = (
            particle_positions + grid_dimensions.unsqueeze(-2)
        ) / cell_size.unsqueeze(-2)

        # Find indices of the lower corners of the cells containing the particles
        cell_indices = torch.floor(normalized_positions).type(torch.int)

        # Calculate the weights for all surrounding cells
        offsets = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            device=cell_indices.device,
        )
        surrounding_indices = cell_indices.unsqueeze(-2) + offsets.unsqueeze(
            -3
        )  # Shape:(.., num_particles, 8, 3)
        weights = 1 - torch.abs(
            normalized_positions.unsqueeze(-2) - surrounding_indices
        )  # Shape: (..., num_particles, 8, 3)
        cell_weights = weights.prod(dim=-1)  # Shape: (..., num_particles, 8)

        # Extract forces from the grids
        surrounding_indices_flattened = surrounding_indices.flatten(
            start_dim=-3, end_dim=-2
        )  # Shape: (..., num_particles * 8, 3)
        idx_vector = (
            torch.arange(cell_indices.shape[0], device=cell_indices.device)
            .repeat(8 * beam.particles.shape[-2], 1)
            .T
        )  # Shape: (..., num_particles * 8)
        idx_x = surrounding_indices_flattened[..., 0]
        idx_y = surrounding_indices_flattened[..., 1]
        idx_tau = surrounding_indices_flattened[..., 2]
        valid_mask = (
            (idx_x >= 0)
            & (idx_x < grid_shape[0])
            & (idx_y >= 0)
            & (idx_y < grid_shape[1])
            & (idx_tau >= 0)
            & (idx_tau < grid_shape[2])
        )

        # Keep dimensions, and set F to zero if non-valid
        force_indices = (
            idx_vector,
            torch.clamp(idx_x, min=0, max=grid_shape[0] - 1),
            torch.clamp(idx_y, min=0, max=grid_shape[1] - 1),
            torch.clamp(idx_tau, min=0, max=grid_shape[2] - 1),
        )

        Fx_values = torch.where(valid_mask, grad_x[force_indices], 0)
        Fy_values = torch.where(valid_mask, grad_y[force_indices], 0)
        Fz_values = torch.where(
            valid_mask, grad_z[force_indices], 0
        )  # (..., 8 * num_particles)

        # Compute interpolated forces
        # Cell weights validation is taken care of by the F_x, F_y, F_z values
        cell_weights_with_e = cell_weights.flatten(start_dim=-2) * elementary_charge
        values_x = cell_weights_with_e * Fx_values
        values_y = cell_weights_with_e * Fy_values
        values_z = cell_weights_with_e * Fz_values

        forces_to_add = torch.stack([values_x, values_y, values_z], dim=-1)

        index_tensor = (
            torch.arange(beam.num_particles, device=beam.particles.device)
            .repeat_interleave(8)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(beam.particles.shape[0], 8 * beam.particles.shape[-2], 3)
        )

        # Add the forces to the particles
        accumulated_forces = torch.scatter_add(
            interpolated_forces, dim=1, index=index_tensor, src=forces_to_add
        )

        return accumulated_forces

    def track(self, incoming: ParticleBeam) -> ParticleBeam:
        """
        Tracks particles through the element. The input must be a `ParticleBeam`.

        :param incoming: Beam of particles entering the element.
        :returns: Beam of particles exiting the element.
        """
        if isinstance(incoming, ParticleBeam):
            # This flattening is a hack to only think about one vector dimension in the
            # following code. It is reversed at the end of the function.

            # Make sure that the incoming beam has at least one vector dimension by
            # broadcasting with a dummy dimension (1,).
            vector_shape = torch.broadcast_shapes(
                incoming.particles.shape[:-2],
                incoming.energy.shape,
                incoming.particle_charges.shape[:-1],
                incoming.survival_probabilities.shape[:-1],
                (1,),
            )
            vectorized_incoming = ParticleBeam(
                particles=torch.broadcast_to(
                    incoming.particles, (*vector_shape, incoming.num_particles, 7)
                ),
                energy=torch.broadcast_to(incoming.energy, vector_shape),
                particle_charges=torch.broadcast_to(
                    incoming.particle_charges, (*vector_shape, incoming.num_particles)
                ),
                survival_probabilities=torch.broadcast_to(
                    incoming.survival_probabilities,
                    (*vector_shape, incoming.num_particles),
                ),
                device=incoming.particles.device,
                dtype=incoming.particles.dtype,
            )

            flattened_incoming = ParticleBeam(
                particles=vectorized_incoming.particles.flatten(end_dim=-3),
                energy=vectorized_incoming.energy.flatten(end_dim=-1),
                particle_charges=vectorized_incoming.particle_charges.flatten(
                    end_dim=-2
                ),
                survival_probabilities=(
                    vectorized_incoming.survival_probabilities.flatten(end_dim=-2)
                ),
                device=vectorized_incoming.particles.device,
                dtype=vectorized_incoming.particles.dtype,
            )
            flattened_length_effect = self.effect_length.flatten(end_dim=-1)

            # Compute useful quantities
            grid_dimensions = torch.stack(
                [
                    self.grid_extent_x * flattened_incoming.sigma_x,
                    self.grid_extent_y * flattened_incoming.sigma_y,
                    self.grid_extent_tau * flattened_incoming.sigma_tau,
                ],
                dim=-1,
            )
            cell_size = (
                2
                * grid_dimensions
                / torch.tensor(
                    self.grid_shape,
                    device=grid_dimensions.device,
                    dtype=grid_dimensions.dtype,
                )
            )
            dt = flattened_length_effect / (
                speed_of_light * flattened_incoming.relativistic_beta
            )

            # Change coordinates to apply the space charge effect
            xp_coordinates = flattened_incoming.to_xyz_pxpypz()
            forces = self._compute_forces(
                flattened_incoming, xp_coordinates, cell_size, grid_dimensions
            )
            xp_coordinates[..., 1] = xp_coordinates[..., 1] + forces[
                ..., 0
            ] * dt.unsqueeze(-1)
            xp_coordinates[..., 3] = xp_coordinates[..., 3] + forces[
                ..., 1
            ] * dt.unsqueeze(-1)
            xp_coordinates[..., 5] = xp_coordinates[..., 5] + forces[
                ..., 2
            ] * dt.unsqueeze(-1)

            # Reverse the flattening of the vector dimensions
            outgoing_vector_shape = torch.broadcast_shapes(
                incoming.particles.shape[:-2],
                incoming.energy.shape,
                incoming.particle_charges.shape[:-1],
                incoming.survival_probabilities.shape[:-1],
                self.effect_length.shape,
            )
            outgoing = ParticleBeam.from_xyz_pxpypz(
                xp_coordinates=xp_coordinates.reshape(
                    (*outgoing_vector_shape, incoming.num_particles, 7)
                ),
                energy=incoming.energy,
                particle_charges=incoming.particle_charges,
                survival_probabilities=incoming.survival_probabilities,
                s=incoming.s,
                species=incoming.species,
            )

            return outgoing
        else:
            raise TypeError(f"Parameter incoming is of invalid type {type(incoming)}")

    @property
    def is_skippable(self) -> bool:
        return False

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        ax = ax or plt.subplot(111)

        plot_s = s[vector_idx] if s.dim() > 0 else s

        ax.axvline(plot_s, ymin=0.01, ymax=0.99, color="orange", linestyle="-")

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "effect_length",
            "grid_shape",
            "grid_extent_x",
            "grid_extent_y",
            "grid_extent_tau",
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(effect_length={repr(self.effect_length)}, "
            + f"grid_shape={repr(self.grid_shape)}, "
            + f"grid_extent_x={repr(self.grid_extent_x)}, "
            + f"grid_extent_y={repr(self.grid_extent_y)}, "
            + f"grid_extent_tau={repr(self.grid_extent_tau)}, "
            + f"name={repr(self.name)})"
        )
