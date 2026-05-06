import math
from typing import Sequence

import torch


def cloud_in_cell_charge_deposition(
    positions: Sequence[torch.Tensor],
    bins: Sequence[torch.Tensor],
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fast and differentiable Cloud-in-Cell (CIC) charge deposition.

    :param positions: List or tuple of particle position tensors, each of shape
        `(..., N)`, leading with an arbitrary number of vector dimensions. `N` is the
        number of particles. The number of tensors passed determines the spatial
        dimensionality of the resulting charge grid.
    :param bins: List or tuple of 1D tensors defining the grid coordinates for each
        dimension. ((N1,), (N2,), ...) where `Ni` is the number of grid points in
        dimension `i`. Each grid coordinate tensor must have uniform spacing. Length
        must match `positions`.
    :param weights: Particle charge weights of shape `(..., N)`. If `None`, all
        particles have weight 1.
    :return: Charge density on the d-dimensional grid with shape
        `(..., N1, N2, ..., Nd)`, where `d = len(positions)`.
    """
    assert len(positions) > 0, "At least one position tensor must be provided"
    assert len(positions) == len(
        bins
    ), "Number of position tensors must match number of bin tensors"

    if weights is None:
        weights = torch.ones_like(positions[0])

    num_hist_dims = len(positions)

    stacked_positions = torch.stack(positions).movedim(
        0, -1
    )  # Shape (..., num_samples, num_hist_dims)

    # Grid dimensions and spacing validation
    # TODO: This interface is bad. If bin widths need to be the same, just change the
    # interface.
    histogram_shape = []
    bin_widths_for_each_hist_dim = []
    for hist_dim, single_dim_bins in enumerate(bins):
        num_bins = single_dim_bins.numel()
        bin_widths = single_dim_bins.diff()
        reference_bin_width = bin_widths[0]

        assert num_bins > 2, f"bins[{hist_dim}] must have at least 2 elements"
        assert torch.allclose(
            bin_widths, reference_bin_width, rtol=1e-4
        ), f"bins[{hist_dim}] must have uniform spacing"

        histogram_shape.append(num_bins)
        bin_widths_for_each_hist_dim.append(reference_bin_width)

    # Set weights to zero for particles outside grid bounds
    extent = torch.stack(
        [single_dim_bins[[0, -1]] for single_dim_bins in bins]
    )  # Shape (num_hist_dims, 2)
    inside_mask = (
        (extent[..., 0] <= stacked_positions) & (stacked_positions < extent[..., 1])
    ).all(
        dim=-1
    )  # Shape (..., num_samples)
    masked_weights = weights * inside_mask  # Shape (..., num_samples)

    # Normalise particle coordinates to grid index space
    idx_space_positions = (
        stacked_positions - extent[..., 0]
    ) / bin_widths_for_each_hist_dim[-1]
    idx_space_integer_positions = idx_space_positions.floor().long()
    idx_space_fractional_positions = idx_space_positions - idx_space_integer_positions

    # Generate all corner combinations and their weights
    num_corners = 2**num_hist_dims
    # TODO speed: torch.tensor(list(itertools.product([0, 1], repeat=D)), device=device)
    corner_offsets = (
        torch.arange(num_corners).unsqueeze(-1)
        // (2 ** torch.arange(num_hist_dims))
        % 2
    )
    corner_positional_weight_factors = idx_space_fractional_positions.unsqueeze(
        -2
    ).where(corner_offsets == 1, 1.0 - idx_space_fractional_positions.unsqueeze(-2))
    idx_space_corner_positions = (
        idx_space_integer_positions.unsqueeze(-1) + corner_offsets
    )  # TODO .clamp(0, histogram_shape[hist_dim] - 1) ?
    corner_weights = masked_weights.unsqueeze(
        -1
    ) * corner_positional_weight_factors.prod(dim=-2)

    def multi_to_flat_index(idx_list):
        """Convert multi-dimensional indices to flat indices."""
        flat_idx = idx_list[0]
        stride = 1
        for hist_dim in range(1, num_hist_dims):
            stride *= histogram_shape[hist_dim - 1]
            flat_idx = flat_idx + idx_list[hist_dim] * stride

        return flat_idx

    # Flatten vector dimensions and particle dimension together
    vector_shape = stacked_positions.shape[:-2]
    num_vector_elements = math.prod(vector_shape)

    # Prepare all indices and weights for vectorised processing
    flattened_idx_space_corner_positions = torch.concatenate(
        [
            multi_to_flat_index(corner_idx_list).flatten(end_dim=-3)
            for corner_idx_list in idx_space_corner_positions
        ],
        dim=1,
    )  # Shape (num_vector_elements, num_corners * num_samples)
    flattened_corner_weights = corner_weights.flatten(
        end_dim=-4
    )  # Flatten vector dimensions  # Shape (num_vector_elements, num_samples, num_hist_dims, num_corners)

    # Output buffer
    total_grid_size = math.prod(histogram_shape)
    charge = positions[0].new_zeros((num_vector_elements, total_grid_size))

    charge_tensor = corner_weights.new_zeros(*vector_shape, *histogram_shape)
    idx_space_corner_positions_vector_indices = tuple(
        torch.arange(this_vector_dim_length).reshape(
            this_vector_dim_length, *([1] * (num_hist_dims - dim_idx))
        )
        for dim_idx, this_vector_dim_length in enumerate(vector_shape)
    )
    idx_space_corner_positions_spatial_indices = (
        idx_space_corner_positions.movedim(-1, -2)
        .flatten(start_dim=-3, end_dim=-2)
        .unbind(-1)
    )
    idx_space_corner_positions_full_indices = (
        idx_space_corner_positions_vector_indices
        + idx_space_corner_positions_spatial_indices
    )
    # charge_tensor[idx_space_corner_positions_full_indices] = corner_weights.movedim(
    #     -1, -2
    # ).flatten(start_dim=-3, end_dim=-2)
    charge_tensor = charge_tensor.index_put(
        idx_space_corner_positions_full_indices,
        corner_weights.movedim(-1, -2).flatten(start_dim=-3, end_dim=-2),
        accumulate=True,
    )

    # Vectorised index_add_
    for flattened_vector_idx in range(num_vector_elements):
        charge[flattened_vector_idx].index_add_(
            0,
            flattened_idx_space_corner_positions[flattened_vector_idx].to(torch.int64),
            flattened_corner_weights[flattened_vector_idx],
        )

    # Compute inverse cell volume
    cell_volume = 1.0
    for bin_width in bin_widths_for_each_hist_dim:
        cell_volume *= bin_width
    inv_cell_volume = 1.0 / cell_volume
    charge = charge * inv_cell_volume

    # Reshape back to original vector dimensions + grid dimensions
    out_shape = (*vector_shape, *histogram_shape[::-1])
    charge = charge.reshape(out_shape)  # Grid dimensions are reversed by the reshape

    num_vector_dims = len(vector_shape)
    spatial_axes = list(range(num_vector_dims, num_vector_dims + num_hist_dims))

    # Permute to put spatial axes in the correct order
    return charge.permute(*range(num_vector_dims), *reversed(spatial_axes))


def cloud_in_cell_charge_deposition_1d(
    x: torch.Tensor, bins: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Fast Cloud-in-Cell (CIC) charge deposition in 1D.

    This is a convenience wrapper around `cloud_in_cell_charge_deposition` for 1D cases.
    Particles outside the grid bounds have their weights set to zero.

    :param x: Particle positions of shape `(..., N)`.
    :param bins: 1D tensor of grid coordinates with shape `(Nx,)`. Must have uniform
        spacing.
    :param weights: Particle charge weights of shape `(..., N)`. If `None`, all
        particles have weight 1.
    :return: Charge density on the 1D grid with shape `(..., Nx)`.
    """
    return cloud_in_cell_charge_deposition([x], [bins], weights)


def cloud_in_cell_charge_deposition_2d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bins1: torch.Tensor,
    bins2: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fast Cloud-in-Cell (CIC) charge deposition in 2D.

    This is a convenience wrapper around `cloud_in_cell_charge_deposition` for 2D cases.
    Particles outside the grid bounds have their weights set to zero.

    :param x1: Particle x positions of shape `(..., N)`.
    :param x2: Particle y positions of shape `(..., N)`.
    :param bins1: 1D tensor of x-grid coordinates with shape `(Nx,)`. Must have uniform
        spacing.
    :param bins2: 1D tensor of y-grid coordinates with shape `(Ny,)`. Must have uniform
        spacing.
    :param weights: Particle charge weights of shape `(..., N)`. If `None`, all
        particles have weight 1.
    :return: Charge density on the 2D grid with shape `(..., Nx, Ny)`.
    """
    return cloud_in_cell_charge_deposition([x1, x2], [bins1, bins2], weights)


def cloud_in_cell_charge_deposition_3d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    bins1: torch.Tensor,
    bins2: torch.Tensor,
    bins3: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fast Cloud-in-Cell (CIC) charge deposition in 3D.

    This is a convenience wrapper around `cloud_in_cell_charge_deposition` for 3D cases.
    Particles outside the grid bounds have their weights set to zero.

    :param x1: Particle x positions of shape `(..., N)`.
    :param x2: Particle y positions of shape `(..., N)`.
    :param x3: Particle z positions of shape `(..., N)`.
    :param bins1: 1D tensor of x-grid coordinates with shape `(Nx,)`. Must have uniform
        spacing.
    :param bins2: 1D tensor of y-grid coordinates with shape `(Ny,)`. Must have uniform
        spacing.
    :param bins3: 1D tensor of z-grid coordinates with shape `(Nz,)`. Must have uniform
        spacing.
    :param weights: Particle charge weights of shape `(..., N)`. If `None`, all
        particles have weight 1.
    :return: Charge density on the 3D grid with shape `(..., Nx, Ny, Nz)`.
    """
    return cloud_in_cell_charge_deposition([x1, x2, x3], [bins1, bins2, bins3], weights)
