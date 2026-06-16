import math
from typing import Sequence

import torch


def cloud_in_cell_charge_deposition(
    positions: torch.Tensor,
    bins: int | Sequence[int],
    extent: torch.Tensor | None = None,
    charges: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fast and differentiable Cloud-in-Cell (CIC) charge deposition.

    :param positions: Tensor of particle positions with shape
        `(..., num_particles, num_hist_dims)`, where `num_hist_dims` is the number of
        spatial dimensions for the charge grid.
    :param bins: Can be a single int or a sequence of ints of length equal to the number
        of position tensors, specifying the number of bins in each spatial dimension.
    :param extent: Tensor of shape (..., num_hist_dims, 2) specifying the leftmost and
        rightmost bin edges in each spatial dimension. If `None`, the extent is inferred
        from the min and max of the positions in each spatial dimension. If provided,
        particles outside the specified extent have their weights set to zero.
    :param charges: Particle charges of shape `(..., num_particles)`. If `None`, all
        particles have charge 1.0.
    :return: Charge density on the d-dimensional grid with shape
        `(..., *histogram_shape*)`, where `d = num_hist_dims`.
    """
    if extent is None:
        extent = torch.stack([positions.amin(dim=-2), positions.amax(dim=-2)], dim=-1)
    if charges is None:
        charges = torch.ones_like(positions[..., 0])

    num_hist_dims = positions.shape[-1]
    histogram_shape = [bins] * num_hist_dims if isinstance(bins, int) else bins
    assert (
        len(histogram_shape) == num_hist_dims
    ), "Number of bin values must match number of position dimensions."

    num_particles = positions.shape[-2]
    vector_shape = positions.shape[:-2]
    num_vector_elements = math.prod(vector_shape) if vector_shape else 1

    # Normalise particle coordinates to normalised bin space
    bin_space_upper_bounds = torch.tensor(
        histogram_shape, device=positions.device, dtype=positions.dtype
    )
    bin_widths = (extent[..., 1] - extent[..., 0]) / bin_space_upper_bounds
    positions_in_bin_space = ((positions - extent[..., 0]) / bin_widths) - 0.5
    positions_in_bin_space_int_component = positions_in_bin_space.floor().long()
    positions_in_bin_space_fractional_components = (
        positions_in_bin_space - positions_in_bin_space_int_component
    )

    # Set charges to zero for particles outside grid bounds
    inside_mask = (
        (0 <= positions_in_bin_space)
        & (positions_in_bin_space < bin_space_upper_bounds)
    ).all(dim=-1)
    masked_charges = charges * inside_mask

    # Precompute strides for converting multi-dimensional indices to flat indices.
    # For histogram_shape [H0, H1, H2] the strides are [H1*H2, H2, 1], i.e. row-major.
    strides = positions_in_bin_space_int_component.new_tensor(
        [math.prod(histogram_shape[i + 1 :]) for i in range(num_hist_dims)]
    )

    # Initialise the output charge grid
    total_grid_size = math.prod(histogram_shape)
    charge_grid = torch.zeros(
        *vector_shape, total_grid_size, dtype=positions.dtype, device=positions.device
    )

    # Flatten vector dimensions into a single dimension for the loop
    flat_positions_int = positions_in_bin_space_int_component.reshape(
        num_vector_elements, num_particles, num_hist_dims
    )
    flat_frac = positions_in_bin_space_fractional_components.reshape(
        num_vector_elements, num_particles, num_hist_dims
    )
    flat_charges = masked_charges.reshape(num_vector_elements, num_particles)
    hist_shape_tensor = flat_positions_int.new_tensor(histogram_shape)

    flat_charge_grid = charge_grid.reshape(num_vector_elements, total_grid_size)

    for vector_idx in range(num_vector_elements):
        # For each of the 2^num_hist_dims corners, compute indices and weights
        for corner in range(2**num_hist_dims):
            corner_indices = []
            corner_weight = flat_charges[vector_idx]  # (num_particles,)

            for dim in range(num_hist_dims):
                use_right = (corner >> dim) & 1
                idx = flat_positions_int[vector_idx, :, dim] + use_right
                idx_clamped = idx.clamp(0, hist_shape_tensor[dim] - 1)

                if use_right:
                    weight_factor = flat_frac[vector_idx, :, dim]
                else:
                    weight_factor = 1.0 - flat_frac[vector_idx, :, dim]

                corner_indices.append(idx_clamped)
                corner_weight = corner_weight * weight_factor

            # Convert multi-dimensional corner indices to flat bin index.
            # flat_idx = idx_0 * strides[0] + idx_1 * strides[1] + ...
            flat_idx = corner_indices[0] * strides[0]
            for dim in range(1, num_hist_dims):
                flat_idx = flat_idx + corner_indices[dim] * strides[dim]

            flat_charge_grid[vector_idx].index_add_(0, flat_idx, corner_weight)

    # Reshape back to the original vector and grid dimensions
    charge_grid = charge_grid.reshape(*vector_shape, *histogram_shape)

    return charge_grid
