import itertools
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

    # Use specialised fast implementations for 1D, 2D and 3D, and a general
    # implementation for higher dimensions.
    if num_hist_dims == 1:
        flat_charge_grid = _cloud_in_cell_1d(
            positions, histogram_shape, extent, charges
        )
    elif num_hist_dims == 2:
        flat_charge_grid = _cloud_in_cell_2d(
            positions, histogram_shape, extent, charges
        )
    elif num_hist_dims == 3:
        flat_charge_grid = _cloud_in_cell_3d(
            positions, histogram_shape, extent, charges
        )
    else:
        flat_charge_grid = _cloud_in_cell_nd(
            positions, histogram_shape, extent, charges
        )

    vector_shape = positions.shape[:-2]
    charge_grid = flat_charge_grid.reshape(*vector_shape, *histogram_shape)

    return charge_grid


def _cloud_in_cell_1d(
    positions: torch.Tensor,
    histogram_shape: Sequence[int],
    extent: torch.Tensor,
    charges: torch.Tensor,
) -> torch.Tensor:
    """Fast specialised implementation of 1D Cloud-in-Cell charge deposition."""
    num_bins_x = histogram_shape[0]
    vector_shape = positions.shape[:-2]
    flat_charge_grid = positions.new_zeros(*vector_shape, num_bins_x)

    positions_x = positions[..., 0].contiguous()
    extent_left_x, extent_right_x = extent[..., 0, 0].unsqueeze(-1), extent[
        ..., 0, 1
    ].unsqueeze(-1)

    in_extent = (positions_x >= extent_left_x) & (positions_x <= extent_right_x)
    masked_charges = charges * in_extent

    bin_width_x = (extent_right_x - extent_left_x) / num_bins_x
    positions_in_bin_space_x = (positions_x - extent_left_x) / bin_width_x - 0.5

    positions_in_bin_space_int_component_x = positions_in_bin_space_x.floor().long()
    positions_in_bin_space_fractional_components_x = (
        positions_in_bin_space_x - positions_in_bin_space_int_component_x
    )

    clamped_corner_positions_x0 = positions_in_bin_space_int_component_x.clamp(
        0, num_bins_x - 1
    )
    clamped_corner_positions_x1 = (positions_in_bin_space_int_component_x + 1).clamp(
        0, num_bins_x - 1
    )

    corner_mask_x0 = (positions_in_bin_space_int_component_x >= 0) & (
        positions_in_bin_space_int_component_x < num_bins_x
    )
    corner_mask_x1 = (positions_in_bin_space_int_component_x + 1 >= 0) & (
        positions_in_bin_space_int_component_x + 1 < num_bins_x
    )

    corner_weight_factors_x0 = (
        1.0 - positions_in_bin_space_fractional_components_x
    ) * corner_mask_x0
    corner_weight_factors_x1 = (
        positions_in_bin_space_fractional_components_x * corner_mask_x1
    )

    flat_charge_grid.scatter_add_(
        dim=-1,
        index=clamped_corner_positions_x0,
        src=masked_charges * corner_weight_factors_x0,
    )
    flat_charge_grid.scatter_add_(
        dim=-1,
        index=clamped_corner_positions_x1,
        src=masked_charges * corner_weight_factors_x1,
    )

    return flat_charge_grid


def _cloud_in_cell_2d(
    positions: torch.Tensor,
    histogram_shape: Sequence[int],
    extent: torch.Tensor,
    charges: torch.Tensor,
) -> torch.Tensor:
    """Fast specialised implementation of 2D Cloud-in-Cell charge deposition."""
    num_bins_x, num_bins_y = histogram_shape[0], histogram_shape[1]
    vector_shape = positions.shape[:-2]
    flat_charge_grid = positions.new_zeros(*vector_shape, num_bins_x * num_bins_y)

    positions_x = positions[..., 0].contiguous()
    positions_y = positions[..., 1].contiguous()

    extent_left_x, extent_right_x = extent[..., 0, 0].unsqueeze(-1), extent[
        ..., 0, 1
    ].unsqueeze(-1)
    extent_left_y, extent_right_y = extent[..., 1, 0].unsqueeze(-1), extent[
        ..., 1, 1
    ].unsqueeze(-1)

    in_extent = (
        (positions_x >= extent_left_x)
        & (positions_x <= extent_right_x)
        & (positions_y >= extent_left_y)
        & (positions_y <= extent_right_y)
    )
    masked_charges = charges * in_extent

    bin_width_x = (extent_right_x - extent_left_x) / num_bins_x
    bin_width_y = (extent_right_y - extent_left_y) / num_bins_y

    positions_in_bin_space_x = (positions_x - extent_left_x) / bin_width_x - 0.5
    positions_in_bin_space_y = (positions_y - extent_left_y) / bin_width_y - 0.5

    positions_in_bin_space_int_component_x = positions_in_bin_space_x.floor().long()
    positions_in_bin_space_int_component_y = positions_in_bin_space_y.floor().long()

    positions_in_bin_space_fractional_components_x = (
        positions_in_bin_space_x - positions_in_bin_space_int_component_x
    )
    positions_in_bin_space_fractional_components_y = (
        positions_in_bin_space_y - positions_in_bin_space_int_component_y
    )

    clamped_corner_positions_x0 = positions_in_bin_space_int_component_x.clamp(
        0, num_bins_x - 1
    )
    clamped_corner_positions_x1 = (positions_in_bin_space_int_component_x + 1).clamp(
        0, num_bins_x - 1
    )
    clamped_corner_positions_y0 = positions_in_bin_space_int_component_y.clamp(
        0, num_bins_y - 1
    )
    clamped_corner_positions_y1 = (positions_in_bin_space_int_component_y + 1).clamp(
        0, num_bins_y - 1
    )

    corner_mask_x0 = (positions_in_bin_space_int_component_x >= 0) & (
        positions_in_bin_space_int_component_x < num_bins_x
    )
    corner_mask_x1 = (positions_in_bin_space_int_component_x + 1 >= 0) & (
        positions_in_bin_space_int_component_x + 1 < num_bins_x
    )
    corner_mask_y0 = (positions_in_bin_space_int_component_y >= 0) & (
        positions_in_bin_space_int_component_y < num_bins_y
    )
    corner_mask_y1 = (positions_in_bin_space_int_component_y + 1 >= 0) & (
        positions_in_bin_space_int_component_y + 1 < num_bins_y
    )

    corner_weight_factors_x0 = (
        1.0 - positions_in_bin_space_fractional_components_x
    ) * corner_mask_x0
    corner_weight_factors_x1 = (
        positions_in_bin_space_fractional_components_x * corner_mask_x1
    )
    corner_weight_factors_y0 = (
        1.0 - positions_in_bin_space_fractional_components_y
    ) * corner_mask_y0
    corner_weight_factors_y1 = (
        positions_in_bin_space_fractional_components_y * corner_mask_y1
    )

    stride_x = num_bins_y
    stride_y = 1

    flat_charge_grid.scatter_add_(
        dim=-1,
        index=clamped_corner_positions_x0 * stride_x
        + clamped_corner_positions_y0 * stride_y,
        src=masked_charges * corner_weight_factors_x0 * corner_weight_factors_y0,
    )
    flat_charge_grid.scatter_add_(
        dim=-1,
        index=clamped_corner_positions_x1 * stride_x
        + clamped_corner_positions_y0 * stride_y,
        src=masked_charges * corner_weight_factors_x1 * corner_weight_factors_y0,
    )
    flat_charge_grid.scatter_add_(
        dim=-1,
        index=clamped_corner_positions_x0 * stride_x
        + clamped_corner_positions_y1 * stride_y,
        src=masked_charges * corner_weight_factors_x0 * corner_weight_factors_y1,
    )
    flat_charge_grid.scatter_add_(
        dim=-1,
        index=clamped_corner_positions_x1 * stride_x
        + clamped_corner_positions_y1 * stride_y,
        src=masked_charges * corner_weight_factors_x1 * corner_weight_factors_y1,
    )

    return flat_charge_grid


def _cloud_in_cell_3d(
    positions: torch.Tensor,
    histogram_shape: Sequence[int],
    extent: torch.Tensor,
    charges: torch.Tensor,
) -> torch.Tensor:
    """Fast specialised implementation of 3D Cloud-in-Cell charge deposition."""
    num_bins_x, num_bins_y, num_bins_z = (
        histogram_shape[0],
        histogram_shape[1],
        histogram_shape[2],
    )
    vector_shape = positions.shape[:-2]
    flat_charge_grid = positions.new_zeros(
        *vector_shape, num_bins_x * num_bins_y * num_bins_z
    )

    positions_x = positions[..., 0].contiguous()
    positions_y = positions[..., 1].contiguous()
    positions_z = positions[..., 2].contiguous()

    extent_left_x, extent_right_x = extent[..., 0, 0].unsqueeze(-1), extent[
        ..., 0, 1
    ].unsqueeze(-1)
    extent_left_y, extent_right_y = extent[..., 1, 0].unsqueeze(-1), extent[
        ..., 1, 1
    ].unsqueeze(-1)
    extent_left_z, extent_right_z = extent[..., 2, 0].unsqueeze(-1), extent[
        ..., 2, 1
    ].unsqueeze(-1)

    in_extent = (
        (positions_x >= extent_left_x)
        & (positions_x <= extent_right_x)
        & (positions_y >= extent_left_y)
        & (positions_y <= extent_right_y)
        & (positions_z >= extent_left_z)
        & (positions_z <= extent_right_z)
    )
    masked_charges = charges * in_extent

    bin_width_x = (extent_right_x - extent_left_x) / num_bins_x
    bin_width_y = (extent_right_y - extent_left_y) / num_bins_y
    bin_width_z = (extent_right_z - extent_left_z) / num_bins_z

    positions_in_bin_space_x = (positions_x - extent_left_x) / bin_width_x - 0.5
    positions_in_bin_space_y = (positions_y - extent_left_y) / bin_width_y - 0.5
    positions_in_bin_space_z = (positions_z - extent_left_z) / bin_width_z - 0.5

    positions_in_bin_space_int_component_x = positions_in_bin_space_x.floor().long()
    positions_in_bin_space_int_component_y = positions_in_bin_space_y.floor().long()
    positions_in_bin_space_int_component_z = positions_in_bin_space_z.floor().long()

    positions_in_bin_space_fractional_components_x = (
        positions_in_bin_space_x - positions_in_bin_space_int_component_x
    )
    positions_in_bin_space_fractional_components_y = (
        positions_in_bin_space_y - positions_in_bin_space_int_component_y
    )
    positions_in_bin_space_fractional_components_z = (
        positions_in_bin_space_z - positions_in_bin_space_int_component_z
    )

    clamped_corner_positions_x0 = positions_in_bin_space_int_component_x.clamp(
        0, num_bins_x - 1
    )
    clamped_corner_positions_x1 = (positions_in_bin_space_int_component_x + 1).clamp(
        0, num_bins_x - 1
    )
    clamped_corner_positions_y0 = positions_in_bin_space_int_component_y.clamp(
        0, num_bins_y - 1
    )
    clamped_corner_positions_y1 = (positions_in_bin_space_int_component_y + 1).clamp(
        0, num_bins_y - 1
    )
    clamped_corner_positions_z0 = positions_in_bin_space_int_component_z.clamp(
        0, num_bins_z - 1
    )
    clamped_corner_positions_z1 = (positions_in_bin_space_int_component_z + 1).clamp(
        0, num_bins_z - 1
    )

    corner_mask_x0 = (positions_in_bin_space_int_component_x >= 0) & (
        positions_in_bin_space_int_component_x < num_bins_x
    )
    corner_mask_x1 = (positions_in_bin_space_int_component_x + 1 >= 0) & (
        positions_in_bin_space_int_component_x + 1 < num_bins_x
    )
    corner_mask_y0 = (positions_in_bin_space_int_component_y >= 0) & (
        positions_in_bin_space_int_component_y < num_bins_y
    )
    corner_mask_y1 = (positions_in_bin_space_int_component_y + 1 >= 0) & (
        positions_in_bin_space_int_component_y + 1 < num_bins_y
    )
    corner_mask_z0 = (positions_in_bin_space_int_component_z >= 0) & (
        positions_in_bin_space_int_component_z < num_bins_z
    )
    corner_mask_z1 = (positions_in_bin_space_int_component_z + 1 >= 0) & (
        positions_in_bin_space_int_component_z + 1 < num_bins_z
    )

    corner_weight_factors_x0 = (
        1.0 - positions_in_bin_space_fractional_components_x
    ) * corner_mask_x0
    corner_weight_factors_x1 = (
        positions_in_bin_space_fractional_components_x * corner_mask_x1
    )
    corner_weight_factors_y0 = (
        1.0 - positions_in_bin_space_fractional_components_y
    ) * corner_mask_y0
    corner_weight_factors_y1 = (
        positions_in_bin_space_fractional_components_y * corner_mask_y1
    )
    corner_weight_factors_z0 = (
        1.0 - positions_in_bin_space_fractional_components_z
    ) * corner_mask_z0
    corner_weight_factors_z1 = (
        positions_in_bin_space_fractional_components_z * corner_mask_z1
    )

    stride_x = num_bins_y * num_bins_z
    stride_y = num_bins_z
    stride_z = 1

    for ox, oy, oz in itertools.product([0, 1], repeat=3):
        idx = (
            (clamped_corner_positions_x0 if ox == 0 else clamped_corner_positions_x1)
            * stride_x
            + (clamped_corner_positions_y0 if oy == 0 else clamped_corner_positions_y1)
            * stride_y
            + (clamped_corner_positions_z0 if oz == 0 else clamped_corner_positions_z1)
            * stride_z
        )
        weight = (
            (corner_weight_factors_x0 if ox == 0 else corner_weight_factors_x1)
            * (corner_weight_factors_y0 if oy == 0 else corner_weight_factors_y1)
            * (corner_weight_factors_z0 if oz == 0 else corner_weight_factors_z1)
        )
        flat_charge_grid.scatter_add_(dim=-1, index=idx, src=masked_charges * weight)

    return flat_charge_grid


def _cloud_in_cell_nd(
    positions: torch.Tensor,
    histogram_shape: Sequence[int],
    extent: torch.Tensor,
    charges: torch.Tensor,
) -> torch.Tensor:
    """General implementation of n-dimensional Cloud-in-Cell charge deposition."""
    num_hist_dims = positions.shape[-1]
    vector_shape = positions.shape[:-2]
    num_histogram_bins = math.prod(histogram_shape)
    flat_charge_grid = positions.new_zeros(*vector_shape, num_histogram_bins)

    in_extent = torch.ones_like(charges, dtype=torch.bool)
    for d in range(num_hist_dims):
        coord = positions[..., d]
        extent_left_d = extent[..., d, 0].unsqueeze(-1)
        extent_right_d = extent[..., d, 1].unsqueeze(-1)
        in_extent = in_extent & (coord >= extent_left_d) & (coord <= extent_right_d)

    masked_charges = charges * in_extent

    positions_in_bin_space_int_components_dims = []
    positions_in_bin_space_fractional_components_dims = []
    for d in range(num_hist_dims):
        coord = positions[..., d].contiguous()
        extent_left_d = extent[..., d, 0].unsqueeze(-1)
        extent_right_d = extent[..., d, 1].unsqueeze(-1)
        num_bins_d = histogram_shape[d]
        bin_width_d = (extent_right_d - extent_left_d) / num_bins_d
        positions_in_bin_space_d = (coord - extent_left_d) / bin_width_d - 0.5

        positions_in_bin_space_int_component_d = positions_in_bin_space_d.floor().long()
        positions_in_bin_space_fractional_components_d = (
            positions_in_bin_space_d - positions_in_bin_space_int_component_d
        )

        positions_in_bin_space_int_components_dims.append(
            positions_in_bin_space_int_component_d
        )
        positions_in_bin_space_fractional_components_dims.append(
            positions_in_bin_space_fractional_components_d
        )

    strides = [math.prod(histogram_shape[d + 1 :]) for d in range(num_hist_dims)]

    for corner in itertools.product([0, 1], repeat=num_hist_dims):
        corner_idx = 0
        corner_weight = masked_charges.clone()
        for d in range(num_hist_dims):
            use_right = corner[d]
            idx = positions_in_bin_space_int_components_dims[d] + use_right
            clamped_corner_positions_d = idx.clamp(0, histogram_shape[d] - 1)
            corner_idx = corner_idx + clamped_corner_positions_d * strides[d]

            corner_mask_d = (idx >= 0) & (idx < histogram_shape[d])
            corner_weight_factor_d = (
                positions_in_bin_space_fractional_components_dims[d]
                if use_right
                else (1.0 - positions_in_bin_space_fractional_components_dims[d])
            ) * corner_mask_d
            corner_weight = corner_weight * corner_weight_factor_d

        flat_charge_grid.scatter_add_(dim=-1, index=corner_idx, src=corner_weight)

    return flat_charge_grid
