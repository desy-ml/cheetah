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
    assert len(histogram_shape) == num_hist_dims, (
        "Number of histogram dimensions defined by bins must match number of position ",
        "tensors",
    )

    # Set weights to zero for particles outside grid bounds
    inside_mask = ((extent[..., 0] <= positions) & (positions < extent[..., 1])).all(
        dim=-1
    )  # Shape (..., num_samples)
    masked_charges = charges * inside_mask  # Shape (..., num_samples)

    # Normalise particle coordinates to normalised bin space
    bin_space_upper_bounds = torch.tensor(histogram_shape, device=positions.device) - 1
    bin_widths = (extent[..., 1] - extent[..., 0]) / bin_space_upper_bounds
    positions_in_bin_space = (positions - extent[..., 0]) / bin_widths
    positions_in_bin_space_int_component = positions_in_bin_space.floor().long()
    positions_in_bin_space_fractional_components = (
        positions_in_bin_space - positions_in_bin_space_int_component
    )

    # Generate all corner combinations and their weights
    corner_offsets = positions_in_bin_space_int_component.new_tensor(
        list(itertools.product([0, 1], repeat=num_hist_dims))
    )  # Shape (num_corners, num_hist_dims)
    corner_positions_in_bin_space = (
        positions_in_bin_space_int_component.unsqueeze(-2) + corner_offsets
    ).clamp(bin_space_upper_bounds.new_zeros(()), bin_space_upper_bounds)
    positions_in_bin_space_fractional_components.unsqueeze(-2)
    corner_weight_factors = torch.where(
        corner_offsets == 0,
        positions_in_bin_space_fractional_components.unsqueeze(-2),
        (1.0 - positions_in_bin_space_fractional_components).unsqueeze(-2),
    )  # Shape (..., num_samples, num_corners, num_hist_dims)
    corner_weights = corner_weight_factors.prod(
        dim=-1
    )  # Shape (..., num_samples, num_corners)
    corner_charges = (  # Actual charge deposition on the corners
        masked_charges.unsqueeze(-1) * corner_weights
    )  # Shape (..., num_samples, num_corners)

    vector_shape = positions.shape[:-2]
    num_histogram_bins = math.prod(histogram_shape)

    corner_positions_in_flat_bin_space = (
        corner_positions_in_bin_space[..., 0]
        + histogram_shape[0] * corner_positions_in_bin_space[..., 1]
    )

    flat_charge_grid = corner_charges.new_zeros(*vector_shape, num_histogram_bins)
    flat_charge_grid.scatter_add_(
        dim=-1,
        index=corner_positions_in_flat_bin_space.flatten(start_dim=-2),
        src=corner_charges.flatten(start_dim=-2),
    )

    charge_grid = flat_charge_grid.reshape(*vector_shape, *histogram_shape)

    return charge_grid

    # TODO: Is this needed? Is this just what in NumPy is called "density=True"?
    # Compute inverse cell volume
    # cell_volume = 1.0
    # for bin_width in bin_widths_for_each_hist_dim:
    #     cell_volume *= bin_width
    # inv_cell_volume = 1.0 / cell_volume
    # charge = charge * inv_cell_volume
