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

    bin_widths = (extent[..., 1] - extent[..., 0]) / torch.tensor(
        histogram_shape, device=positions.device
    )

    # Set weights to zero for particles outside grid bounds
    inside_mask = ((extent[..., 0] <= positions) & (positions < extent[..., 1])).all(
        dim=-1
    )  # Shape (..., num_samples)
    masked_charges = charges * inside_mask  # Shape (..., num_samples)

    # Normalise particle coordinates to normalised bin space
    positions_in_bin_space = (positions - extent[..., 0]) / bin_widths[-1]
    positions_in_bin_space_int_component = positions_in_bin_space.floor().long()
    positions_in_bin_space_fractional_components = (
        positions_in_bin_space - positions_in_bin_space_int_component
    )

    # Generate all corner combinations and their weights
    num_corners = 2**num_hist_dims
    # TODO speed: torch.tensor(list(itertools.product([0, 1], repeat=D)), device=device)
    corner_offsets = (
        torch.arange(num_corners).unsqueeze(-1).to(positions.device)
        // (2 ** torch.arange(num_hist_dims).to(positions.device))
        % 2
    )
    corner_positions_in_bin_space = (
        positions_in_bin_space_int_component.unsqueeze(-2) + corner_offsets
    )
    clamped_corner_positions_in_bin_space = corner_positions_in_bin_space.clamp(
        corner_positions_in_bin_space.new_zeros(()),
        corner_positions_in_bin_space.new_tensor(histogram_shape) - 1,
    )
    corner_weight_factors = positions_in_bin_space_fractional_components.unsqueeze(
        -2
    ).where(
        corner_offsets == 1,
        1.0 - positions_in_bin_space_fractional_components.unsqueeze(-2),
    )  # Shape (..., num_samples, num_corners, num_hist_dims)
    corner_weights = corner_weight_factors.prod(
        dim=-1
    )  # Shape (..., num_samples, num_corners)
    corner_charges = (  # Actual charge deposition on the corners
        masked_charges.unsqueeze(-1) * corner_weights
    )  # Shape (..., num_samples, num_corners)

    vector_shape = positions.shape[:-2]
    charge_grid = corner_charges.new_zeros(*vector_shape, *histogram_shape)

    vector_indices = tuple(
        torch.arange(this_vector_dim_length).reshape(
            this_vector_dim_length, *([1] * (num_hist_dims - dim_idx + 1))
        )
        for dim_idx, this_vector_dim_length in enumerate(vector_shape)
    )

    charge_grid = charge_grid.index_put(
        vector_indices + clamped_corner_positions_in_bin_space.unbind(-1),
        corner_charges,
        accumulate=True,
    )

    return charge_grid

    # TODO: Is this needed? Is this just what in NumPy is called "density=True"?
    # Compute inverse cell volume
    # cell_volume = 1.0
    # for bin_width in bin_widths_for_each_hist_dim:
    #     cell_volume *= bin_width
    # inv_cell_volume = 1.0 / cell_volume
    # charge = charge * inv_cell_volume
