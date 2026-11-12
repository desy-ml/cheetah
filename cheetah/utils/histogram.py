from typing import Sequence

import torch


def vectorized_histogram_2d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bins: int | Sequence[int],
    extent: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute vectorised 2D histograms for coordinate pairs `(x1, x2)`.

    :param x1: First coordinate tensor with shape `(*vector_shape, num_particles)`.
    :param x2: Second coordinate tensor with shape `(*vector_shape, num_particles)`.
    :param bins: Single int or sequence of two ints specifying the number of
        bins along the `x1` and `x2` axes.
    :param extent: Tensor of shape `(..., 2, 2)` or `(..., 4)` specifying
        `[[x1_min, x1_max], [x2_min, x2_max]]`. If `None`, the extent is inferred from
        the minimum and maximum of `x1` and `x2`.
    :param weights: Optional weights, broadcastable with `x1`.
    :return: Histogram of shape `(*vector_shape, num_bins_x1, num_bins_x2)`.
    """
    vector_shape = x1.shape[:-1]

    if isinstance(bins, int):
        num_bins_x1 = num_bins_x2 = bins
    else:
        num_bins_x1, num_bins_x2 = bins[0], bins[1]

    if extent is None:
        extent = torch.stack(
            [
                torch.stack([x1.amin(dim=-1), x1.amax(dim=-1)], dim=-1),
                torch.stack([x2.amin(dim=-1), x2.amax(dim=-1)], dim=-1),
            ],
            dim=-2,
        )
    elif extent.shape[-1] == 4:
        extent = extent.unflatten(-1, (2, 2))

    x1_min = extent[..., 0, 0].unsqueeze(-1)
    x1_max = extent[..., 0, 1].unsqueeze(-1)
    x2_min = extent[..., 1, 0].unsqueeze(-1)
    x2_max = extent[..., 1, 1].unsqueeze(-1)

    inv_bin_width_x1 = num_bins_x1 / (x1_max - x1_min)
    inv_bin_width_x2 = num_bins_x2 / (x2_max - x2_min)

    bin_idx_x1 = (
        ((x1 - x1_min) * inv_bin_width_x1).floor().long().clamp(0, num_bins_x1 - 1)
    )
    bin_idx_x2 = (
        ((x2 - x2_min) * inv_bin_width_x2).floor().long().clamp(0, num_bins_x2 - 1)
    )

    flat_bin_idx = bin_idx_x1 * num_bins_x2 + bin_idx_x2

    in_bounds_mask = (x1 >= x1_min) & (x1 <= x1_max) & (x2 >= x2_min) & (x2 <= x2_max)
    masked_weights = (
        in_bounds_mask.to(x1.dtype) if weights is None else weights * in_bounds_mask
    )

    histogram = (
        x1.new_zeros(*vector_shape, num_bins_x1 * num_bins_x2)
        .scatter_add_(-1, flat_bin_idx, masked_weights)
        .view(*vector_shape, num_bins_x1, num_bins_x2)
    )

    return histogram
