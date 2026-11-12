import torch


def vectorized_histogram_2d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bins1: torch.Tensor,
    bins2: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute vectorised 2D histograms for coordinate pairs ``(x1, x2)``.

    :param x1: First coordinate tensor with shape ``(*vector_shape, num_particles)``.
    :param x2: Second coordinate tensor, broadcastable with ``x1``.
    :param bins1: Bin centers for the first axis, shape ``(num_bins_x1,)``.
    :param bins2: Bin centers for the second axis, shape ``(num_bins_x2,)``.
    :param weights: Optional weights, broadcastable with ``x1`` and ``x2``.
    :returns: Histogram of shape
        ``(*broadcast_vector_shape, num_bins_x1, num_bins_x2)``.
    """
    if weights is None:
        weights = torch.ones_like(x1)

    x1, x2, weights = torch.broadcast_tensors(x1, x2, weights)

    vector_shape = x1.shape[:-1]
    dtype = x1.dtype

    x1 = x1.contiguous()
    x2 = x2.contiguous()

    x1_edges = _bin_centers_to_edges(bins1)
    x2_edges = _bin_centers_to_edges(bins2)
    num_bins_x1 = x1_edges.numel() - 1
    num_bins_x2 = x2_edges.numel() - 1

    use_int32 = (num_bins_x1 * num_bins_x2) < 2_147_483_647

    ix1 = (torch.bucketize(x1, x1_edges, out_int32=use_int32) - 1).clamp(
        0, num_bins_x1 - 1
    )
    ix2 = (torch.bucketize(x2, x2_edges, out_int32=use_int32) - 1).clamp(
        0, num_bins_x2 - 1
    )

    idx = ix1 * num_bins_x2 + ix2

    # Filter out-of-bounds particles without dynamically sized tensor allocations
    valid = (
        (x1 >= x1_edges[0])
        & (x1 <= x1_edges[-1])
        & (x2 >= x2_edges[0])
        & (x2 <= x2_edges[-1])
    )
    masked_weights = (weights * valid).to(dtype)

    hist = x1.new_zeros(*vector_shape, num_bins_x1, num_bins_x2)
    hist.view(*vector_shape, -1).scatter_add_(dim=-1, index=idx, src=masked_weights)

    return hist


def _bin_centers_to_edges(centers: torch.Tensor) -> torch.Tensor:
    """Convert a 1D tensor of evenly-spaced bin centers to bin edges."""
    if centers.numel() < 2:
        raise ValueError("Need at least 2 bin centers to infer edges.")

    step = centers[1] - centers[0]
    return torch.linspace(
        centers[0] - step / 2.0,
        centers[-1] + step / 2.0,
        centers.numel() + 1,
        device=centers.device,
        dtype=centers.dtype,
    )
