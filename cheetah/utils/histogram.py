import torch


def _bin_centers_to_edges(centers: torch.Tensor) -> torch.Tensor:
    """Convert a 1D tensor of evenly-spaced bin centers to bin edges."""
    if centers.numel() < 2:
        raise ValueError("Need at least 2 bin centers to infer edges.")

    step = centers[1] - centers[0]
    left_edge = centers[0] - step / 2
    right_edge = centers[-1] + step / 2
    midpoints = (centers[1:] + centers[:-1]) / 2
    return torch.cat([left_edge.unsqueeze(0), midpoints, right_edge.unsqueeze(0)])


def vectorized_histogram_2d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bins1: torch.Tensor,
    bins2: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute batched 2D histograms for coordinate pairs ``(x1, x2)``.

    :param x1: First coordinate tensor with shape ``(*batch_shape, N)``.
    :param x2: Second coordinate tensor, broadcastable with ``x1``.
    :param bins1: Bin centers for the first axis, shape ``(N_bins1,)``.
    :param bins2: Bin centers for the second axis, shape ``(N_bins2,)``.
    :param weights: Optional weights, broadcastable with ``x1`` and ``x2``.
    :returns: Histogram of shape ``(*broadcast_batch_shape, N_bins1, N_bins2)``.
    """
    if weights is None:
        weights = torch.ones_like(x1)

    x1, x2, weights = torch.broadcast_tensors(x1, x2, weights)

    batch_shape = x1.shape[:-1]
    N = x1.shape[-1]

    x1_flat = x1.reshape(-1, N).contiguous()
    x2_flat = x2.reshape(-1, N).contiguous()
    weights_flat = weights.reshape(-1, N)

    B = x1_flat.shape[0]
    device = x1_flat.device
    dtype = x1_flat.dtype

    x1_edges = _bin_centers_to_edges(bins1)
    x2_edges = _bin_centers_to_edges(bins2)
    bins_x1 = x1_edges.numel() - 1
    bins_x2 = x2_edges.numel() - 1

    ix1 = (
        (torch.bucketize(x1_flat, x1_edges) - 1).clamp(0, bins_x1 - 1).long()
    )  # (B, N)
    ix2 = (
        (torch.bucketize(x2_flat, x2_edges) - 1).clamp(0, bins_x2 - 1).long()
    )  # (B, N)

    idx_flat = ix1 * bins_x2 + ix2  # (B, N)

    # Offset each batch so all histograms can be accumulated with one bincount.
    offset = torch.arange(
        B, device=device, dtype=idx_flat.dtype
    ) * (bins_x1 * bins_x2)

    idx_flat_offset = (idx_flat + offset.unsqueeze(1)).reshape(-1)

    weights_flat = weights_flat.reshape(-1).to(dtype)

    valid = (
        (x1_flat >= x1_edges[0])
        & (x1_flat <= x1_edges[-1])
        & (x2_flat >= x2_edges[0])
        & (x2_flat <= x2_edges[-1])
    ).reshape(-1)

    hist_flat = torch.bincount(
        idx_flat_offset[valid],
        weights=weights_flat[valid],
        minlength=B * bins_x1 * bins_x2,
    ).to(dtype)

    hist = hist_flat.view(*batch_shape, bins_x1, bins_x2)

    return hist
