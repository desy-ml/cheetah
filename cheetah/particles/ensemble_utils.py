from typing import Literal

import torch
import torch.nn.functional as F
from scipy import stats


def compute_mean_and_bounds(
    histograms: torch.Tensor,
    uncertainty_type: Literal["percentile", "std_error"] = "percentile",
    confidence_level: float = 0.9,
):
    """
    Compute elementwise mean and two-sided confidence bounds over an
    ensemble of histograms.

    :param histograms: Ensemble of histograms with shape (n_draws, ...).
        Computations are performed across the first dimension (dim=0).
    :param uncertainty_type: Method used to compute the uncertainty
        bounds. One of "percentile" or "std_error". "percentile" uses
        empirical two-sided percentiles at (1 - confidence_level)/2 and
        1 - (1 - confidence_level)/2. "std_error" uses a parametric
        two-sided t-interval based on the sample mean and the standard
        error with the Student's t critical value.
    :param confidence_level: Nominal confidence level for the returned
        interval (0 < confidence_level < 1). Default is 0.9.

    :returns: Tuple (mean_histogram, lower_bound, upper_bound) where:
        - mean_histogram: Pointwise mean across the ensemble with shape
          equal to histograms.shape[1:].
        - lower_bound: Lower bound of the two-sided confidence interval;
          same shape and device as mean_histogram.
        - upper_bound: Upper bound of the two-sided confidence interval;
          same shape and device as mean_histogram.

    :raises ValueError: If uncertainty_type is not one of "percentile"
        or "std_error".
    """
    alpha = 1 - confidence_level
    mean_histogram = torch.mean(histograms, dim=0)

    if uncertainty_type == "percentile":
        lower_bound = torch.quantile(histograms, alpha / 2, dim=0)
        upper_bound = torch.quantile(histograms, 1 - alpha / 2, dim=0)
    elif uncertainty_type == "std_error":
        n = torch.tensor(
            histograms.shape[0], dtype=histograms.dtype, device=histograms.device
        )
        std_error = torch.std(histograms, dim=0, correction=1) / torch.sqrt(n)
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin = t_crit * std_error

        lower_bound = mean_histogram - margin
        upper_bound = mean_histogram + margin
    else:
        raise ValueError(
            "Invalid uncertainty_type: "
            f"{uncertainty_type}. Must be 'percentile' or 'std_error'."
        )

    return mean_histogram, lower_bound, upper_bound


def vectorized_histogram_1d(
    x: torch.Tensor,
    bins: int = 100,
    bin_range: tuple[float] | None = None,
) -> tuple[torch.Tensor]:
    """
    Compute n 1D histograms for a (n, m) tensor using torch.bincount
    with offsets.

    :param x: Input tensor of shape (n, m) or (m,). 1D inputs are
        promoted to shape (1, m).
    :param bins: Number of histogram bins.
    :param bin_range: Optional tuple (min, max) specifying the bin
        edges range. If None the range is inferred.

    :returns: Tuple (hist, bin_edges)
        - hist: Tensor of shape (n, bins) (or (bins,) if input was 1D)
          containing counts per bin.
        - bin_edges: Tensor of length bins + 1 containing the histogram
          bin edges.
    """
    # Make 2D for consistency
    if x.dim() == 1:
        x = x.unsqueeze(0)  # shape (1, m)
    x = x.contiguous()

    n = x.shape[0]
    device = x.device
    dtype = x.dtype

    # Bin edges
    if bin_range is None:
        bin_range = (x.min(), x.max())
    bin_edges = torch.linspace(
        bin_range[0], bin_range[1], bins + 1, device=device, dtype=dtype
    )

    # Bin indices
    idx = torch.bucketize(x, bin_edges) - 1
    idx = idx.clamp(0, bins - 1).long()  # (n, m)

    # Flatten batch with offsets
    offset = torch.arange(n, device=device) * bins  # shape (n,)
    idx_flat = idx + offset.unsqueeze(1)  # shape (n, m)
    idx_flat = idx_flat.flatten()  # shape (n*m,)

    # Count occurrences
    hist_flat = torch.bincount(idx_flat, minlength=n * bins).to(dtype)
    hist = hist_flat.view(n, bins)  # (n, bins)

    # Squeeze if input was 1D
    if hist.size(0) == 1:
        hist = hist.squeeze(0)

    return hist, bin_edges


def vectorized_gaussian_filter_1d(
    hist: torch.Tensor, sigma: float, truncate: float = 4.0
) -> torch.Tensor:
    """
    Apply a 1D Gaussian smoothing filter to one or multiple histograms
    in a vectorized manner.

    :param hist: Input histogram(s). Accepts a 1D tensor of shape (m,)
        for a single histogram or a 2D tensor of shape (n, m) for a
        batch of n histograms. Values, dtype and device are preserved.
    :param sigma: Standard deviation of the Gaussian kernel in units
        of histogram bins.
    :param truncate: Truncate the filter at this many standard
        deviations (default 4.0).
    :return: Smoothed histogram(s) with the same shape, dtype and
        device as the input.
    """
    if hist.dim() == 1:
        hist = hist.unsqueeze(0)  # (1, m)

    device = hist.device
    dtype = hist.dtype

    # Gaussian kernel
    radius = int(truncate * sigma + 0.5)
    pos = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (pos / sigma) ** 2)
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, -1)  # shape (1, 1, k)

    # Prepare input: (B, C, L)
    hist = hist.unsqueeze(1)  # (n, 1, m)

    # Convolve each histogram separately
    smoothed = F.conv1d(hist, kernel, padding=radius)
    smoothed = smoothed.squeeze(1)  # (n, m)

    if smoothed.size(0) == 1:
        smoothed = smoothed.squeeze(0)

    return smoothed


def compute_statistics_1d(
    x: torch.Tensor,
    bins: int = 100,
    bin_range: tuple[float] | None = None,
    smoothing: float = 0.0,
    uncertainty_type: Literal["percentile", "std_error"] = "percentile",
    confidence_level: float = 0.9,
) -> tuple[torch.Tensor]:
    """
    Compute the mean 1D histogram and a two-sided confidence interval
    for an ensemble of samples.

    :param x: Input samples. Accepts a 1D tensor of shape
        (n_samples,) for a single set or a 2D tensor of shape
        (n_draws, n_samples) representing an ensemble; statistics are
        computed across the first dimension for ensembles.
    :param bins: Number of histogram bins (default: 100).
    :param bin_range: Tuple (min, max) range for the histogram bins.
        If None the range is inferred from the provided data.
    :param smoothing: If > 0, apply a Gaussian smoothing kernel with
        this sigma (in bins) to each histogram prior to computing the
        mean and bounds (default: 0.0).
    :param uncertainty_type: Method used to compute the uncertainty
        bounds; either "percentile" or "std_error" (default:
        "percentile").
    :param confidence_level: Nominal confidence level for the returned
        interval in (0, 1) (default: 0.9).

    :returns: Tuple (bin_centers, mean_hist, lower_bound, upper_bound)
        - bin_centers: Tensor of shape (bins,) with the centers of the
          histogram bins.
        - mean_hist: Pointwise mean histogram (shape (bins,)).
        - lower_bound: Lower bound of the two-sided confidence interval
          (same shape as mean_hist).
        - upper_bound: Upper bound of the two-sided confidence interval
          (same shape as mean_hist).
    """
    hist, bin_edges = vectorized_histogram_1d(
        x,
        bins=bins,
        bin_range=bin_range,
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if smoothing:
        hist = vectorized_gaussian_filter_1d(hist, sigma=smoothing)

    mean_hist, lower_bound, upper_bound = compute_mean_and_bounds(
        hist,
        confidence_level=confidence_level,
        uncertainty_type=uncertainty_type,
    )

    return (bin_centers, mean_hist, lower_bound, upper_bound)


def vectorized_histogram_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    bins: tuple[int] = (100, 100),
    bin_ranges: tuple[tuple[float]] | None = None,
) -> tuple[torch.Tensor]:
    """
    Compute batched 2D histograms for coordinate pairs (x, y).

    :param x: Coordinate tensor for x. Shape (N,) for a single set or
        (B, N) for a batch. 1D inputs are promoted to batch size 1.
    :param y: Coordinate tensor for y. Must have the same shape as x.
    :param bins: Number of bins along x and y. If an int is provided
        the same number is used for both axes. If a tuple, use
        (bins_x, bins_y).
    :param bin_ranges: Optional ranges for the x and y axes as
        ((x_min, x_max), (y_min, y_max)). If None, ranges are inferred
        from the data.

    :returns: Tuple (hist, x_edges, y_edges) where:
        - hist is a batched tensor of shape (B, bins_x, bins_y).
          (B=1 if inputs were 1D). Counts are cast to the input dtype
          and device.
        - x_edges and y_edges are 1D tensors of bin edges with lengths
          bins_x+1 and bins_y+1 respectively.
    """
    # Ensure inputs have a batch dimension: promote 1D (N,) -> (1, N)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.contiguous()
    if y.dim() == 1:
        y = y.unsqueeze(0)
    y = y.contiguous()

    # Basic validation
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 1D or 2D tensors")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    bins_x, bins_y = bins

    B = x.shape[0]
    device = x.device
    dtype = x.dtype

    # Infer ranges if not provided, convert to floats
    if bin_ranges is None:
        range_x = (float(x.min()), float(x.max()))
        range_y = (float(y.min()), float(y.max()))
    else:
        range_x = bin_ranges[0]
        range_y = bin_ranges[1]

    # Compute bin edges
    x_edges = torch.linspace(
        range_x[0], range_x[1], bins_x + 1, device=device, dtype=dtype
    )
    y_edges = torch.linspace(
        range_y[0], range_y[1], bins_y + 1, device=device, dtype=dtype
    )

    # Map points to bin indices and clamp
    ix = (torch.bucketize(x, x_edges) - 1).clamp(0, bins_x - 1).long()  # (B, N)
    iy = (torch.bucketize(y, y_edges) - 1).clamp(0, bins_y - 1).long()  # (B, N)

    # Flatten 2D indices to 1D
    idx_flat = ix * bins_y + iy  # shape (B, N)

    # To vectorize: offset indices for each batch
    offset = torch.arange(B, device=device, dtype=idx_flat.dtype) * (bins_x * bins_y)
    idx_flat_offset = (idx_flat + offset.unsqueeze(1)).view(-1)

    # Count occurrences
    hist_flat = torch.bincount(idx_flat_offset, minlength=B * bins_x * bins_y).to(dtype)

    # Reshape to (B, bins_x, bins_y)
    hist = hist_flat.view(B, bins_x, bins_y)

    return hist, x_edges, y_edges


def vectorized_gaussian_filter_2d(
    x: torch.Tensor, sigma: float, truncate: float = 4.0
) -> torch.Tensor:
    """
    Apply a 2D Gaussian filter to a batch of 2D tensors (vectorized).

    :param x: Input tensor of shape (B, H, W).
    :param sigma: Standard deviation of the Gaussian kernel in bins.
    :param truncate: Truncate the filter at this many standard
        deviations. Default is 4.0.
    :returns: Smoothed tensor of shape (B, H, W) with same dtype and
        device as the input.
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)  # (1, H, W)

    device, dtype = x.device, x.dtype

    # 1. Compute 1D Gaussian kernel
    radius = int(truncate * sigma + 0.5)
    kernel_size = 2 * radius + 1
    pos = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-0.5 * (pos / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()

    # 2. Build 2D separable kernel
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()  # normalize again for safety

    # 3. Reshape for conv2d
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    # 4. Add channel dimension
    x = x.unsqueeze(1)  # (B, 1, H, W)

    # 5. Apply convolution
    y = F.conv2d(x, kernel_2d, padding=radius)

    # 6. Remove channel dimension
    y = y.squeeze(1)  # (B, H, W)
    if y.size(0) == 1:
        y = y.squeeze(0)

    return y


def compute_statistics_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    bins: tuple[int, int] = (100, 100),
    bin_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
    smoothing: float = 0.0,
    uncertainty_type: Literal["percentile", "std_error"] = "percentile",
    confidence_level: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the mean 2D histogram and uncertainty bounds for paired
    samples.

    :param x: Values for the x dimension for each sample in the
        ensemble. Shape (B, N) or (N,).
    :param y: Values for the y dimension for each sample in the
        ensemble. Shape (B, N) or (N,).
    :param bins: Number of histogram bins for x and y as (nx, ny).
    :param bin_ranges: Ranges for the histogram axes as
        ((x_min, x_max), (y_min, y_max)). If None, ranges are inferred
        from the data.
    :param smoothing: Standard deviation (sigma) for an optional
        Gaussian smoothing applied to the mean histogram. A value of
        0.0 disables smoothing.
    :param uncertainty_type: Method to compute uncertainty bounds.
        Allowed values:
        - "percentile": use empirical percentiles of the ensemble
          histograms
        - "std_error": use mean ± (standard error * t critical value)
    :param confidence_level: Confidence level for the uncertainty
        bounds (0 < confidence_level < 1).
    :return: Tuple (x_centers, y_centers, mean_hist, lower_bound,
        upper_bound)
        - x_centers: Centers of the x histogram bins (length nx).
        - y_centers: Centers of the y histogram bins (length ny).
        - mean_hist: Mean 2D histogram across the ensemble with shape
          (nx, ny).
        - lower_bound: Lower bound of the confidence interval for each
          bin (shape (nx, ny)).
        - upper_bound: Upper bound of the confidence interval for each
          bin (shape (nx, ny)).
    """
    # Construct per-sample 2D histograms (batched)
    hist, x_edges, y_edges = vectorized_histogram_2d(
        x,
        y,
        bins=bins,
        bin_ranges=bin_ranges,
    )

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Compute ensemble mean and bounds across batch dimension
    mean_hist, lower_bound, upper_bound = compute_mean_and_bounds(
        hist, confidence_level=confidence_level, uncertainty_type=uncertainty_type
    )

    # Optionally smooth the mean histogram
    if smoothing:
        mean_hist = vectorized_gaussian_filter_2d(mean_hist, sigma=smoothing)

    return x_centers, y_centers, mean_hist, lower_bound, upper_bound
