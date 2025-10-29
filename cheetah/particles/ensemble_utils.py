import math

import torch


def compute_statistics_1d(
    inputs: torch.Tensor,
    bins: int = 100,
    bin_range: tuple[float] | None = None,
    errorbar: tuple[str, int | float] | str = ("pi", 95),
) -> tuple[torch.Tensor]:
    """
    Compute the mean 1D histogram and a two-sided confidence interval for a
    1-dimensional distribution or multiple vectorised samples of the latter.

    :param inputs: Input samples. Accepts a 1D tensor of shape (num_samples,) or a
        vectorised from with shape (..., num_samples).
    :param bins: Number of histogram bins.
    :param bin_range: Tuple (min, max) specifying the histogram range, or `None` to
        infer from the data.
    :param errorbar: Method to compute uncertainty over vectorised beams. Pass either a
        method string or a tuple `(method, level)`. Available methods are "sd", "se",
        "pi" and "jp".
    :return: Tuple (bin_centers, mean_histogram, lower_bound, upper_bound) with vector
        dimensions reduced.
    """
    histogram, bin_edges = vectorized_histogram_1d(
        inputs, bins=bins, bin_range=bin_range
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if len(inputs.shape) == 1:
        return (bin_centers, histogram, None, None)
    else:
        mean_histogram, lower_bound, upper_bound = compute_mean_and_bounds(
            histogram.to(inputs.dtype).flatten(start_dim=0, end_dim=-2),
            errorbar=errorbar,
        )

        return (bin_centers, mean_histogram, lower_bound, upper_bound)


def compute_statistics_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    bins: tuple[int, int] = (100, 100),
    bin_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
    errorbar: tuple[str, int | float] | str = ("pi", 95),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the mean 2D histogram and uncertainty bounds for paired samples.

    :param x: Values for the x dimension for each sample in the ensemble of shape (B, N)
        or (N,).
    :param y: Values for the y dimension for each sample in the ensemble of shape (B, N)
        or (N,).
    :param bins: Number of histogram bins for x and y as (nx, ny).
    :param bin_ranges: Ranges for the histogram axes as
        ((x_min, x_max), (y_min, y_max)). If None, ranges are inferred from the data.
    :param errorbar: Method to compute uncertainty over vectorised beams. Pass either a
        method string or a tuple `(method, level)`. Available methods are "sd", "se",
        "pi" and "jp".
    :returns: Tuple (x_centers, y_centers, mean_histogram, lower_bound, upper_bound).
    """
    histogram, x_edges, y_edges = vectorized_histogram_2d(
        x, y, bins=bins, bin_ranges=bin_ranges
    )

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    if len(x.shape) == 1:
        return x_centers, y_centers, histogram, None, None
    else:
        mean_histogram, lower_bound, upper_bound = compute_mean_and_bounds(
            histogram.to(x.dtype).flatten(start_dim=0, end_dim=-3), errorbar=errorbar
        )
        return x_centers, y_centers, mean_histogram, lower_bound, upper_bound


def vectorized_histogram_1d(
    inputs: torch.Tensor, bins: int = 100, bin_range: tuple[float] | None = None
) -> tuple[torch.Tensor]:
    """
    Compute a 1-dimensional histogram for each 1D entry in a vectorised tensor of shape
    (..., num_samples).

    :note: Uses `torch.bincount` with offsets.

    :param inputs: Input tensor of shape (..., num_samples).
    :param bins: Number of histogram bins.
        :param bin_range: Tuple (min, max) specifying the histogram range, or `None` to
            infer from the data.
    :return: Tuple (histogram, bin_edges) with `histogram` the bin counts of shape
        (..., bins) and `bin_edges` a 1D tensor of shape (bins + 1,).
    """
    factory_kwargs = {"device": inputs.device, "dtype": inputs.dtype}

    if bin_range is None:
        bin_range = (inputs.min(), inputs.max())

    # If the input is not vectorised, make it vectorised with vector size 1
    was_input_vectorized = inputs.dim() > 1
    if not was_input_vectorized:
        inputs = inputs.unsqueeze(0)  # (1, num_samples)

    # Flatten the vector dimensions for the following computations
    original_vector_shape = inputs.shape[:-1]
    inputs = inputs.flatten(
        start_dim=0, end_dim=-2
    )  # (num_vector_elements, num_samples)
    num_vector_elements = inputs.shape[0]

    bin_edges = torch.linspace(*bin_range, bins + 1, **factory_kwargs)
    bin_indicies = torch.bucketize(inputs.contiguous(), bin_edges) - 1

    # Flatten batch with offsets
    vector_offsets = (
        torch.arange(num_vector_elements, device=inputs.device) * bins
    )  # (num_vector_elements,)
    bin_indicies_flat = (
        bin_indicies + vector_offsets.unsqueeze(-1)
    ).flatten()  # (num_vector_elements * num_bins,)

    # Count occurrences
    histogram_flat = bin_indicies_flat.bincount(
        minlength=num_vector_elements * bins
    )  # (num_vector_elements * bins,)
    histogram = histogram_flat.reshape(*original_vector_shape, bins)

    # Remove the vector dimension if the input was not vectorised
    if not was_input_vectorized:
        histogram = histogram.squeeze(0)

    return histogram, bin_edges


def vectorized_histogram_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    bins: tuple[int] = (100, 100),
    bin_ranges: tuple[tuple[float]] | None = None,
) -> tuple[torch.Tensor]:
    """
    Compute a 2-dimensional histogram for each pair of 2D entries in vectorised tensors.

    :note: Uses `torch.bincount` with offsets.

    :param x: Values for the x dimension for each sample in the ensemble of shape (B, N)
        or (N,).
    :param y: Values for the y dimension for each sample in the ensemble of shape (B, N)
        or (N,).
    :param bins: Number of histogram bins for x and y as (nx, ny).
    :param bin_ranges: Ranges for the histogram axes as
        ((x_min, x_max), (y_min, y_max)). If None, ranges are inferred from the data.
    :return: Tuple (histogram, x_edges, y_edges) with `histogram` the bin counts of
        shape (B, nx, ny) or (nx, ny) and `x_edges`, `y_edges` 1D tensors of shape
        (nx + 1,) and (ny + 1,), respectively.
    """
    factory_kwargs = {"device": x.device, "dtype": x.dtype}

    if bin_ranges is None:
        bin_ranges = (
            (float(x.min()), float(x.max())),
            (float(y.min()), float(y.max())),
        )

    # If the inputs are not vectorised, make them vectorised with vector size 1
    was_input_vectorized = x.dim() > 1
    if not was_input_vectorized:
        x = x.unsqueeze(0)  # (1, N)
        y = y.unsqueeze(0)  # (1, N)

    # Flatten the vector dimensions for the following computations
    original_vector_shape = x.shape[:-1]
    x_flat = x.flatten(start_dim=0, end_dim=-2)  # (num_vector_elements, num_samples)
    y_flat = y.flatten(start_dim=0, end_dim=-2)  # (num_vector_elements, num_samples)
    num_vector_elements = x.shape[0]

    bin_edges_x = torch.linspace(
        bin_ranges[0][0], bin_ranges[0][1], bins[0] + 1, **factory_kwargs
    )
    bin_edges_y = torch.linspace(
        bin_ranges[1][0], bin_ranges[1][1], bins[1] + 1, **factory_kwargs
    )
    bin_indicies_x = torch.bucketize(x_flat.contiguous(), bin_edges_x) - 1
    bin_indicies_y = torch.bucketize(y_flat.contiguous(), bin_edges_y) - 1

    # Flatten 2-dimensional bin indices to 1 dimension
    bin_indicies_flat = bin_indicies_x * bins[1] + bin_indicies_y

    # Flatten batch with offsets
    vector_offsets = torch.arange(num_vector_elements, device=x.device) * (
        bins[0] * bins[1]
    )
    bin_indicies_flat = (bin_indicies_flat + vector_offsets.unsqueeze(1)).flatten()

    # Count occurrences
    histogram_flat = torch.bincount(
        bin_indicies_flat, minlength=num_vector_elements * bins[0] * bins[1]
    )
    histogram = histogram_flat.reshape(*original_vector_shape, *bins)

    # Remove the vector dimension if the input was not vectorised
    if not was_input_vectorized:
        histogram = histogram.squeeze(0)

    return histogram, bin_edges_x, bin_edges_y


def compute_mean_and_bounds(
    inputs: torch.Tensor, errorbar: tuple[str, int | float] | str = ("pi", 95)
):
    """
    Compute elementwise mean and two-sided confidence bounds over an ensemble of
    histograms.

    :param inputs: Ensemble of histograms with shape (n_draws, ...). Computations are
        performed across the first dimension (dim=0).
    :param errorbar: Method to compute uncertainty over vectorised beams. Pass either a
        method string or a tuple `(method, level)`. Available methods are "sd", "se",
        "pi" and "jp".
    :return: Tuple (mean, lower_bound, upper_bound).
    """
    if isinstance(errorbar, str):
        error_method = errorbar
        error_level = None
    else:
        error_method, error_level = errorbar

    mean = inputs.mean(dim=0)

    if error_method == "sd":
        if error_level is None:
            error_level = 3.0

        std_dev = inputs.std(dim=0)
        lower_bound = mean - error_level * std_dev
        upper_bound = mean + error_level * std_dev
    elif error_method == "se":
        if error_level is None:
            error_level = 3.0

        std_error = inputs.std(dim=0) / math.sqrt(inputs.shape[0])
        lower_bound = mean - error_level * std_error
        upper_bound = mean + error_level * std_error
    elif error_method == "pi":
        if error_level is None:
            error_level = 95.0

        alpha = 1 - error_level / 100
        lower_bound = inputs.quantile(alpha / 2, dim=0)
        upper_bound = inputs.quantile(1 - alpha / 2, dim=0)
    else:
        raise ValueError(
            f"Invalid error method: {error_method}. Must be 'sd', 'se', 'pi' or 'jp'."
        )

    return mean, lower_bound, upper_bound
