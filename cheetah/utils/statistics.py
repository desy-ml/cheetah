import math

import torch


def unbiased_weighted_covariance(
    inputs1: torch.Tensor, inputs2: torch.Tensor, weights: torch.Tensor, dim: int = None
) -> torch.Tensor:
    """
    Compute the unbiased weighted covariance of two tensors.

    :param input1: Input tensor 1 of shape (..., sample_size).
    :param input2: Input tensor 2 of shape (..., sample_size).
    :param weights: Weights tensor of shape (..., sample_size).
    :param dim: Dimension along which to compute the covariance.
    :return: Unbiased weighted covariance of shape (..., 2, 2).
    """
    weighted_mean1 = (inputs1 * weights).sum(dim=dim) / weights.sum(dim=dim)
    weighted_mean2 = (inputs2 * weights).sum(dim=dim) / weights.sum(dim=dim)
    correction_factor = weights.sum(dim=dim) - weights.square().sum(
        dim=dim
    ) / weights.sum(dim=dim)
    covariance = (
        weights
        * (inputs1 - weighted_mean1.unsqueeze(-1))
        * (inputs2 - weighted_mean2.unsqueeze(-1))
    ).sum(dim=dim) / correction_factor

    return covariance


def unbiased_weighted_variance(
    inputs: torch.Tensor, weights: torch.Tensor, dim: int = None
) -> torch.Tensor:
    """
    Compute the unbiased weighted variance of a tensor.

    :param inputs: Input tensor.
    :param weights: Weights tensor.
    :param dim: Dimension along which to compute the variance.
    :return: Unbiased weighted variance.
    """
    sum_of_weights = weights.sum(dim=dim)
    weighted_mean = (inputs * weights).sum(dim=dim) / sum_of_weights
    correction_factor = sum_of_weights - weights.square().sum(dim=dim) / sum_of_weights
    variance = (weights * (inputs - weighted_mean.unsqueeze(-1)).square()).sum(
        dim=dim
    ) / correction_factor

    return variance


def unbiased_weighted_std(
    inputs: torch.Tensor, weights: torch.Tensor, dim: int = None
) -> torch.Tensor:
    """
    Compute the unbiased weighted standard deviation of a tensor.

    :param inputs: Input tensor.
    :param weights: Weights tensor.
    :param dim: Dimension along which to compute the standard deviation.
    :return: Unbiased weighted standard deviation.
    """
    return unbiased_weighted_variance(inputs, weights, dim=dim).sqrt()


def unbiased_weighted_covariance_matrix(
    inputs: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """
    Compute the unbiased weighted covariance matrix of a tensor.

    :param inputs: Input tensor of shape (..., sample_size, num_features).
    :param weights: Weights tensor of shape (..., sample_size).
    :return: Unbiased weighted covariance matrix.
    """
    normalized_weights = weights / weights.sum(dim=-1, keepdim=True)
    correction_factor = 1 - normalized_weights.square().sum(dim=-1)

    weighted_means = (inputs * normalized_weights.unsqueeze(-1)).sum(
        dim=-2, keepdim=True
    )
    centered_inputs = inputs - weighted_means

    covariance = torch.matmul(
        (normalized_weights.unsqueeze(-1) * centered_inputs).mT,
        centered_inputs,
    ) / correction_factor.unsqueeze(-1).unsqueeze(-1)

    return covariance


def match_distribution_moments(
    samples: torch.Tensor,
    target_mu: torch.Tensor,
    target_cov: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Match the first and second moments of a sample distribution to a target
    distribution.

    :param samples: Input samples of shape (..., num_samples, num_features).
    :param target_mu: Mean of the target distribution of shape (..., num_features).
    :param target_cov: Covariance of the target distribution of shape
        (..., num_features, num_features).
    :param weights: Weights for the samples of shape (..., num_samples).
    :return: Transformed samples.
    """
    factory_kwargs = {"device": samples.device, "dtype": samples.dtype}
    num_samples = samples.shape[-2]
    num_features = samples.shape[-1]

    # Compute the inverse square root of the sample covariance
    if weights is None:
        weights = torch.ones_like(samples[..., 0])
    sample_cov = unbiased_weighted_covariance_matrix(samples, weights)
    sample_mu = (samples * weights.unsqueeze(-1)).sum(dim=-2) / weights.sum(
        dim=-1, keepdim=True
    )
    cholesky_sample_cov = torch.linalg.cholesky(sample_cov).contiguous()
    inverse_sqrt_sample_cov = torch.linalg.solve_triangular(
        cholesky_sample_cov,
        torch.eye(cholesky_sample_cov.shape[-1], **factory_kwargs),
        upper=False,
    )

    vector_shape = torch.broadcast_shapes(target_mu.shape[:-1], target_cov.shape[:-2])
    broadcasted_sample_mu = sample_mu.expand(*vector_shape, num_features).unsqueeze(-2)
    broadcasted_inverse_sqrt_sample_cov = inverse_sqrt_sample_cov.expand(
        *vector_shape, num_features, num_features
    )

    mu = target_mu.expand(*vector_shape, num_features).unsqueeze(-2)
    cov = target_cov.expand(*vector_shape, num_features, num_features)

    # Compute the Cholesky decomposition
    chol_cov = torch.linalg.cholesky(cov)

    transformed_samples = samples.expand(*vector_shape, num_samples, num_features)
    transformed_samples = (transformed_samples - broadcasted_sample_mu) @ (
        chol_cov @ broadcasted_inverse_sqrt_sample_cov
    ).mT + mu

    return transformed_samples


def distribution_histogram_and_confidence_1d(
    a: torch.Tensor,
    bins: int = 100,
    bin_range: tuple[float] | None = None,
    errorbar: tuple[str, int | float] | str = ("pi", 95),
) -> tuple[torch.Tensor]:
    """
    Compute the mean histogram and confidence interval over vectorised samples of a
    1-dimensional distribution.

    Also works for non-vectorised inputs, in which case the histogram is computed and
    `None` is returned for the confidence bounds.

    :param a: Input distribution. Can be a single distribution of shape (num_samples,)
        or a vectorisation of multiple distributions of shape (..., num_samples).
    :param bins: Number of histogram bins.
    :param bin_range: Tuple (min, max) specifying the histogram range, or `None` to
        infer from the data.
    :param errorbar: Method to compute uncertainty over vectorised beams. Pass either a
        method string or a tuple `(method, level)`. Available methods are "sd", "se" and
        "pi".
    :return: Tuple (bin_centers, mean_histogram, lower_bound, upper_bound) with vector
        dimensions reduced.
    """
    histogram, bin_edges = vectorized_histogram_1d(a, bins=bins, bin_range=bin_range)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if len(a.shape) == 1:
        return (bin_centers, histogram, None, None)
    else:
        mean_histogram, lower_bound, upper_bound = histograms_mean_and_confidence(
            histogram.to(a.dtype).flatten(start_dim=0, end_dim=-2), errorbar=errorbar
        )

        return (bin_centers, mean_histogram, lower_bound, upper_bound)


def distribution_histogram_and_confidence_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    bins: tuple[int, int] = (100, 100),
    bin_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
    errorbar: tuple[str, int | float] | str = ("pi", 95),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the mean histogram and confidence interval over vectorised samples of a
    2-dimensional distribution.

    Also works for non-vectorised inputs, in which case the histogram is computed and
    `None` is returned for the confidence bounds.

    :param x: x coordinates of the distribution to be histogrammed. Can be a single
        distribution of shape (num_samples,) or a vectorisation of multiple
        distributions of shape (..., num_samples).
    :param y: y coordinates of the distribution to be histogrammed. Can be a single
        distribution of shape (num_samples,) or a vectorisation of multiple
        distributions of shape (..., num_samples).
    :param bins: Tuple (nx, ny) specifying the number of histogram bins for x and y.
    :param bin_ranges: Tuple ((x_min, x_max), (y_min, y_max)) specifying the histogram
        ranges for x and y, or `None` to infer from the data.
    :param errorbar: Method to compute uncertainty over vectorised beams. Pass either a
        method string or a tuple `(method, level)`. Available methods are "sd", "se" and
        "pi".
    :returns: Tuple (bin_centers_x, bin_centers_y, mean_histogram, lower_bound,
        upper_bound) with vector dimensions reduced.
    """
    histogram, x_edges, y_edges = vectorized_histogram_2d(
        x, y, bins=bins, bin_ranges=bin_ranges
    )

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    if len(x.shape) == 1:
        return x_centers, y_centers, histogram, None, None
    else:
        mean_histogram, lower_bound, upper_bound = histograms_mean_and_confidence(
            histogram.to(x.dtype).flatten(start_dim=0, end_dim=-3), errorbar=errorbar
        )
        return x_centers, y_centers, mean_histogram, lower_bound, upper_bound


def vectorized_histogram_1d(
    a: torch.Tensor, bins: int = 100, bin_range: tuple[float] | None = None
) -> tuple[torch.Tensor]:
    """
    Compute a histogram for a 1-dimensional distribution or multiple vectorised
    histograms for vectorised distributions.

    :param a: Input distribution. Can be a single distribution of shape (num_samples,)
        or a vectorisation of multiple distributions of shape (..., num_samples).
    :param bins: Number of histogram bins.
    :param bin_range: Tuple (min, max) specifying the histogram range, or `None` to
        infer from the data.
    :return: Tuple (histogram, bin_edges) with `histogram` the bin counts of shape
        (..., bins) and `bin_edges` a 1-dimensional tensor of shape (bins + 1,).
    """
    factory_kwargs = {"device": a.device, "dtype": a.dtype}

    if bin_range is None:
        bin_range = (a.min(), a.max())

    # If the input is not vectorised, make it vectorised with vector size 1
    was_input_vectorized = a.dim() > 1
    if not was_input_vectorized:
        a = a.unsqueeze(0)  # (1, num_samples)

    # Flatten the vector dimensions for the following computations
    original_vector_shape = a.shape[:-1]
    a = a.flatten(start_dim=0, end_dim=-2)  # (num_vector_elements, num_samples)
    num_vector_elements = a.shape[0]

    bin_edges = torch.linspace(*bin_range, bins + 1, **factory_kwargs)
    boundaries = bin_edges[1:-1]
    bin_indicies = torch.bucketize(a.contiguous(), boundaries)

    # Flatten batch with offsets
    vector_offsets = (
        torch.arange(num_vector_elements, device=a.device) * bins
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
    bins: tuple[int, int] = (100, 100),
    bin_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[torch.Tensor]:
    """
    Compute a histogram for a 2-dimensional distribution or multiple vectorised
    histograms for vectorised distributions.

    :param x: x coordinates of the distribution to be histogrammed. Can be a single
        distribution of shape (num_samples,) or a vectorisation of multiple
        distributions of shape (..., num_samples).
    :param y: y coordinates of the distribution to be histogrammed. Can be a single
        distribution of shape (num_samples,) or a vectorisation of multiple
        distributions of shape (..., num_samples).
    :param bins: Tuple (nx, ny) specifying the number of histogram bins for x and y.
    :param bin_ranges: Tuple ((x_min, x_max), (y_min, y_max)) specifying the histogram
        ranges for x and y, or `None` to infer from the data.
    :return: Tuple (histogram, bin_edges_x, bin_edges_y) with `histogram` the bin counts
        of shape (..., nx, ny) or (nx, ny) and `bin_edges_x`, `bin_edges_y`
        1-dimensional tensors of shape (nx + 1,) and (ny + 1,), respectively.
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
    num_vector_elements = x_flat.shape[0]

    bin_edges_x = torch.linspace(
        bin_ranges[0][0], bin_ranges[0][1], bins[0] + 1, **factory_kwargs
    )
    bin_edges_y = torch.linspace(
        bin_ranges[1][0], bin_ranges[1][1], bins[1] + 1, **factory_kwargs
    )
    boundaries_x = bin_edges_x[1:-1]
    boundaries_y = bin_edges_y[1:-1]
    bin_indicies_x = torch.bucketize(x_flat.contiguous(), boundaries_x)
    bin_indicies_y = torch.bucketize(y_flat.contiguous(), boundaries_y)

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


def histograms_mean_and_confidence(
    histograms: torch.Tensor, errorbar: tuple[str, int | float] | str = ("pi", 95)
):
    """
    Compute elementwise mean and two-sided confidence bounds over an ensemble of
    histograms.

    NOTE: This function assumes exactly one vector dimension. If you have multiple
        vector dimensions, you must flatten them before passing to this function.

    :param histograms: Tensor of multiple histograms with shape
        (vector_size, *single_histogram_shape).
    :param errorbar: Method to compute uncertainty over vectorised beams. Pass either a
        method string or a tuple `(method, level)`. Available methods are "sd", "se" and
        "pi".
    :return: Tuple of the mean histogram, lower bound and upper bound, with each the
        vector dimensions reduced.
    """
    if isinstance(errorbar, str):
        error_method = errorbar
        error_level = None
    else:
        error_method, error_level = errorbar

    mean = histograms.mean(dim=0)

    if error_method == "sd":
        if error_level is None:
            error_level = 3.0

        std_dev = histograms.std(dim=0)
        lower_bound = mean - error_level * std_dev
        upper_bound = mean + error_level * std_dev
    elif error_method == "se":
        if error_level is None:
            error_level = 3.0

        std_error = histograms.std(dim=0) / math.sqrt(histograms.shape[0])
        lower_bound = mean - error_level * std_error
        upper_bound = mean + error_level * std_error
    elif error_method == "pi":
        if error_level is None:
            error_level = 95.0

        alpha = 1 - error_level / 100
        lower_bound = histograms.quantile(alpha / 2, dim=0)
        upper_bound = histograms.quantile(1 - alpha / 2, dim=0)
    else:
        raise ValueError(
            f"Invalid error method: {error_method}. Must be 'sd', 'se' and 'pi'."
        )

    return mean, lower_bound, upper_bound
