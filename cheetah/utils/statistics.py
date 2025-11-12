import torch


def unbiased_weighted_covariance(
    input1: torch.Tensor, input2: torch.Tensor, weights: torch.Tensor, dim: int = None
) -> torch.Tensor:
    """
    Compute the unbiased weighted covariance of two tensors.

    :param input1: Input tensor 1 of shape (..., sample_size).
    :param input2: Input tensor 2 of shape (..., sample_size).
    :param weights: Weights tensor of shape (..., sample_size).
    :param dim: Dimension along which to compute the covariance.
    :return: Unbiased weighted covariance of shape (..., 2, 2).
    """
    weighted_mean1 = torch.sum(input1 * weights, dim=dim) / torch.sum(weights, dim=dim)
    weighted_mean2 = torch.sum(input2 * weights, dim=dim) / torch.sum(weights, dim=dim)
    correction_factor = torch.sum(weights, dim=dim) - torch.sum(
        weights.square(), dim=dim
    ) / torch.sum(weights, dim=dim)
    covariance = torch.sum(
        weights
        * (input1 - weighted_mean1.unsqueeze(-1))
        * (input2 - weighted_mean2.unsqueeze(-1)),
        dim=dim,
    ) / (correction_factor)
    return covariance


def unbiased_weighted_variance(
    input: torch.Tensor, weights: torch.Tensor, dim: int = None
) -> torch.Tensor:
    """
    Compute the unbiased weighted variance of a tensor.

    :param input: Input tensor.
    :param weights: Weights tensor.
    :param dim: Dimension along which to compute the variance.
    :return: Unbiased weighted variance.
    """
    weighted_mean = torch.sum(input * weights, dim=dim) / torch.sum(weights, dim=dim)
    correction_factor = torch.sum(weights, dim=dim) - torch.sum(
        weights.square(), dim=dim
    ) / torch.sum(weights, dim=dim)
    variance = torch.sum(
        weights * (input - weighted_mean.unsqueeze(-1)).square(), dim=dim
    ) / (correction_factor)
    return variance


def unbiased_weighted_std(
    input: torch.Tensor, weights: torch.Tensor, dim: int = None
) -> torch.Tensor:
    """
    Compute the unbiased weighted standard deviation of a tensor.

    :param input: Input tensor.
    :param weights: Weights tensor.
    :param dim: Dimension along which to compute the standard deviation.
    :return: Unbiased weighted standard deviation.
    """
    return torch.sqrt(unbiased_weighted_variance(input, weights, dim=dim))


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
    correction_factor = 1 - torch.sum(normalized_weights.square(), dim=-1)

    weighted_means = torch.sum(
        inputs * normalized_weights.unsqueeze(-1), dim=-2, keepdim=True
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
