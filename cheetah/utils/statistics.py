import torch


def unbiased_weighted_covariance(
    input1: torch.Tensor, input2: torch.Tensor, weights: torch.Tensor, dim: int = None
) -> torch.Tensor:
    """
    Compute the unbiased weighted covariance of two tensors.

    :param input1: Input tensor 1. (..., sample_size)
    :param input2: Input tensor 2. (..., sample_size)
    :param weights: Weights tensor. (..., sample_size)
    :param dim: Dimension along which to compute the covariance.
    :return: Unbiased weighted covariance. (..., 2, 2)
    """
    weighted_mean1 = torch.sum(input1 * weights, dim=dim) / torch.sum(weights, dim=dim)
    weighted_mean2 = torch.sum(input2 * weights, dim=dim) / torch.sum(weights, dim=dim)
    correction_factor = torch.sum(weights, dim=dim) - torch.sum(
        weights**2, dim=dim
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
        weights**2, dim=dim
    ) / torch.sum(weights, dim=dim)
    variance = torch.sum(
        weights * (input - weighted_mean.unsqueeze(-1)) ** 2, dim=dim
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

    :param inputs: Input tensor of shape (..., sample_size, n_features).
    :param weights: Weights tensor of shape (..., sample_size).
    :return: Unbiased weighted covariance matrix.
    """
    normalized_weights = weights / weights.sum(dim=-1, keepdim=True)
    correction_factor = 1 - torch.sum(normalized_weights**2, dim=-1)

    weighted_means = torch.sum(
        inputs * normalized_weights.unsqueeze(-1), dim=-2, keepdim=True
    )
    centered_inputs = inputs - weighted_means

    covariance = torch.matmul(
        (normalized_weights.unsqueeze(-1) * centered_inputs).transpose(-1, -2),
        centered_inputs,
    ) / correction_factor.unsqueeze(-1).unsqueeze(-1)
    return covariance
