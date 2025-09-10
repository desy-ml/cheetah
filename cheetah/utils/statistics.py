import torch


def unbiased_weighted_covariance(
    inputs1: torch.Tensor, inputs2: torch.Tensor, weights: torch.Tensor, dim: int = None
) -> torch.Tensor:
    """
    Compute the unbiased weighted covariance of two tensors.

    :param inputs1: Input tensor 1. (..., sample_size)
    :param inputs2: Input tensor 2. (..., sample_size)
    :param weights: Weights tensor. (..., sample_size)
    :param dim: Dimension along which to compute the covariance.
    :return: Unbiased weighted covariance. (..., 2, 2)
    """
    weighted_mean1 = (inputs1 * weights).sum(dim=dim) / weights.sum(dim=dim)
    weighted_mean2 = (inputs2 * weights).sum(dim=dim) / weights.sum(dim=dim)
    correction_factor = weights.sum(dim=dim) - (weights**2).sum(
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
    weighted_mean = (inputs * weights).sum(dim=dim) / weights.sum(dim=dim)
    correction_factor = weights.sum(dim=dim) - (weights**2).sum(
        dim=dim
    ) / weights.sum(dim=dim)
    variance = (weights * (inputs - weighted_mean.unsqueeze(-1)) ** 2).sum(
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

    :param inputs: Input tensor of shape (..., sample_size, n_features).
    :param weights: Weights tensor of shape (..., sample_size).
    :return: Unbiased weighted covariance matrix.
    """
    normalized_weights = weights / weights.sum(dim=-1, keepdim=True)
    correction_factor = 1 - (normalized_weights**2).sum(dim=-1)

    weighted_means = (inputs * normalized_weights.unsqueeze(-1)).sum(
        dim=-2, keepdim=True
    )
    centered_inputs = inputs - weighted_means

    covariance = torch.matmul(
        (normalized_weights.unsqueeze(-1) * centered_inputs).transpose(-1, -2),
        centered_inputs,
    ) / correction_factor.unsqueeze(-1).unsqueeze(-1)

    return covariance
