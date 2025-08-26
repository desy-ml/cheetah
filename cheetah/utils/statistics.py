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
    weights_sum = weights.sum(dim=dim)
    weighted_mean1 = (input1 * weights).sum(dim=dim) / weights_sum
    weighted_mean2 = (input2 * weights).sum(dim=dim) / weights_sum
    correction_factor = weights_sum - (weights * weights).sum(dim=dim) / weights_sum
    covariance = (
        weights
        * (input1 - weighted_mean1.unsqueeze(-1))
        * (input2 - weighted_mean2.unsqueeze(-1))
    ).sum(dim=dim) / correction_factor

    return covariance


def unbiased_weighted_variance(
    input1: torch.Tensor, weights: torch.Tensor, dim: int = None
) -> torch.Tensor:
    """
    Compute the unbiased weighted variance of a tensor.

    :param input1: Input tensor.
    :param weights: Weights tensor.
    :param dim: Dimension along which to compute the variance.
    :return: Unbiased weighted variance.
    """
    weights_sum = weights.sum(dim=dim)
    weighted_mean = (input1 * weights).sum(dim=dim) / weights_sum
    correction_factor = weights_sum - (weights * weights).sum(dim=dim) / weights_sum
    variance = (weights * (input1 - weighted_mean.unsqueeze(-1)).square()).sum(
        dim=dim
    ) / correction_factor

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
    return (unbiased_weighted_variance(input, weights, dim=dim)).sqrt()


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
    correction_factor = 1 - (normalized_weights * normalized_weights).sum(dim=-1)

    weighted_means = (inputs * normalized_weights.unsqueeze(-1)).sum(
        dim=-2, keepdim=True
    )
    centered_inputs = inputs - weighted_means

    covariance = (
        (normalized_weights.unsqueeze(-1) * centered_inputs).mT @ centered_inputs
    ) / correction_factor.unsqueeze(-1).unsqueeze(-1)

    return covariance
