import torch

from cheetah.utils import unbiased_weighted_covariance, unbiased_weighted_variance


def test_unbiased_weighted_variance_with_same_weights():
    """Test that the variance is calculated correctly with equal weights."""
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    expected_variance = torch.var(data, unbiased=True)
    calculated_variance = unbiased_weighted_variance(data, weights)
    assert torch.allclose(calculated_variance, expected_variance)


def test_unbiased_weighted_variance_with_single_element():
    """Test that the variance is nan when there is only one element."""
    data = torch.tensor([42.0])
    weights = torch.tensor([1.0])
    assert torch.isnan(unbiased_weighted_variance(data, weights))


def test_unbiased_weighted_variance_with_different_weights():
    """Test that the variance is calculated correctly with different weights."""
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = torch.tensor([0.5, 0.5, 0.5, 0, 0])
    expected_variance = torch.var(torch.tensor([1.0, 2.0, 3.0]), unbiased=True)
    calculated_variance = unbiased_weighted_variance(data, weights)
    assert torch.allclose(calculated_variance, expected_variance)


def test_unbiased_weighted_variance_with_zero_weights():
    """Test that the variance is nan when all weights are zero."""
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.isnan(unbiased_weighted_variance(data, weights))


def test_unbiased_weighted_variance_with_small_numbers():
    """Test that the variance is calculated correctly with small numbers."""
    data = torch.tensor([1e-10, 2e-10, 3e-10, 4e-10, 5e-10], dtype=torch.float32)
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    expected_variance = torch.var(data, unbiased=True)
    calculated_variance = unbiased_weighted_variance(data, weights)
    assert torch.allclose(calculated_variance, expected_variance)


def test_unbiased_weighted_covariance_reduced_to_variance():
    """Test that the covariance calculation is reduced to the variance when both inputs
    are the same.
    """
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    equal_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    expected_variance = torch.var(data, unbiased=True)
    calculated_covariance = unbiased_weighted_covariance(data, data, equal_weights)
    assert torch.allclose(calculated_covariance, expected_variance)

    different_weights = torch.tensor([0.5, 1.0, 1.0, 0.9, 0.9])
    assert torch.allclose(
        unbiased_weighted_covariance(data, data, different_weights),
        unbiased_weighted_variance(data, different_weights),
    )
