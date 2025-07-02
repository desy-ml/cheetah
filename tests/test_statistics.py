import torch

from cheetah.utils import (
    unbiased_weighted_covariance,
    unbiased_weighted_covariance_matrix,
    unbiased_weighted_variance,
)


def test_unbiased_weighted_variance_with_single_element():
    """Test that the variance is NaN when there is only one element."""
    data = torch.tensor([42.0])
    weights = torch.tensor([1.0])

    computed_variance = unbiased_weighted_variance(data, weights)

    assert torch.isnan(computed_variance)


def test_unbiased_weighted_variance_with_same_weights():
    """
    Test that the weighted variance with all weights the same equals the unweighted
    variance implementated in PyTorch.
    """
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    expected_variance = torch.var(data)
    computed_variance = unbiased_weighted_variance(data, weights)

    assert torch.allclose(computed_variance, expected_variance)


def test_unbiased_weighted_variance_with_different_weights():
    """Test that the variance is computed when some weights are different."""
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = torch.tensor([0.5, 0.5, 0.5, 0, 0])

    expected_variance = torch.var(torch.tensor([1.0, 2.0, 3.0]))
    computed_variance = unbiased_weighted_variance(data, weights)

    assert torch.allclose(computed_variance, expected_variance)


def test_unbiased_weighted_variance_with_zero_weights():
    """Test that the variance is NaN when all weights are zero."""
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

    computed_variance = unbiased_weighted_variance(data, weights)

    assert torch.isnan(computed_variance)


def test_unbiased_weighted_variance_with_small_numbers():
    """Test that the variance is correct for small numbers."""
    data = torch.tensor([1e-10, 2e-10, 3e-10, 4e-10, 5e-10], dtype=torch.float32)
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    expected_variance = torch.var(data)
    computed_variance = unbiased_weighted_variance(data, weights)

    assert torch.allclose(computed_variance, expected_variance)


def test_unbiased_weighted_covariance_reduced_to_variance():
    """
    Test that the covariance computation is correctly reduced to the variance when both
    inputs are the same.
    """
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = torch.tensor([0.5, 1.0, 1.0, 0.9, 0.9])

    variance = unbiased_weighted_variance(data, weights)
    covariance = unbiased_weighted_covariance(data, data, weights)

    assert torch.allclose(covariance, variance)


def test_unbiased_weighted_covariance_matrix_reduced_to_scalar():
    """
    Test that the unbiased weighted covariance matrix calculation reduces to the
    variance for scalar variables.
    """
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = torch.tensor([0.5, 1.0, 1.0, 0.9, 0.9])

    matrix = unbiased_weighted_covariance_matrix(data.unsqueeze(-1), weights)
    variance = unbiased_weighted_variance(data, weights)

    assert torch.allclose(matrix.squeeze(), variance)


def test_unbiased_weighted_covariance_matrix_elementwise_reduction():
    """
    Test that the unbiased weighted covariance matrix agrees with the covariance if
    calculated elementwise.
    """
    series = torch.arange(5.0)
    data = torch.stack([series, series**2, series**3], dim=-1)
    weights = torch.tensor([[0.5, 1.0, 1.0, 0.9, 0.9], [0.4, 1.2, 1.4, 0.6, 0.7]])

    matrix = unbiased_weighted_covariance_matrix(data, weights)

    for i in range(3):
        for j in range(3):
            covariance = unbiased_weighted_covariance(
                data[:, i], data[:, j], weights, dim=-1
            )
            assert torch.allclose(matrix[:, i, j], covariance)


def test_unbiased_weighted_covariance_matrix_torch():
    """
    Test that the unbiased weighted covariance matrix is equal to the result computed
    by torch for the unvectorized case.
    """
    series = torch.arange(5.0)
    data = torch.stack([series, series**2, series**3], dim=-1)
    weights = torch.tensor([0.5, 1.0, 1.0, 0.9, 0.9])

    expected_covariance_matrix = torch.cov(data.T, aweights=weights)
    computed_covariance_matrix = unbiased_weighted_covariance_matrix(data, weights)

    assert torch.allclose(computed_covariance_matrix, expected_covariance_matrix)
