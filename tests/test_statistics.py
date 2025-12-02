import pytest
import torch

from cheetah.utils import (
    distribution_histogram_and_confidence_1d,
    distribution_histogram_and_confidence_2d,
    histograms_mean_and_confidence,
    match_distribution_moments,
    unbiased_weighted_covariance,
    unbiased_weighted_covariance_matrix,
    unbiased_weighted_variance,
    vectorized_histogram_1d,
    vectorized_histogram_2d,
)


def test_unbiased_weighted_variance_with_single_element():
    """Test that the variance is NaN when there is only one element."""
    data = torch.tensor([42.0])
    weights = torch.tensor([1.0])

    computed_variance = unbiased_weighted_variance(data, weights)

    assert computed_variance.isnan()


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

    assert computed_variance.isnan()


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
    data = torch.stack([series, series.square(), series.pow(3)], dim=-1)
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
    data = torch.stack([series, series.square(), series.pow(3)], dim=-1)
    weights = torch.tensor([0.5, 1.0, 1.0, 0.9, 0.9])

    expected_covariance_matrix = torch.cov(data.mT, aweights=weights)
    computed_covariance_matrix = unbiased_weighted_covariance_matrix(data, weights)

    assert torch.allclose(computed_covariance_matrix, expected_covariance_matrix)


def test_match_distribution_moments():
    """
    Test that the first and second moments of the samples after transformation are
    matched to the target values.
    """

    # Randomly sample from a normal distribution
    samples = torch.randn(1000, 3, dtype=torch.float64)
    weights = torch.ones(1000, dtype=torch.float64)

    # Define target moments
    target_mu = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    # Randomly generate a target covariance matrix
    target_L = torch.randn(3, 3, dtype=torch.float64)
    target_cov = target_L @ target_L.mT

    # Transform samples to match target moments
    transformed_samples = match_distribution_moments(
        samples, target_mu, target_cov, weights=weights
    )

    # Compute moments of transformed samples
    transformed_mu = (transformed_samples * weights.unsqueeze(-1)).sum(
        dim=-2
    ) / weights.sum(dim=-1, keepdim=True)
    transformed_cov = unbiased_weighted_covariance_matrix(transformed_samples, weights)

    # Check if moments match
    assert torch.allclose(transformed_mu, target_mu)
    assert torch.allclose(transformed_cov, target_cov)


@pytest.mark.parametrize("errorbar", ["sd", "se", "pi"])
@pytest.mark.parametrize("histogram_dimensions", [1, 2])
def test_histogram_and_confidence_1d(errorbar, histogram_dimensions):
    """
    Test that the output shapes of `histograms_mean_and_confidence` are correct, and
    that lower and upper bounds are below and above the mean, respectively.
    """
    num_histograms = 100
    histogram_shape = (20,) if histogram_dimensions == 1 else (20, 30)
    histograms = torch.rand(num_histograms, *histogram_shape)

    mean_histogram, lower_bound, upper_bound = histograms_mean_and_confidence(
        histograms, errorbar=errorbar
    )

    assert mean_histogram.shape == histogram_shape
    assert lower_bound.shape == histogram_shape
    assert upper_bound.shape == histogram_shape

    assert (lower_bound <= mean_histogram).all()
    assert (upper_bound >= mean_histogram).all()


@pytest.mark.parametrize(
    "vector_shape", [None, (4,), (3, 2)], ids=["(,)", "(4,)", "(3,2)"]
)
def test_vectorized_histogram_1d(vector_shape):
    """Test that the output shape of `vectorized_histogram_1d` is correct."""
    num_distribution_samples = 100
    num_bins = 15

    distribution_shape = (
        (vector_shape + (num_distribution_samples,))
        if vector_shape is not None
        else (num_distribution_samples,)
    )
    distribution = torch.rand(*distribution_shape)

    histogram, bin_edges = vectorized_histogram_1d(distribution, bins=num_bins)

    expected_histogram_shape = (
        (*vector_shape, num_bins) if vector_shape is not None else (num_bins,)
    )

    assert histogram.shape == expected_histogram_shape
    assert bin_edges.shape == (num_bins + 1,)


@pytest.mark.parametrize(
    "vector_shape", [None, (4,), (3, 2)], ids=["(,)", "(4,)", "(3,2)"]
)
def test_vectorized_histogram_2d(vector_shape):
    """Test that the output shape of `vectorized_histogram_2d` is correct."""
    num_distribution_samples = 100
    num_bins = (15, 20)

    distribution_shape = (
        (vector_shape + (num_distribution_samples,))
        if vector_shape is not None
        else (num_distribution_samples,)
    )
    x_distribution = torch.rand(*distribution_shape)
    y_distribution = torch.rand(*distribution_shape)

    histogram, bin_edges_x, bin_edges_y = vectorized_histogram_2d(
        x_distribution, y_distribution, bins=num_bins
    )

    expected_histogram_shape = (
        (*vector_shape, num_bins[0], num_bins[1])
        if vector_shape is not None
        else (num_bins[0], num_bins[1])
    )

    assert histogram.shape == expected_histogram_shape
    assert bin_edges_x.shape == (num_bins[0] + 1,)
    assert bin_edges_y.shape == (num_bins[1] + 1,)


@pytest.mark.parametrize(
    "vector_shape", [None, (4,), (3, 2)], ids=["(,)", "(4,)", "(3,2)"]
)
def test_distribution_histogram_and_confidence_1d(vector_shape):
    """
    Test that the output shapes of `distribution_histogram_and_confidence_1d` are
    correct, and that lower and upper bounds are below and above the mean, respectively.
    """
    num_distribution_samples = 100
    num_bins = 20

    distribution_shape = (
        (vector_shape + (num_distribution_samples,))
        if vector_shape is not None
        else (num_distribution_samples,)
    )
    distribution = torch.rand(*distribution_shape)

    bin_centers, mean_histogram, lower_bound, upper_bound = (
        distribution_histogram_and_confidence_1d(
            distribution, bins=num_bins, errorbar="sd"
        )
    )

    assert bin_centers.shape == (num_bins,)
    assert mean_histogram.shape == (num_bins,)

    if vector_shape is not None:
        assert lower_bound.shape == (num_bins,)
        assert upper_bound.shape == (num_bins,)

        assert (lower_bound <= mean_histogram).all()
        assert (upper_bound >= mean_histogram).all()


@pytest.mark.parametrize(
    "vector_shape", [None, (4,), (3, 2)], ids=["(,)", "(4,)", "(3,2)"]
)
def test_distribution_histogram_and_confidence_2d(vector_shape):
    """
    Test that the output shapes of `distribution_histogram_and_confidence_2d` are
    correct, and that lower and upper bounds are below and above the mean, respectively.
    """
    num_distribution_samples = 100
    num_bins = (15, 20)

    distribution_shape = (
        (vector_shape + (num_distribution_samples,))
        if vector_shape is not None
        else (num_distribution_samples,)
    )
    x_distribution = torch.rand(*distribution_shape)
    y_distribution = torch.rand(*distribution_shape)

    (
        bin_centers_x,
        bin_centers_y,
        mean_histogram,
        lower_bound,
        upper_bound,
    ) = distribution_histogram_and_confidence_2d(
        x_distribution, y_distribution, bins=num_bins, errorbar="sd"
    )

    assert bin_centers_x.shape == (num_bins[0],)
    assert bin_centers_y.shape == (num_bins[1],)
    assert mean_histogram.shape == (num_bins[0], num_bins[1])

    if vector_shape is not None:
        assert lower_bound.shape == (num_bins[0], num_bins[1])
        assert upper_bound.shape == (num_bins[0], num_bins[1])

        assert (lower_bound <= mean_histogram).all()
        assert (upper_bound >= mean_histogram).all()
