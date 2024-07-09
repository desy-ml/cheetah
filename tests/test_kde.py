import torch

from cheetah.utils import kde_histogram_1d, kde_histogram_2d


def test_weighted_samples_1d():
    """
    Test that the 1d KDE histogram implementation correctly handles
    heterogeneously weighted samples.
    """
    x_unweighted = torch.tensor([1.0, 1.0, 1.0, 2.0])
    x_weighted = torch.tensor([1.0, 2.0])

    bins = torch.linspace(0, 3, 10)
    sigma = torch.tensor(0.3)

    # Explicitly use all the samples with the same weights
    hist_unweighted = kde_histogram_1d(x_unweighted, bins, sigma)
    # Use samples and taking the weights into account
    hist_weighted = kde_histogram_1d(
        x_weighted, bins, sigma, weights=torch.tensor([3.0, 1.0])
    )
    # Use samples but neglect the weights
    hist_neglect_weights = kde_histogram_1d(x_weighted, bins, sigma)

    assert torch.allclose(hist_unweighted, hist_weighted)
    assert not torch.allclose(hist_weighted, hist_neglect_weights)


def test_weighted_samples_2d():
    """
    Test that the 2d KDE histogram implementation correctly handles
    heterogeneously weighted samples.
    """
    x_unweighted = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [2.0, 1.0]])
    x_weighted = torch.tensor([[1.0, 2.0], [2.0, 1.0]])

    bins1 = torch.linspace(0, 3, 10)
    bins2 = torch.linspace(0, 3, 10)
    sigma = torch.tensor(0.3)

    # Explicitly use all the samples with the same weights
    hist_unweighted = kde_histogram_2d(
        x_unweighted[:, 0], x_unweighted[:, 1], bins1, bins2, sigma
    )
    # Use samples and taking the weights into account
    hist_weighted = kde_histogram_2d(
        x_weighted[:, 0],
        x_weighted[:, 1],
        bins1,
        bins2,
        sigma,
        weights=torch.tensor([3.0, 1.0]),
    )
    # Use samples but neglect the weights
    hist_neglect_weights = kde_histogram_2d(
        x_weighted[:, 0], x_weighted[:, 1], bins1, bins2, sigma
    )
    print(hist_unweighted[5])
    print(hist_weighted[5])
    print(hist_neglect_weights[5])
    assert torch.allclose(hist_unweighted, hist_weighted)
    assert not torch.allclose(hist_weighted, hist_neglect_weights)
