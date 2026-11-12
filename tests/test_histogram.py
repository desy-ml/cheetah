import pytest
import torch

from cheetah.utils import vectorized_histogram_2d


@pytest.mark.parametrize(
    "bins",
    [10, (10, 15)],
)
def test_compare_torch_histogramdd(bins):
    """
    Test that `vectorized_histogram_2d` matches `torch.histogramdd` for 1D batch.
    """
    num_particles = 1_000
    extent = torch.tensor([[-1.0, 1.0], [-2.0, 2.0]])
    x1 = torch.empty(num_particles).uniform_(-1.0, 1.0)
    x2 = torch.empty(num_particles).uniform_(-2.0, 2.0)
    weights = torch.empty(num_particles).uniform_(0.1, 1.0)

    num_bins = [bins, bins] if isinstance(bins, int) else list(bins)
    expected, _ = torch.histogramdd(
        torch.stack([x1, x2], dim=-1),
        bins=num_bins,
        range=extent.flatten().tolist(),
        weight=weights,
    )

    actual = vectorized_histogram_2d(x1, x2, bins=bins, extent=extent, weights=weights)

    assert actual.shape == tuple(num_bins)
    assert torch.allclose(actual, expected)


def test_vectorized_shapes_and_broadcasting():
    """
    Test `vectorized_histogram_2d` with multidimensional batch and broadcasting weights.
    """
    num_particles = 500
    bins = (8, 6)
    extent = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])

    # 3D vector shape: (2, 3, num_particles)
    x1 = torch.randn(2, 3, num_particles)
    x2 = torch.randn(2, 3, num_particles)

    # 1. Without weights
    hist_noweights = vectorized_histogram_2d(x1, x2, bins=bins, extent=extent)
    assert hist_noweights.shape == (2, 3, 8, 6)

    # 2. Unvectorised weights (num_particles,)
    weights_unvec = torch.rand(num_particles)
    hist_unvec = vectorized_histogram_2d(
        x1, x2, bins=bins, extent=extent, weights=weights_unvec
    )
    assert hist_unvec.shape == (2, 3, 8, 6)

    # 3. Vectorised weights (2, 3, num_particles)
    weights_vec = torch.rand(2, 3, num_particles)
    hist_vec = vectorized_histogram_2d(
        x1, x2, bins=bins, extent=extent, weights=weights_vec
    )
    assert hist_vec.shape == (2, 3, 8, 6)


def test_out_of_bounds_particles():
    """Test that particles outside extent are excluded and boundary points included."""
    x1 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    x2 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    extent = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])

    hist = vectorized_histogram_2d(x1, x2, bins=2, extent=extent)
    # 5 particles: 2 out of bounds (-2 and +2), 3 in bounds (-1, 0, 1)
    assert hist.sum().item() == 3.0
