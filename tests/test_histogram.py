import pytest
import torch

from cheetah.utils import is_mps_available_and_functional, vectorized_histogram_2d


@pytest.mark.parametrize("bins", [10, (10, 15)])
@pytest.mark.parametrize(
    "device, dtype",
    [
        (torch.device("cpu"), torch.float32),
        (torch.device("cpu"), torch.float64),
        pytest.param(
            torch.device("cuda"),
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            torch.float32,
            marks=pytest.mark.skipif(
                not is_mps_available_and_functional(), reason="MPS not available"
            ),
        ),
    ],
    ids=["cpu-float32", "cpu-float64", "cuda-float32", "mps-float32"],
)
def test_compare_torch_histogramdd(bins, device, dtype):
    """
    Test that `vectorized_histogram_2d` matches `torch.histogramdd` for non-vectorised
    particle tensors.
    """
    factory_kwargs = {"device": device, "dtype": dtype}
    num_particles = 1_000
    extent = torch.tensor([[-1.0, 1.0], [-2.0, 2.0]], **factory_kwargs)
    x1 = torch.rand(num_particles, **factory_kwargs) * 2.0 - 1.0
    x2 = torch.rand(num_particles, **factory_kwargs) * 4.0 - 2.0
    weights = torch.rand(num_particles, **factory_kwargs) * 0.9 + 0.1

    num_bins = [bins, bins] if isinstance(bins, int) else list(bins)

    actual = vectorized_histogram_2d(x1, x2, bins=bins, extent=extent, weights=weights)

    expected, _ = torch.histogramdd(
        torch.stack([x1.cpu(), x2.cpu()], dim=-1),
        bins=num_bins,
        range=extent.cpu().flatten().tolist(),
        weight=weights.cpu(),
    )

    assert actual.shape == tuple(num_bins)
    assert actual.dtype == dtype
    assert actual.device.type == device.type
    assert torch.allclose(actual.cpu(), expected)


@pytest.mark.parametrize(
    "x_shape, weights_shape",
    [
        ((2, 3, 500), None),
        ((2, 3, 500), (500,)),
        ((2, 3, 500), (2, 3, 500)),
        ((500,), (2, 3, 500)),
    ],
    ids=[
        "no-weights",
        "non-vectorised-weights",
        "vectorised-weights",
        "vectorised-weights-only",
    ],
)
def test_vectorized_shapes_and_broadcasting(x_shape, weights_shape):
    """
    Test that `vectorized_histogram_2d` produces the expected output shape for
    multidimensional vector shapes and broadcastable weights.
    """
    bins = (8, 6)
    extent = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])

    x1 = torch.randn(*x_shape)
    x2 = torch.randn(*x_shape)
    weights = None if weights_shape is None else torch.rand(*weights_shape)

    hist = vectorized_histogram_2d(x1, x2, bins=bins, extent=extent, weights=weights)

    assert hist.shape == (2, 3, 8, 6)


def test_inferred_extent():
    """
    Test that `vectorized_histogram_2d` infers the extent from data when `extent=None`
    and matches `torch.histogramdd`.
    """
    num_particles = 500
    bins = (8, 6)
    x1 = torch.rand(num_particles) * 2.0 - 1.0
    x2 = torch.rand(num_particles) * 4.0 - 2.0

    actual = vectorized_histogram_2d(x1, x2, bins=bins, extent=None)

    expected, _ = torch.histogramdd(torch.stack([x1, x2], dim=-1), bins=list(bins))

    assert actual.shape == bins
    assert torch.allclose(actual, expected)


def test_flat_extent():
    """
    Test that `vectorized_histogram_2d` accepts a flat `(4,)` extent tensor and matches
    `torch.histogramdd`.
    """
    num_particles = 500
    bins = (8, 6)
    x1 = torch.rand(num_particles) * 2.0 - 1.0
    x2 = torch.rand(num_particles) * 4.0 - 2.0
    extent = torch.tensor([-1.0, 1.0, -2.0, 2.0])

    actual = vectorized_histogram_2d(x1, x2, bins=bins, extent=extent)

    expected, _ = torch.histogramdd(
        torch.stack([x1, x2], dim=-1),
        bins=list(bins),
        range=extent.tolist(),
    )

    assert actual.shape == bins
    assert torch.allclose(actual, expected)


def test_out_of_bounds_particles():
    """Test that particles outside extent are excluded and boundary points included."""
    x1 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    x2 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    extent = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])

    hist = vectorized_histogram_2d(x1, x2, bins=2, extent=extent)

    # 5 particles: 2 out of bounds (-2 and +2), 3 in bounds (-1, 0, 1)
    assert hist.sum().item() == 3.0
