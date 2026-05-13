import pytest
import torch

from cheetah.utils import is_mps_available_and_functional
from cheetah.utils.cloud_in_cell import cloud_in_cell_charge_deposition


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_deposit_charge_cic_1d_basic(device):
    """Test basic 1D CIC charge deposition."""
    x = torch.tensor([0.5, 1.5], device=device, dtype=torch.float32)
    bins = torch.tensor([0.0, 1.0, 2.0], device=device)

    result = cloud_in_cell_charge_deposition_1d(x, bins)

    # Particle at 0.5 -> 0.5 to bin 0, 0.5 to bin 1
    # Particle at 1.5 -> 0.5 to bin 1, 0.5 to bin 2
    expected = torch.tensor([0.5, 1.0, 0.5], device=device)

    assert result.shape == (3,)
    assert torch.allclose(result, expected, atol=1e-6)


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
def test_2d_compare_histogramdd(device, dtype):
    """
    Test for the case of a 2D histogram, where all particles are exactly at the center
    of their respective bins that the Cloud-in-Cell charge deposition produces the same
    result as `torch.histogramdd`.
    """
    factory_kwargs = {"device": device, "dtype": dtype}

    extent = torch.tensor([[0.0, 4.0], [0.0, 3.0]], **factory_kwargs)
    bins = (4, 3)
    positions = torch.tensor([[0.5, 0.5], [1.5, 0.5], [3.5, 1.5]], **factory_kwargs)
    charges = torch.tensor([1.0, 1.0, 2.0], **factory_kwargs)

    # Simple test case with known expected result
    cloud_in_cellresult = cloud_in_cell_charge_deposition(
        positions, bins, extent, charges
    )

    histogram_result, _ = torch.histogramdd(
        positions, bins=bins, range=extent.flatten().tolist(), weight=charges
    )

    expected = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        **factory_kwargs
    )

    assert cloud_in_cellresult.shape == histogram_result.shape
    assert (cloud_in_cellresult == expected).all()
    assert cloud_in_cellresult.dtype == histogram_result.dtype
    assert cloud_in_cellresult.device.type == histogram_result.device.type


def test_deposit_charge_cic_3d_basic():
    """Test basic 3D CIC charge deposition."""
    x1 = torch.tensor([0.5])
    x2 = torch.tensor([0.5])
    x3 = torch.tensor([0.5])

    bins1 = torch.tensor([0.0, 1.0])
    bins2 = torch.tensor([0.0, 1.0])
    bins3 = torch.tensor([0.0, 1.0])

    result = cloud_in_cell_charge_deposition_3d(x1, x2, x3, bins1, bins2, bins3)

    # Particle at (0.5, 0.5, 0.5) should contribute equally to all 8 corners
    expected = torch.ones(2, 2, 2) * 0.125

    assert result.shape == (2, 2, 2)
    assert torch.allclose(result, expected, atol=1e-6)


def test_2d_vectorized():
    """
    Test that the vectorised 2D Cloud-in-Cell charge deposition can handle
    multi-dimensional vector inputs and produces the expected output shape. This test
    does not check the numerical correctness of the output, just that it can process the
    input without errors and produces an output of the correct shape.
    """
    x1 = torch.tensor(
        [
            [[0.5, 1.5, 2.8, 3.0, 4.3], [1.2, 2.3, 3.7, 0.9, 1.4]],
            [[0.6, 1.4, 2.2, 3.4, 0.5], [0.9, 1.1, 2.4, 3.6, 0.7]],
        ]
    )
    x2 = torch.tensor(
        [
            [[0.2, 1.7, 2.9, 3.1, 4.4], [1.4, 2.5, 3.6, 4.8, 0.2]],
            [[0.7, 1.3, 2.0, 3.5, 4.6], [0.8, 1.2, 2.3, 3.7, 4.9]],
        ]
    )

    ranges = torch.tensor([[0.0, 3.0], [0.0, 4.0]])

    result = cloud_in_cell_charge_deposition(
        positions=torch.stack([x1, x2], dim=-1), bins=[3, 4], extent=ranges
    )

    # Should be (vector_dim1, vector_dim2, histogram_dim1, histogram_dim2)
    assert result.shape == (2, 2, 3, 4)


def test_2d_some_outside_bounds():
    """
    Test behaviour of 2D Cloud-in-Cell charge deposition with some particles outside of
    grid bounds. The sum of the deposited should match the sum of the weights of the
    particles and be less than the total input charge.
    """
    x1 = torch.tensor([-0.5, 1.0, 2.5])  # First and last outside grid bounds [0, 2]
    x2 = torch.tensor([0.5, 1.0, 1.5])
    weights = torch.tensor([1.0, 2.0, 3.0])

    bins1 = torch.tensor([0.0, 1.0, 2.0])
    bins2 = torch.tensor([0.0, 1.0, 2.0])

    result = cloud_in_cell_charge_deposition_2d(x1, x2, bins1, bins2, weights)

    # Should not raise errors
    assert result.shape == (3, 3)
    assert not result.isnan().any()
    assert not result.isinf().any()

    # Only the middle particle (inside bounds) should contribute, so total deposited
    # charge should be less than total input charge.
    assert result.sum() == 2.0
    assert result.sum() < weights.sum()
