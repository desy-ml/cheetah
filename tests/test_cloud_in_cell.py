import pytest
import torch

from cheetah.utils import is_mps_available_and_functional
from cheetah.utils.cloud_in_cell import cloud_in_cell_charge_deposition


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
def test_1d_compare_histogram(device, dtype):
    """
    Test for the case of a 1D histogram, where all particles are exactly at the center
    of their respective bins that the Cloud-in-Cell charge deposition produces the same
    result as `torch.histogram`.
    """
    factory_kwargs = {"device": device, "dtype": dtype}

    extent = torch.tensor([[0.0, 4.0]], **factory_kwargs)
    bins = 4
    positions = torch.tensor([0.5, 1.5, 3.5], **factory_kwargs).unsqueeze(-1)
    charges = torch.tensor([1.0, 1.0, 2.0], **factory_kwargs)

    cloud_in_cell_result = cloud_in_cell_charge_deposition(
        positions, bins, extent, charges
    )

    histogram_result, _ = torch.histogram(
        positions.squeeze(-1),
        bins=bins,
        range=extent.flatten().tolist(),
        weight=charges,
    )

    assert cloud_in_cell_result.shape == histogram_result.shape
    assert (cloud_in_cell_result == histogram_result).all()
    assert cloud_in_cell_result.dtype == histogram_result.dtype
    assert cloud_in_cell_result.device.type == histogram_result.device.type


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

    cloud_in_cell_result = cloud_in_cell_charge_deposition(
        positions, bins, extent, charges
    )

    histogram_result, _ = torch.histogramdd(
        positions, bins=bins, range=extent.flatten().tolist(), weight=charges
    )

    assert cloud_in_cell_result.shape == histogram_result.shape
    assert (cloud_in_cell_result == histogram_result).all()
    assert cloud_in_cell_result.dtype == histogram_result.dtype
    assert cloud_in_cell_result.device.type == histogram_result.device.type


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
def test_3d_compare_histogramdd(device, dtype):
    """
    Test for the case of a 3D histogram, where all particles are exactly at the center
    of their respective bins that the Cloud-in-Cell charge deposition produces the same
    result as `torch.histogramdd`.
    """
    factory_kwargs = {"device": device, "dtype": dtype}

    extent = torch.tensor([[0.0, 2.0], [0.0, 3.0], [0.0, 4.0]], **factory_kwargs)
    bins = (2, 3, 4)
    positions = torch.tensor(
        [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 2.5, 3.5]], **factory_kwargs
    )
    charges = torch.tensor([1.0, 1.0, 2.0], **factory_kwargs)

    cloud_in_cell_result = cloud_in_cell_charge_deposition(
        positions, bins, extent, charges
    )

    histogram_result, _ = torch.histogramdd(
        positions, bins=bins, range=extent.flatten().tolist(), weight=charges
    )

    assert cloud_in_cell_result.shape == histogram_result.shape
    assert (cloud_in_cell_result == histogram_result).all()
    assert cloud_in_cell_result.dtype == histogram_result.dtype
    assert cloud_in_cell_result.device.type == histogram_result.device.type


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
def test_4d_compare_histogramdd(device, dtype):
    """
    Test for the case of a 4D histogram, where all particles are exactly at the center
    of their respective bins that the Cloud-in-Cell charge deposition produces the same
    result as `torch.histogramdd`.
    """
    factory_kwargs = {"device": device, "dtype": dtype}

    extent = torch.tensor([[0.0, 2.0], [0.0, 3.0], [0.0, 4.0], [0.0, 2.0]], **factory_kwargs)
    bins = (2, 3, 4, 2)
    positions = torch.tensor(
        [[0.5, 0.5, 0.5, 0.5], [1.5, 0.5, 0.5, 1.5], [0.5, 2.5, 3.5, 0.5]], **factory_kwargs
    )
    charges = torch.tensor([1.0, 1.0, 2.0], **factory_kwargs)

    cloud_in_cell_result = cloud_in_cell_charge_deposition(
        positions, bins, extent, charges
    )

    histogram_result, _ = torch.histogramdd(
        positions, bins=bins, range=extent.flatten().tolist(), weight=charges
    )

    assert cloud_in_cell_result.shape == histogram_result.shape
    assert (cloud_in_cell_result == histogram_result).all()
    assert cloud_in_cell_result.dtype == histogram_result.dtype
    assert cloud_in_cell_result.device.type == histogram_result.device.type


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
    Test (for the 2D histogram case) that particles outside the grid bounds do not
    contribute to the charge deposited on the grid.
    """
    extent = torch.tensor([[0.0, 2.0], [0.0, 2.0]])
    bins = (3, 3)
    positions = torch.tensor(
        [[-0.5, 0.5], [1.0, 1.0], [2.5, 1.5]]
    )  # First and last outside of the grid bounds [0.0, 2.0]
    charges = torch.tensor([1.0, 2.0, 3.0])

    result = cloud_in_cell_charge_deposition(positions, bins, extent, charges)

    assert result.sum() < charges.sum()
    assert result.sum() == 2.0  # Only the middle particle should contribute
