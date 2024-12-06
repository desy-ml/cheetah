import numpy as np
import pytest
import torch

import cheetah
from cheetah.utils import is_mps_available_and_functional


def test_fodo():
    """Test importing a FODO lattice defined in the Elegant file format."""
    file_path = "tests/resources/fodo.lte"
    converted = cheetah.Segment.from_elegant(file_path, "fodo")

    correct_lattice = cheetah.Segment(
        [
            cheetah.Marker(name="c"),
            cheetah.Quadrupole(
                name="q1", length=torch.tensor(0.1), k1=torch.tensor(1.5)
            ),
            cheetah.Drift(name="d1", length=torch.tensor(1.0)),
            cheetah.Marker(name="m1"),
            cheetah.Dipole(
                name="s1", length=torch.tensor(0.3), dipole_e1=torch.tensor(0.25)
            ),
            cheetah.Drift(name="d1", length=torch.tensor(1.0)),
            cheetah.Quadrupole(
                name="q2", length=torch.tensor(0.2), k1=torch.tensor(-3.0)
            ),
            cheetah.Drift(name="d2", length=torch.tensor(2.0)),
        ],
        name="fodo",
    )

    assert converted.name == correct_lattice.name
    assert [element.name for element in converted.elements] == [
        element.name for element in correct_lattice.elements
    ]
    assert torch.isclose(converted.q1.length, correct_lattice.q1.length)
    assert torch.isclose(converted.q1.k1, correct_lattice.q1.k1)
    assert torch.isclose(converted.q2.length, correct_lattice.q2.length)
    assert torch.isclose(converted.q2.k1, correct_lattice.q2.k1)
    for i in range(2):
        assert torch.isclose(converted.d1[i].length, correct_lattice.d1[i].length)
    assert torch.isclose(converted.d2.length, correct_lattice.d2.length)
    assert torch.isclose(converted.s1.length, correct_lattice.s1.length)
    assert torch.isclose(converted.s1.dipole_e1, correct_lattice.s1.dipole_e1)


def test_cavity_import():
    """Test importing an accelerating cavity defined in the Elegant file format."""
    file_path = "tests/resources/cavity.lte"
    converted = cheetah.Segment.from_elegant(file_path, "cavity")

    assert np.isclose(converted.c1.length, 0.7)
    assert np.isclose(converted.c1.frequency, 1.2e9)
    assert np.isclose(converted.c1.voltage, 16.175e6)

    # Cheetah and Elegant use different phase conventions shifted by 90 deg
    assert np.isclose(converted.c1.phase, 0.0)


def test_custom_transfer_map_import():
    """Test importing an Elegant EMATRIX into a Cheetah CustomTransferMap."""
    file_path = "tests/resources/cavity.lte"
    converted = cheetah.Segment.from_elegant(file_path, "cavity")

    correct_transfer_map = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.04, 1.0, 0.003, 0.0, 0.0, 0.0, -0.0027],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.003, 0.0, -0.04, 1.0, 0.0, 0.0, -0.15],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert torch.allclose(converted.c1e.predefined_transfer_map, correct_transfer_map)


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            marks=pytest.mark.skipif(
                not is_mps_available_and_functional(), reason="MPS not available"
            ),
        ),
    ],
)
def test_device_passing(device: torch.device):
    """Test that the device is passed correctly."""
    file_path = "tests/resources/fodo.lte"

    # Convert the lattice while passing the device
    converted = cheetah.Segment.from_elegant(file_path, "fodo", device=device)

    # Check that the properties of the loaded elements are on the correct device
    assert converted.q1.length.device.type == device.type
    assert converted.q1.k1.device.type == device.type
    assert converted.q2.length.device.type == device.type
    assert converted.q2.k1.device.type == device.type
    assert [d.length.device.type for d in converted.d1] == [device.type, device.type]
    assert converted.d2.length.device.type == device.type
    assert converted.s1.length.device.type == device.type
    assert converted.s1.dipole_e1.device.type == device.type


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_passing(dtype: torch.dtype):
    """Test that the dtype is passed correctly."""
    file_path = "tests/resources/fodo.lte"

    # Convert the lattice while passing the device
    converted = cheetah.Segment.from_elegant(file_path, "fodo", dtype=dtype)

    # Check that the properties of the loaded elements are of the correct dtype
    assert converted.q1.length.dtype == dtype
    assert converted.q1.k1.dtype == dtype
    assert converted.q2.length.dtype == dtype
    assert converted.q2.k1.dtype == dtype
    assert [d.length.dtype for d in converted.d1] == [dtype, dtype]
    assert converted.d2.length.dtype == dtype
    assert converted.s1.length.dtype == dtype
    assert converted.s1.dipole_e1.dtype == dtype
