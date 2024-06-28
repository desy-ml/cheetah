import pytest
import torch

import cheetah


def test_bmad_tutorial():
    """Test importing the lattice example file from the Bmad and Tao tutorial."""
    file_path = "tests/resources/bmad_tutorial_lattice.bmad"
    converted = cheetah.Segment.from_bmad(file_path)
    converted.name = "bmad_tutorial"

    correct = cheetah.Segment(
        [
            cheetah.Drift(length=torch.tensor(0.5), name="d"),
            cheetah.Dipole(
                length=torch.tensor(0.5), e1=torch.tensor(0.1), name="b"
            ),  # TODO: What are g and dg?
            cheetah.Quadrupole(
                length=torch.tensor(0.6), k1=torch.tensor(0.23), name="q"
            ),
        ],
        name="bmad_tutorial",
    )

    assert converted.name == correct.name
    assert [element.name for element in converted.elements] == [
        element.name for element in correct.elements
    ]
    assert converted.d.length == correct.d.length
    assert converted.b.length == correct.b.length
    assert converted.b.e1 == correct.b.e1
    assert converted.q.length == correct.q.length
    assert converted.q.k1 == correct.q.k1


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
                not torch.backends.mps.is_available(), reason="MPS not available"
            ),
        ),
    ],
)
def test_device_passing(device: torch.device):
    """Test that the device is passed correctly."""
    file_path = "tests/resources/bmad_tutorial_lattice.bmad"

    # Convert the lattice while passing the device
    converted = cheetah.Segment.from_bmad(file_path, device=device)

    # Check that the properties of the loaded elements are on the correct device
    assert converted.d.length.device.type == device.type
    assert converted.b.length.device.type == device.type
    assert converted.b.e1.device.type == device.type
    assert converted.q.length.device.type == device.type
    assert converted.q.k1.device.type == device.type


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_passing(dtype: torch.dtype):
    """Test that the dtype is passed correctly."""
    file_path = "tests/resources/bmad_tutorial_lattice.bmad"

    # Convert the lattice while passing the dtype
    converted = cheetah.Segment.from_bmad(file_path, dtype=dtype)

    # Check that the properties of the loaded elements are of the correct dtype
    assert converted.d.length.dtype == dtype
    assert converted.b.length.dtype == dtype
    assert converted.b.e1.dtype == dtype
    assert converted.q.length.dtype == dtype
    assert converted.q.k1.dtype == dtype
