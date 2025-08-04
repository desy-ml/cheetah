import pytest
import torch

import cheetah
from cheetah.utils import NotUnderstoodPropertyWarning, is_mps_available_and_functional


def test_bmad_tutorial():
    """Test importing the lattice example file from the Bmad and Tao tutorial."""
    file_path = "tests/resources/bmad_tutorial_lattice.bmad"
    with pytest.warns(NotUnderstoodPropertyWarning, match="( d | g | dg )"):
        converted = cheetah.Segment.from_bmad(file_path)
    converted.name = "bmad_tutorial"

    correct = cheetah.Segment(
        [
            cheetah.Drift(length=torch.tensor(0.5), name="d"),
            cheetah.Dipole(
                length=torch.tensor(0.5), dipole_e1=torch.tensor(0.1), name="b"
            ),  # TODO: What are g and dg?
            cheetah.Drift(length=torch.tensor(-0.4), name="n"),
            cheetah.Quadrupole(
                length=torch.tensor(0.6), k1=torch.tensor(0.23), name="q"
            ),
            cheetah.Sextupole(
                length=torch.tensor(0.3),
                k2=torch.tensor(0.42),
                tilt=torch.tensor(-0.1),
                name="s",
            ),
            cheetah.Drift(length=torch.tensor(-0.6), name="v"),
        ],
        name="bmad_tutorial",
    )

    assert converted.name == correct.name
    assert converted.length == correct.length
    assert [element.name for element in converted.elements] == [
        element.name for element in correct.elements
    ]
    assert converted.d.length == correct.d.length
    assert converted.b.length == correct.b.length
    assert converted.b.dipole_e1 == correct.b.dipole_e1
    assert converted.n.length == correct.n.length
    assert converted.q.length == correct.q.length
    assert converted.q.k1 == correct.q.k1
    assert converted.s.length == correct.s.length
    assert converted.s.k2 == correct.s.k2
    assert converted.s.tilt == correct.s.tilt
    assert converted.v.length == correct.v.length


@pytest.mark.filterwarnings(
    r"ignore:.*( d | g | dg ).*:cheetah.utils.NotUnderstoodPropertyWarning"
)
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
    file_path = "tests/resources/bmad_tutorial_lattice.bmad"

    # Convert the lattice while passing the device
    converted = cheetah.Segment.from_bmad(file_path, device=device)

    # Check that the properties of the loaded elements are on the correct device
    assert converted.d.length.device.type == device.type
    assert converted.b.length.device.type == device.type
    assert converted.b.dipole_e1.device.type == device.type
    assert converted.q.length.device.type == device.type
    assert converted.q.k1.device.type == device.type
    assert converted.s.length.device.type == device.type
    assert converted.s.k2.device.type == device.type


@pytest.mark.filterwarnings(
    r"ignore:.*( d | g | dg ).*:cheetah.utils.NotUnderstoodPropertyWarning"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_passing(dtype: torch.dtype):
    """Test that the dtype is passed correctly."""
    file_path = "tests/resources/bmad_tutorial_lattice.bmad"

    # Convert the lattice while passing the dtype
    converted = cheetah.Segment.from_bmad(file_path, dtype=dtype)

    # Check that the properties of the loaded elements are of the correct dtype
    assert converted.d.length.dtype == dtype
    assert converted.b.length.dtype == dtype
    assert converted.b.dipole_e1.dtype == dtype
    assert converted.q.length.dtype == dtype
    assert converted.q.k1.dtype == dtype
    assert converted.s.length.dtype == dtype
    assert converted.s.k2.dtype == dtype


@pytest.mark.filterwarnings(
    r"ignore:.*( d | g | dg ).*:cheetah.utils.NotUnderstoodPropertyWarning"
)
@pytest.mark.parametrize(
    "default_torch_dtype", [torch.float32, torch.float64], indirect=True
)
def test_default_dtype(default_torch_dtype):
    """Test that the default dtype is used if no explicit type is passed."""
    file_path = "tests/resources/bmad_tutorial_lattice.bmad"

    # Convert the lattice while passing the dtype
    converted = cheetah.Segment.from_bmad(file_path)

    # Check that the properties of the loaded elements are of the correct dtype
    assert converted.d.length.dtype == default_torch_dtype
    assert converted.b.length.dtype == default_torch_dtype
    assert converted.b.dipole_e1.dtype == default_torch_dtype
    assert converted.q.length.dtype == default_torch_dtype
    assert converted.q.k1.dtype == default_torch_dtype
    assert converted.s.length.dtype == default_torch_dtype
    assert converted.s.k2.dtype == default_torch_dtype
