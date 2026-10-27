import pytest
import torch

import cheetah
import cheetah.converters.bmad as bmad_converter
from cheetah.utils import (
    NotUnderstoodPropertyWarning,
    PhysicsWarning,
    is_mps_available_and_functional,
)


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
    ids=["cpu", "cuda", "mps"],
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
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
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
    "default_torch_dtype",
    [torch.float32, torch.float64],
    indirect=True,
    ids=["float32", "float64"],
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


def test_cu_hxr_lcls_fixture_conversion():
    """Test converting the reduced split CU_HXR fixture into Cheetah."""
    file_path = "tests/resources/lcls/cu_hxr.lat.bmad"

    converted = cheetah.Segment.from_bmad(file_path, dtype=torch.float64)
    flattened = converted.flattened()

    assert isinstance(converted, cheetah.Segment)
    assert converted.name == "cu_hxr"
    assert converted.gunl0a.element_names == [
        "beggunb",
        "l0a",
        "l0b",
        "qa01_split_0",
        "bpm4",
        "qa01_split_1",
        "d_mid",
        "qa02_split_0",
        "bpm5",
        "otr3",
        "qa02_split_1",
        "qe01_split_0",
        "otr2",
        "qe01_split_1",
    ]
    assert isinstance(converted.gunl0a.qa01_split_0, cheetah.Quadrupole)
    assert isinstance(converted.gunl0a.bpm4, cheetah.Marker)

    assert isinstance(flattened.bx11, cheetah.Dipole)
    assert flattened.qa01_split_0.k1.item() == pytest.approx(0.384840836193)
    assert flattened.qa01_split_0.metadata["alias"] == "quad:in20:121"

    assert flattened.l0a.phase.item() == pytest.approx(-3600.0)
    assert flattened.l0b.phase.item() == pytest.approx(-3600.0)

    assert isinstance(flattened.tcxdg0, cheetah.TransverseDeflectingCavity)
    assert flattened.tcxdg0.metadata["type"] == "stcav_x"
    assert flattened.tcxdg0.frequency.item() == pytest.approx(2.856e9)
    assert flattened.tcxdg0.length.item() == pytest.approx(0.254)


def test_superimpose_non_zero_length_warns():
    """
    Test that a superimposed element with a non-zero length is dropped with a warning,
    because Cheetah can only superimpose zero-length elements.
    """
    context = {
        "qa01": {"element_type": "quadrupole", "l": 0.1, "k1": 1.0},
        "trim": {
            "element_type": "hkicker",
            "l": 0.05,
            "kick": 1e-4,
            "superimpose": "T",
            "ref": "qa01",
        },
        "line1": ["qa01"],
    }

    superimpositions = bmad_converter.collect_superimpositions(context)
    with pytest.warns(PhysicsWarning, match="zero length"):
        converted = bmad_converter.convert_element(
            "line1", context, superimpositions=superimpositions
        )

    assert isinstance(converted, cheetah.Segment)
    assert converted.element_names == ["qa01"]
    assert isinstance(converted.elements[0], cheetah.Quadrupole)


def test_partial_superimpose_non_zero_length_skips_only_non_zero():
    """
    Test that when multiple elements are superimposed on the same base element,
    only the non-zero-length ones are skipped, while zero-length ones are superimposed.
    """
    context = {
        "qa01": {"element_type": "quadrupole", "l": 0.1, "k1": 1.0},
        "bpm1": {"element_type": "marker", "superimpose": "T", "ref": "qa01"},
        "trim": {
            "element_type": "hkicker",
            "l": 0.05,
            "kick": 1e-4,
            "superimpose": "T",
            "ref": "qa01",
        },
        "line1": ["qa01"],
    }

    superimpositions = bmad_converter.collect_superimpositions(context)
    with pytest.warns(PhysicsWarning, match="zero length"):
        converted = bmad_converter.convert_element(
            "line1", context, superimpositions=superimpositions
        )

    assert isinstance(converted, cheetah.Segment)
    assert converted.element_names == ["qa01_split_0", "bpm1", "qa01_split_1"]


def test_superimpose_split_failure_falls_back_to_base(monkeypatch):
    """
    Test that superimpose conversion falls back to the
    base element on split errors.
    """
    file_path = "tests/resources/lcls/cu_hxr.lat.bmad"

    def _raise_split_assertion_error(self, resolution):
        raise AssertionError("forced split failure for test")

    monkeypatch.setattr(cheetah.Quadrupole, "split", _raise_split_assertion_error)

    with pytest.warns(PhysicsWarning, match="Keeping only the base element"):
        converted = cheetah.Segment.from_bmad(file_path, dtype=torch.float64)

    assert isinstance(converted.gunl0a.qe01, cheetah.Quadrupole)
    assert isinstance(converted.gunl0a.qa02, cheetah.Quadrupole)
