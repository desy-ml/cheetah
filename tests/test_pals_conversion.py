import math

import pytest
import torch

import cheetah
from cheetah.converters import pals as cheetah_pals

pals = pytest.importorskip("pals")

UNDER_CONSTRUCTION_WARNING = (
    "ignore:The .* element is marked as 'Under Construction' in the PALS "
    "standard:UserWarning"
)


def assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Test that values and dypes of two tensors are equal."""
    assert actual.dtype == expected.dtype
    assert torch.allclose(actual, expected)


def assert_roundtrip_equal(
    original: cheetah.Segment, converted: cheetah.Segment
) -> None:
    assert converted.name == original.name
    assert len(converted.elements) == len(original.elements)

    for actual, expected in zip(converted.elements, original.elements):
        assert actual.name == expected.name
        assert actual.__class__ is expected.__class__

        if isinstance(expected, cheetah.Drift):
            assert_tensor_equal(actual.length, expected.length)
        elif isinstance(expected, cheetah.Quadrupole):
            assert_tensor_equal(actual.length, expected.length)
            assert_tensor_equal(actual.k1, expected.k1)
            assert_tensor_equal(actual.tilt, expected.tilt)
        elif isinstance(expected, cheetah.Dipole):
            assert_tensor_equal(actual.length, expected.length)
            assert_tensor_equal(actual.angle, expected.angle)
            assert_tensor_equal(actual.k1, expected.k1)
            assert_tensor_equal(actual.dipole_e1, expected.dipole_e1)
            assert_tensor_equal(actual.dipole_e2, expected.dipole_e2)
            assert_tensor_equal(actual.tilt, expected.tilt)
            assert_tensor_equal(actual.fringe_integral, expected.fringe_integral)
            assert_tensor_equal(
                actual.fringe_integral_exit, expected.fringe_integral_exit
            )
        elif isinstance(expected, cheetah.Aperture):
            assert_tensor_equal(actual.x_max, expected.x_max)
            assert_tensor_equal(actual.y_max, expected.y_max)
            assert actual.shape == expected.shape
            assert actual.is_active == expected.is_active
        elif isinstance(expected, cheetah.Cavity):
            assert_tensor_equal(actual.length, expected.length)
            assert_tensor_equal(actual.voltage, expected.voltage)
            assert_tensor_equal(actual.phase, expected.phase)
            assert_tensor_equal(actual.frequency, expected.frequency)
            assert actual.cavity_type == expected.cavity_type
        elif isinstance(expected, cheetah.HorizontalCorrector):
            assert_tensor_equal(actual.length, expected.length)
            assert_tensor_equal(actual.angle, expected.angle)
        elif isinstance(expected, cheetah.VerticalCorrector):
            assert_tensor_equal(actual.length, expected.length)
            assert_tensor_equal(actual.angle, expected.angle)
        elif isinstance(expected, cheetah.CombinedCorrector):
            assert_tensor_equal(actual.length, expected.length)
            assert_tensor_equal(actual.horizontal_angle, expected.horizontal_angle)
            assert_tensor_equal(actual.vertical_angle, expected.vertical_angle)
        elif isinstance(expected, cheetah.Solenoid):
            assert_tensor_equal(actual.length, expected.length)
            assert_tensor_equal(actual.k, expected.k)
            assert_tensor_equal(actual.misalignment, expected.misalignment)


def cheetah_test_segment(dtype: torch.dtype = torch.float32) -> cheetah.Segment:
    return cheetah.Segment(
        [
            cheetah.Drift(length=torch.tensor(0.25, dtype=dtype), name="d1"),
            cheetah.Quadrupole(
                length=torch.tensor(1.0, dtype=dtype),
                k1=torch.tensor(2.0, dtype=dtype),
                misalignment=torch.tensor([0.01, -0.02], dtype=dtype),
                tilt=torch.tensor(0.1, dtype=dtype),
                name="q1",
            ),
            cheetah.Dipole(
                length=torch.tensor(0.5, dtype=dtype),
                angle=torch.tensor(0.05, dtype=dtype),
                k1=torch.tensor(0.3, dtype=dtype),
                dipole_e1=torch.tensor(0.01, dtype=dtype),
                dipole_e2=torch.tensor(0.02, dtype=dtype),
                tilt=torch.tensor(0.03, dtype=dtype),
                fringe_integral=torch.tensor(0.4, dtype=dtype),
                fringe_integral_exit=torch.tensor(0.5, dtype=dtype),
                name="b1",
            ),
            cheetah.Aperture(
                x_max=torch.tensor(0.01, dtype=dtype),
                y_max=torch.tensor(0.02, dtype=dtype),
                shape="elliptical",
                name="ap1",
            ),
            cheetah.Cavity(
                length=torch.tensor(0.3, dtype=dtype),
                voltage=torch.tensor(1.0e6, dtype=dtype),
                phase=torch.tensor(30.0, dtype=dtype),
                frequency=torch.tensor(1.3e9, dtype=dtype),
                name="cav1",
            ),
            cheetah.HorizontalCorrector(
                length=torch.tensor(0.1, dtype=dtype),
                angle=torch.tensor(1.0e-3, dtype=dtype),
                name="hcor1",
            ),
            cheetah.VerticalCorrector(
                length=torch.tensor(0.1, dtype=dtype),
                angle=torch.tensor(-2.0e-3, dtype=dtype),
                name="vcor1",
            ),
            cheetah.CombinedCorrector(
                length=torch.tensor(0.1, dtype=dtype),
                horizontal_angle=torch.tensor(3.0e-3, dtype=dtype),
                vertical_angle=torch.tensor(-4.0e-3, dtype=dtype),
                name="ccor1",
            ),
            cheetah.Solenoid(
                length=torch.tensor(0.2, dtype=dtype),
                k=torch.tensor(0.5, dtype=dtype),
                misalignment=torch.tensor([-0.03, 0.04], dtype=dtype),
                name="sol1",
            ),
            cheetah.Marker(name="m1"),
        ],
        name="cell",
    )


def test_convert_lattice_to_pals_fodo_shape():
    segment = cheetah.Segment(
        [
            cheetah.Drift(length=torch.tensor(0.25), name="d1"),
            cheetah.Quadrupole(
                length=torch.tensor(1.0), k1=torch.tensor(1.2), name="qf"
            ),
            cheetah.Drift(length=torch.tensor(0.5), name="d2"),
            cheetah.Quadrupole(
                length=torch.tensor(1.0), k1=torch.tensor(-1.2), name="qd"
            ),
            cheetah.Drift(length=torch.tensor(0.25), name="d3"),
        ],
        name="fodo",
    )

    lattice = cheetah_pals.convert_lattice_to_pals(segment, name="fodo_lattice")

    assert isinstance(lattice, pals.Lattice)
    assert lattice.name == "fodo_lattice"
    assert len(lattice.branches) == 1
    assert [element.kind for element in lattice.branches[0].line] == [
        "Drift",
        "Quadrupole",
        "Drift",
        "Quadrupole",
        "Drift",
    ]
    assert lattice.branches[0].line[1].MagneticMultipoleP.Kn1 == pytest.approx(1.2)


def test_pals_example_fodo_roundtrip():
    drift1 = pals.Drift(name="drift1", length=0.25)
    quad1 = pals.Quadrupole(
        name="quad1",
        length=1.0,
        MagneticMultipoleP=pals.MagneticMultipoleParameters(Bn1=1.0),
    )
    drift2 = pals.Drift(name="drift2", length=0.5)
    quad2 = pals.Quadrupole(
        name="quad2",
        length=1.0,
        MagneticMultipoleP=pals.MagneticMultipoleParameters(Bn1=-1.0),
    )
    drift3 = pals.Drift(name="drift3", length=0.5)
    line = pals.BeamLine(name="fodo_cell", line=[drift1, quad1, drift2, quad2, drift3])
    lattice = pals.Lattice(name="fodo_lattice", branches=[line])

    segment = cheetah_pals.convert_lattice_from_pals(lattice)
    roundtrip = cheetah_pals.convert_lattice_to_pals(segment, name="fodo_lattice")

    assert roundtrip == lattice


@pytest.mark.filterwarnings(UNDER_CONSTRUCTION_WARNING)
def test_elements_roundtrip():
    original = cheetah_test_segment()

    converted = cheetah_pals.convert_lattice_from_pals(
        cheetah_pals.convert_lattice_to_pals(original, name="pals_lattice")
    )

    assert_roundtrip_equal(original, converted)


@pytest.mark.filterwarnings(UNDER_CONSTRUCTION_WARNING)
def test_extended_elements_to_pals_shape():
    lattice = cheetah_pals.convert_lattice_to_pals(
        cheetah_test_segment(), name="pals_lattice"
    )
    by_name = {element.name: element for element in lattice.branches[0].line}

    assert by_name["ap1"].kind == "Marker"
    assert by_name["ap1"].ApertureP.shape == "ELLIPTICAL"
    assert by_name["ap1"].ApertureP.x_limits == pytest.approx([-0.01, 0.01])
    assert by_name["ap1"].ApertureP.y_limits == pytest.approx([-0.02, 0.02])

    assert by_name["cav1"].kind == "RFCavity"
    assert by_name["cav1"].RFP.voltage == pytest.approx(1.0e6)
    assert by_name["cav1"].RFP.frequency == pytest.approx(1.3e9)
    assert by_name["cav1"].RFP.phase == pytest.approx(math.pi / 6.0)

    assert by_name["q1"].BodyShiftP.x_offset == pytest.approx(0.01)
    assert by_name["q1"].BodyShiftP.y_offset == pytest.approx(-0.02)

    assert by_name["hcor1"].kind == "Kicker"
    assert by_name["hcor1"].MagneticMultipoleP.Kn0 == pytest.approx(1.0e-3)
    assert by_name["vcor1"].MagneticMultipoleP.Ks0 == pytest.approx(-2.0e-3)
    assert by_name["ccor1"].MagneticMultipoleP.Kn0 == pytest.approx(3.0e-3)
    assert by_name["ccor1"].MagneticMultipoleP.Ks0 == pytest.approx(-4.0e-3)

    assert by_name["sol1"].kind == "Solenoid"
    assert by_name["sol1"].BodyShiftP.x_offset == pytest.approx(-0.03)
    assert by_name["sol1"].BodyShiftP.y_offset == pytest.approx(0.04)
    assert by_name["sol1"].SolenoidP.Ksol == pytest.approx(0.5)


@pytest.mark.filterwarnings(UNDER_CONSTRUCTION_WARNING)
def test_zero_correctors_roundtrip_by_multipole_key():
    segment = cheetah.Segment(
        [
            cheetah.HorizontalCorrector(
                length=torch.tensor(0.1), angle=torch.tensor(0.0), name="hcor"
            ),
            cheetah.VerticalCorrector(
                length=torch.tensor(0.1), angle=torch.tensor(0.0), name="vcor"
            ),
            cheetah.CombinedCorrector(
                length=torch.tensor(0.1),
                horizontal_angle=torch.tensor(0.0),
                vertical_angle=torch.tensor(0.0),
                name="ccor",
            ),
        ],
        name="zero_correctors",
    )

    converted = cheetah_pals.convert_lattice_from_pals(
        cheetah_pals.convert_lattice_to_pals(segment)
    )

    assert isinstance(converted.hcor, cheetah.HorizontalCorrector)
    assert isinstance(converted.vcor, cheetah.VerticalCorrector)
    assert isinstance(converted.ccor, cheetah.CombinedCorrector)


def test_convert_lattice_from_pals_dtype_and_multipole_family_roundtrip():
    beamline = pals.BeamLine(
        name="line",
        line=[
            pals.Quadrupole(
                name="q1",
                length=1.0,
                MagneticMultipoleP=pals.MagneticMultipoleParameters(Bn1=3.0, tilt1=0.2),
            )
        ],
    )

    segment = cheetah_pals.convert_lattice_from_pals(beamline, dtype=torch.float64)

    assert segment.q1.length.dtype == torch.float64
    assert segment.q1.k1.dtype == torch.float64
    assert torch.isclose(segment.q1.k1, torch.tensor(3.0, dtype=torch.float64))
    assert torch.isclose(segment.q1.tilt, torch.tensor(0.2, dtype=torch.float64))

    lattice = cheetah_pals.convert_lattice_to_pals(segment)
    multipole = lattice.branches[0].line[0].MagneticMultipoleP.model_dump()
    assert multipole["Bn1"] == pytest.approx(3.0)
    assert "Kn1" not in multipole


@pytest.mark.filterwarnings(UNDER_CONSTRUCTION_WARNING)
def test_file_io_roundtrip(tmp_path):
    original = cheetah_test_segment()

    for suffix in ("yaml", "json"):
        filename = tmp_path / f"lattice.pals.{suffix}"
        cheetah_pals.save_lattice_to_pals(original, filename, name="saved_lattice")

        converted = cheetah_pals.load_lattice_from_pals(filename)

        assert_roundtrip_equal(original, converted)


@pytest.mark.filterwarnings(UNDER_CONSTRUCTION_WARNING)
def test_segment_pals_methods(tmp_path):
    original = cheetah_test_segment()

    lattice = original.to_pals(name="segment_methods")
    converted = cheetah.Segment.from_pals(lattice)

    assert isinstance(lattice, pals.Lattice)
    assert_roundtrip_equal(original, converted)

    filename = tmp_path / "segment_methods.pals.json"
    original.to_pals_file(filename, name="segment_methods_file")
    loaded = cheetah.Segment.from_pals_file(filename)

    assert_roundtrip_equal(original, loaded)


def test_pals_extras_roundtrip():
    beamline = pals.BeamLine(
        name="line",
        line=[
            pals.Quadrupole(
                name="q1",
                length=1.0,
                BodyShiftP=pals.BodyShiftParameters(x_offset=0.5),
                MagneticMultipoleP=pals.MagneticMultipoleParameters(Kn1=2.0, Kn2=5.0),
            )
        ],
    )

    segment = cheetah_pals.convert_lattice_from_pals(beamline)

    assert_tensor_equal(segment.q1.misalignment, torch.tensor([0.5, 0.0]))
    assert "x_offset" not in segment.q1.pals_extras.get("BodyShiftP", {})
    assert "y_offset" not in segment.q1.pals_extras.get("BodyShiftP", {})
    assert segment.q1.pals_extras["MagneticMultipoleP"]["Kn2"] == pytest.approx(5.0)

    lattice = cheetah_pals.convert_lattice_to_pals(segment)
    roundtrip_quad = lattice.branches[0].line[0]

    assert roundtrip_quad.BodyShiftP.x_offset == pytest.approx(0.5)
    assert roundtrip_quad.BodyShiftP.y_offset == pytest.approx(0.0)
    assert roundtrip_quad.MagneticMultipoleP.Kn2 == pytest.approx(5.0)


def test_unknown_pals_element_becomes_marker():
    with pytest.warns(UserWarning, match="Under Construction"):
        unknown = pals.NullEle(name="unknown")
    beamline = pals.BeamLine(name="line", line=[unknown])

    with pytest.warns(cheetah.UnknownElementWarning):
        segment = cheetah_pals.convert_lattice_from_pals(beamline)

    assert isinstance(segment.unknown, cheetah.Marker)
    assert segment.unknown.pals_extras["kind"] == "NullEle"


def test_unknown_cheetah_element_rejected():
    segment = cheetah.Segment(
        [cheetah.Sextupole(length=torch.tensor(1.0), k2=torch.tensor(2.0), name="s1")],
        name="unsupported",
    )

    with pytest.raises(NotImplementedError, match="not supported"):
        cheetah_pals.convert_lattice_to_pals(segment)


def test_batched_tensor_rejected():
    segment = cheetah.Segment(
        [cheetah.Drift(length=torch.tensor([1.0, 2.0]), name="d1")],
        name="batched",
    )

    with pytest.raises(ValueError, match="scalar"):
        cheetah_pals.convert_lattice_to_pals(segment)
