import numpy as np
import pytest
import torch

import cheetah
from cheetah.utils import (
    DirtyNameWarning,
    NoBeamPropertiesInLatticeWarning,
    NotUnderstoodPropertyWarning,
    is_mps_available_and_functional,
)
from cheetah.utils.warnings import PhysicsWarning


def test_fodo_import():
    """Test importing a FODO lattice defined in the Elegant file format."""
    file_path = "tests/resources/fodo.lte"

    with pytest.warns(
        NoBeamPropertiesInLatticeWarning, match=("c.*charge")
    ), pytest.warns(DirtyNameWarning, match="long-name-quad"), pytest.warns(
        DirtyNameWarning, match="a:q3"
    ), pytest.warns(
        NotUnderstoodPropertyWarning, match="nonsense"
    ):
        converted = cheetah.Segment.from_elegant(file_path, "fodo")

    with pytest.warns(DirtyNameWarning, match="long-name-quad"), pytest.warns(
        DirtyNameWarning, match="a:q3"
    ):
        correct_lattice = cheetah.Segment(
            [
                cheetah.Marker(name="c"),
                cheetah.Quadrupole(
                    name="q1", length=torch.tensor(0.1), k1=torch.tensor(1.5)
                ),
                cheetah.Drift(name="d1", length=torch.tensor(1.0)),
                cheetah.Marker(name="m1"),
                cheetah.Dipole(
                    name="b1",
                    length=torch.tensor(0.3),
                    angle=torch.tensor(0.25),
                    dipole_e1=torch.tensor(0.25),
                    gap=torch.tensor(0.04),
                    fringe_integral=torch.tensor(0.5),
                ),
                cheetah.Drift(name="d1", length=torch.tensor(1.0)),
                cheetah.Quadrupole(
                    name="q2", length=torch.tensor(0.2), k1=torch.tensor(-3.0)
                ),
                cheetah.Drift(name="d2", length=torch.tensor(-2.0)),
                cheetah.Sextupole(
                    name="s1", length=torch.tensor(0.2), k2=torch.tensor(-87.1)
                ),
                cheetah.Dipole(
                    name="csrbend",
                    length=torch.tensor(0.200981),
                    angle=torch.tensor(0.113612175128842),
                    dipole_e2=torch.tensor(0.113612175128842),
                    k1=torch.tensor(0.0),
                ),
                cheetah.Quadrupole(
                    name="long-name-quad",
                    length=torch.tensor(0.3),
                    k1=torch.tensor(2.0),
                ),
                cheetah.Drift(
                    name="d3", length=torch.tensor(0.0)
                ),  # No length `l` provided
                cheetah.Quadrupole(
                    name="a:q3", length=torch.tensor(0.1), k1=torch.tensor(1.5)
                ),  # Element with a colon in the name
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
    assert torch.isclose(
        getattr(converted, "long-name-quad").length,
        getattr(correct_lattice, "long-name-quad").length,
    )
    assert torch.isclose(
        getattr(converted, "long-name-quad").k1,
        getattr(correct_lattice, "long-name-quad").k1,
    )

    for i in range(2):
        assert torch.isclose(converted.d1[i].length, correct_lattice.d1[i].length)
    assert torch.isclose(converted.d2.length, correct_lattice.d2.length)

    assert torch.isclose(converted.b1.length, correct_lattice.b1.length)
    assert torch.isclose(converted.b1.angle, correct_lattice.b1.angle)
    assert torch.isclose(converted.b1.dipole_e1, correct_lattice.b1.dipole_e1)
    assert torch.isclose(converted.b1.gap, correct_lattice.b1.gap)
    assert torch.isclose(
        converted.b1.fringe_integral, correct_lattice.b1.fringe_integral
    )
    assert torch.isclose(converted.csrbend.length, correct_lattice.csrbend.length)
    assert torch.isclose(converted.csrbend.angle, correct_lattice.csrbend.angle)
    assert torch.isclose(converted.csrbend.dipole_e2, correct_lattice.csrbend.dipole_e2)
    assert torch.isclose(converted.csrbend.k1, correct_lattice.csrbend.k1)

    assert torch.isclose(converted.s1.length, correct_lattice.s1.length)
    assert torch.isclose(converted.s1.k2, correct_lattice.s1.k2)


@pytest.mark.filterwarnings("ignore:.*long-name-quad.*:cheetah.utils.DirtyNameWarning")
@pytest.mark.filterwarnings("ignore:.*a.*q3.*:cheetah.utils.DirtyNameWarning")
@pytest.mark.filterwarnings(
    "ignore:.*c.*charge.*:cheetah.utils.NoBeamPropertiesInLatticeWarning"
)
@pytest.mark.filterwarnings(
    "ignore:.*nonsense.*:cheetah.utils.NotUnderstoodPropertyWarning"
)
def test_reverse_beamline_import():
    """Test importing a reversed beamline."""
    file_path = "tests/resources/fodo.lte"

    converted_forward = cheetah.Segment.from_elegant(file_path, "fodo")
    correct_lattice = converted_forward.reversed()
    correct_lattice.name = "reversed_fodo"

    converted_reversed = cheetah.Segment.from_elegant(
        file_path, "reversed_fodo"
    ).flattened()

    assert converted_reversed.name == correct_lattice.name
    assert [element.name for element in converted_reversed.elements] == [
        element.name for element in correct_lattice.elements
    ]


def test_cavity_import():
    """Test importing an accelerating cavity defined in the Elegant file format."""
    file_path = "tests/resources/cavity.lte"
    with pytest.warns(
        NotUnderstoodPropertyWarning, match="(end[12]_focus|body_focus_model|change_p0)"
    ), pytest.warns(PhysicsWarning, match="srs"):
        converted = cheetah.Segment.from_elegant(file_path, "cavity")

    assert np.isclose(converted.c1.length, 0.7)
    assert np.isclose(converted.c1.frequency, 1.2e9)
    assert np.isclose(converted.c1.voltage, 16.175e6)

    # Cheetah and Elegant use different phase conventions shifted by 90 deg
    assert np.isclose(converted.c1.phase, 0.0)


@pytest.mark.filterwarnings(
    "ignore:"
    ".*(end[12]_focus|body_focus_model|change_p0).*:"
    "cheetah.utils.NotUnderstoodPropertyWarning"
)
@pytest.mark.filterwarnings("ignore:.*srs.*:cheetah.utils.PhysicsWarning")
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
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert torch.allclose(converted.c1e.predefined_transfer_map, correct_transfer_map)


@pytest.mark.filterwarnings("ignore:.*long-name-quad.*:cheetah.utils.DirtyNameWarning")
@pytest.mark.filterwarnings("ignore:.*a.*q3.*:cheetah.utils.DirtyNameWarning")
@pytest.mark.filterwarnings(
    "ignore:.*c.*charge.*:cheetah.utils.NoBeamPropertiesInLatticeWarning"
)
@pytest.mark.filterwarnings(
    "ignore:.*nonsense.*:cheetah.utils.NotUnderstoodPropertyWarning"
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
def test_lattice_device(device: torch.device):
    """Test that the device is passed correctly."""
    file_path = "tests/resources/fodo.lte"

    # Convert the lattice while passing the device
    converted = cheetah.Segment.from_elegant(file_path, "fodo", device=device)

    # Check that the properties of the loaded elements are on the correct device
    assert converted.q1.length.device.type == device.type
    assert converted.q1.k1.device.type == device.type
    assert converted.q2.length.device.type == device.type
    assert converted.q2.k1.device.type == device.type
    assert getattr(converted, "long-name-quad").length.device.type == device.type
    assert getattr(converted, "long-name-quad").k1.device.type == device.type

    assert [d.length.device.type for d in converted.d1] == [device.type, device.type]
    assert converted.d2.length.device.type == device.type

    assert converted.b1.length.device.type == device.type
    assert converted.b1.angle.device.type == device.type
    assert converted.b1.dipole_e1.device.type == device.type
    assert converted.b1.gap.device.type == device.type
    assert converted.csrbend.length.device.type == device.type
    assert converted.csrbend.angle.device.type == device.type
    assert converted.csrbend.dipole_e2.device.type == device.type
    assert converted.csrbend.k1.device.type == device.type

    assert converted.s1.length.device.type == device.type
    assert converted.s1.k2.device.type == device.type


@pytest.mark.filterwarnings("ignore:.*long-name-quad.*:cheetah.utils.DirtyNameWarning")
@pytest.mark.filterwarnings("ignore:.*a.*q3.*:cheetah.utils.DirtyNameWarning")
@pytest.mark.filterwarnings(
    "ignore:.*c.*charge.*:cheetah.utils.NoBeamPropertiesInLatticeWarning"
)
@pytest.mark.filterwarnings(
    "ignore:.*nonsense.*:cheetah.utils.NotUnderstoodPropertyWarning"
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
def test_lattice_dtype(dtype: torch.dtype):
    """Test that the dtype is passed correctly."""
    file_path = "tests/resources/fodo.lte"

    # Convert the lattice while passing the device
    converted = cheetah.Segment.from_elegant(file_path, "fodo", dtype=dtype)

    # Check that the properties of the loaded elements are of the correct dtype
    assert converted.q1.length.dtype == dtype
    assert converted.q1.k1.dtype == dtype
    assert converted.q2.length.dtype == dtype
    assert converted.q2.k1.dtype == dtype
    assert getattr(converted, "long-name-quad").length.dtype == dtype
    assert getattr(converted, "long-name-quad").k1.dtype == dtype

    assert [d.length.dtype for d in converted.d1] == [dtype, dtype]
    assert converted.d2.length.dtype == dtype

    assert converted.b1.length.dtype == dtype
    assert converted.b1.anlge.dtype == dtype
    assert converted.b1.dipole_e1.dtype == dtype
    assert converted.b1.gap.dtype == dtype
    assert converted.csrbend.length.dtype == dtype
    assert converted.csrbend.angle.dtype == dtype
    assert converted.csrbend.dipole_e2.dtype == dtype
    assert converted.csrbend.k1.dtype == dtype

    assert converted.s1.length.dtype == dtype
    assert converted.s1.k2.dtype == dtype


@pytest.mark.filterwarnings("ignore:.*long-name-quad.*:cheetah.utils.DirtyNameWarning")
@pytest.mark.filterwarnings("ignore:.*a.*q3.*:cheetah.utils.DirtyNameWarning")
@pytest.mark.filterwarnings(
    "ignore:.*c.*charge.*:cheetah.utils.NoBeamPropertiesInLatticeWarning"
)
@pytest.mark.filterwarnings(
    "ignore:.*nonsense.*:cheetah.utils.NotUnderstoodPropertyWarning"
)
@pytest.mark.parametrize(
    "default_torch_dtype",
    [torch.float32, torch.float64],
    indirect=True,
    ids=["float32", "float64"],
)
def test_lattice_default_dtype(default_torch_dtype):
    """Test that the default dtype is used if no explicit type is passed."""
    file_path = "tests/resources/fodo.lte"

    # Convert the lattice while passing the device
    converted = cheetah.Segment.from_elegant(file_path, "fodo")

    # Check that the properties of the loaded elements are of the correct dtype
    assert converted.q1.length.dtype == default_torch_dtype
    assert converted.q1.k1.dtype == default_torch_dtype
    assert converted.q2.length.dtype == default_torch_dtype
    assert converted.q2.k1.dtype == default_torch_dtype
    assert getattr(converted, "long-name-quad").length.dtype == default_torch_dtype
    assert getattr(converted, "long-name-quad").k1.dtype == default_torch_dtype

    assert [d.length.dtype for d in converted.d1] == [default_torch_dtype] * 2
    assert converted.d2.length.dtype == default_torch_dtype

    assert converted.b1.length.dtype == default_torch_dtype
    assert converted.b1.angle.dtype == default_torch_dtype
    assert converted.b1.dipole_e1.dtype == default_torch_dtype
    assert converted.b1.gap.dtype == default_torch_dtype
    assert converted.csrbend.length.dtype == default_torch_dtype
    assert converted.csrbend.angle.dtype == default_torch_dtype
    assert converted.csrbend.dipole_e2.dtype == default_torch_dtype
    assert converted.csrbend.k1.dtype == default_torch_dtype

    assert converted.s1.length.dtype == default_torch_dtype
    assert converted.s1.k2.dtype == default_torch_dtype


def test_particle_beam_import():
    """Test that Elegant SDDS beams are correctly loaded into particle beams."""
    beam = cheetah.ParticleBeam.from_elegant(
        "tests/resources/ACHIP_EA1_2021.1351.001.sdds", dtype=torch.float64
    )

    # These values are the same as those in `test_astra_to_particle_beam` since the
    # Elegant SDDS file was generated from the Astra beam file used in that test.
    assert beam.num_particles == 100_000
    assert np.allclose(beam.mu_x.cpu().numpy(), 8.24126345833065e-07)
    assert np.allclose(beam.mu_px.cpu().numpy(), 5.988477624896404e-08)
    assert np.allclose(beam.mu_y.cpu().numpy(), -1.7276204289373709e-06)
    assert np.allclose(beam.mu_py.cpu().numpy(), -1.1746412553748087e-07)
    assert np.allclose(beam.sigma_x.cpu().numpy(), 0.00017489789752289653)
    assert np.allclose(beam.sigma_px.cpu().numpy(), 3.679402198031312e-06)
    assert np.allclose(beam.sigma_y.cpu().numpy(), 0.00017519544053357095)
    assert np.allclose(beam.sigma_py.cpu().numpy(), 3.6941000871593133e-06)
    assert np.allclose(beam.sigma_tau.cpu().numpy(), 8.011552381503861e-06)
    assert np.allclose(beam.sigma_p.cpu().numpy(), 0.0022804534528404474)
    assert np.allclose(beam.energy.cpu().numpy(), 107315902.44394557)
    assert np.allclose(beam.total_charge.cpu().numpy(), 5.000000000010205e-13)


@pytest.mark.parametrize(
    "requested_dtype",
    [None, torch.float32, torch.float64],
    ids=["default", "float32", "float64"],
)
def test_particle_beam_dtype(requested_dtype: torch.dtype):
    """
    Test that Elegant SDDS beams are correctly converted to Cheetah `ParticleBeams` with
    different torch dtypes.
    """
    beam = cheetah.ParticleBeam.from_elegant(
        "tests/resources/ACHIP_EA1_2021.1351.001.sdds", dtype=requested_dtype
    )

    correct_dtype = (
        requested_dtype if requested_dtype is not None else torch.get_default_dtype()
    )

    assert beam.particles.dtype == correct_dtype
    assert beam.energy.dtype == correct_dtype
    assert beam.particle_charges.dtype == correct_dtype
    assert beam.survival_probabilities.dtype == correct_dtype
    assert beam.s.dtype == correct_dtype
    assert beam.species.num_elementary_charges.dtype == correct_dtype
    assert beam.species.mass_eV.dtype == correct_dtype
