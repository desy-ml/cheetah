import numpy as np
import ocelot
import pytest
import torch

import cheetah
from cheetah.utils import DefaultParameterWarning

from .resources import ARESlatticeStage3v1_9 as ares


@pytest.mark.parametrize(
    "name",
    [
        "ARLIBSCX1",
        "ARLIBSCR1",
        "ARLIBSCR2",
        "ARLIBSCR3",
        "AREABSCR1",
        "ARMRBSCR1",
        "ARMRBSCR2",
        "ARMRBSCR3",
        "ARBCBSCE1",
        "ARDLBSCR1",
        "ARDLBSCE1",
        "ARSHBSCE2",
        "ARSHBSCE1",
    ],
)
def test_screen_conversion(name: str):
    """
    Test on the example of the ARES lattice that all screens are correctly converted to
    `cheetah.Screen`.
    """
    with pytest.warns(DefaultParameterWarning):
        segment = cheetah.Segment.from_ocelot(ares.cell)
    screen = getattr(segment, name)
    assert isinstance(screen, cheetah.Screen)


def test_ocelot_to_parameterbeam():
    parray = ocelot.astraBeam2particleArray("tests/resources/ACHIP_EA1_2021.1351.001")
    beam = cheetah.ParameterBeam.from_ocelot(parray)

    assert np.allclose(beam.mu_x.cpu().numpy(), np.mean(parray.x()))
    assert np.allclose(beam.mu_px.cpu().numpy(), np.mean(parray.px()))
    assert np.allclose(beam.mu_y.cpu().numpy(), np.mean(parray.y()))
    assert np.allclose(beam.mu_py.cpu().numpy(), np.mean(parray.py()))
    assert np.allclose(beam.sigma_x.cpu().numpy(), np.std(parray.x()))
    assert np.allclose(beam.sigma_px.cpu().numpy(), np.std(parray.px()))
    assert np.allclose(beam.sigma_y.cpu().numpy(), np.std(parray.y()))
    assert np.allclose(beam.sigma_py.cpu().numpy(), np.std(parray.py()))
    assert np.allclose(beam.sigma_tau.cpu().numpy(), np.std(parray.tau()))
    assert np.allclose(beam.sigma_p.cpu().numpy(), np.std(parray.p()))
    assert np.allclose(beam.energy.cpu().numpy(), parray.E * 1e9)
    assert np.allclose(beam.total_charge.cpu().numpy(), parray.total_charge)


def test_ocelot_to_particlebeam():
    parray = ocelot.astraBeam2particleArray("tests/resources/ACHIP_EA1_2021.1351.001")
    beam = cheetah.ParticleBeam.from_ocelot(parray)

    assert np.allclose(beam.particles[:, 0].cpu().numpy(), parray.x())
    assert np.allclose(beam.particles[:, 1].cpu().numpy(), parray.px())
    assert np.allclose(beam.particles[:, 2].cpu().numpy(), parray.y())
    assert np.allclose(beam.particles[:, 3].cpu().numpy(), parray.py())
    assert np.allclose(beam.particles[:, 4].cpu().numpy(), parray.tau())
    assert np.allclose(beam.particles[:, 5].cpu().numpy(), parray.p())
    assert np.allclose(beam.energy.cpu().numpy(), parray.E * 1e9)
    assert np.allclose(beam.particle_charges.cpu().numpy(), parray.q_array)


@pytest.mark.parametrize("BeamClass", [cheetah.ParameterBeam, cheetah.ParticleBeam])
@pytest.mark.parametrize("desired_dtype", [None, torch.float32, torch.float64])
def test_beam_desired_dtype(BeamClass: cheetah.Beam, desired_dtype: torch.dtype):
    """Test that Ocelot import respects the desired dtype."""
    parray = ocelot.astraBeam2particleArray("tests/resources/ACHIP_EA1_2021.1351.001")
    beam = BeamClass.from_ocelot(parray, dtype=desired_dtype)

    correct_dtype = desired_dtype or torch.get_default_dtype()

    assert beam.mu_x.dtype == correct_dtype
    assert beam.mu_px.dtype == correct_dtype
    assert beam.mu_y.dtype == correct_dtype
    assert beam.mu_py.dtype == correct_dtype
    assert beam.sigma_x.dtype == correct_dtype
    assert beam.sigma_px.dtype == correct_dtype
    assert beam.sigma_y.dtype == correct_dtype
    assert beam.sigma_py.dtype == correct_dtype
    assert beam.sigma_tau.dtype == correct_dtype
    assert beam.sigma_p.dtype == correct_dtype
    assert beam.energy.dtype == correct_dtype
    assert beam.total_charge.dtype == correct_dtype


def test_ocelot_lattice_import():
    """Tests if a lattice is importet correctly (and to the device requested)."""
    cell = [
        ocelot.Drift(l=0.3),
        ocelot.Quadrupole(l=0.2),
        ocelot.Drift(l=1.0),
        ocelot.Sextupole(l=0.4),
    ]
    segment = cheetah.Segment.from_ocelot(cell=cell)

    assert isinstance(segment.elements[0], cheetah.Drift)
    assert isinstance(segment.elements[1], cheetah.Quadrupole)
    assert isinstance(segment.elements[2], cheetah.Drift)
    assert isinstance(segment.elements[3], cheetah.Sextupole)

    assert segment.elements[0].length.device.type == "cpu"
    assert segment.elements[1].length.device.type == "cpu"
    assert segment.elements[1].k1.device.type == "cpu"
    assert segment.elements[1].misalignment.device.type == "cpu"
    assert segment.elements[2].length.device.type == "cpu"
    assert segment.elements[3].length.device.type == "cpu"
    assert segment.elements[3].k2.device.type == "cpu"
