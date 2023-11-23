import numpy as np
import ocelot
import pytest

from cheetah import ParameterBeam, ParticleBeam, Screen, Segment

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
    ˚"""
    segment = Segment.from_ocelot(ares.cell)
    screen = getattr(segment, name)
    assert isinstance(screen, Screen)


def test_ocelot_to_parameterbeam():
    parray = ocelot.astraBeam2particleArray("tests/resources/ACHIP_EA1_2021.1351.001")
    beam = ParameterBeam.from_ocelot(parray)

    assert np.allclose(beam.mu_x.cpu().numpy(), np.mean(parray.x()))
    assert np.allclose(beam.mu_xp.cpu().numpy(), np.mean(parray.px()))
    assert np.allclose(beam.mu_y.cpu().numpy(), np.mean(parray.y()))
    assert np.allclose(beam.mu_yp.cpu().numpy(), np.mean(parray.py()))
    assert np.allclose(beam.sigma_x.cpu().numpy(), np.std(parray.x()))
    assert np.allclose(beam.sigma_xp.cpu().numpy(), np.std(parray.px()))
    assert np.allclose(beam.sigma_y.cpu().numpy(), np.std(parray.y()))
    assert np.allclose(beam.sigma_yp.cpu().numpy(), np.std(parray.py()))
    assert np.allclose(beam.sigma_s.cpu().numpy(), np.std(parray.tau()))
    assert np.allclose(beam.sigma_p.cpu().numpy(), np.std(parray.p()))
    assert np.allclose(beam.energy.cpu().numpy(), parray.E * 1e9)
    assert np.allclose(beam.total_charge.cpu().numpy(), parray.total_charge)


def test_ocelot_to_particlebeam():
    parray = ocelot.astraBeam2particleArray("tests/resources/ACHIP_EA1_2021.1351.001")
    beam = ParticleBeam.from_ocelot(parray)

    assert np.allclose(beam.particles[:, 0].cpu().numpy(), parray.x())
    assert np.allclose(beam.particles[:, 1].cpu().numpy(), parray.px())
    assert np.allclose(beam.particles[:, 2].cpu().numpy(), parray.y())
    assert np.allclose(beam.particles[:, 3].cpu().numpy(), parray.py())
    assert np.allclose(beam.particles[:, 4].cpu().numpy(), parray.tau())
    assert np.allclose(beam.particles[:, 5].cpu().numpy(), parray.p())
    assert np.allclose(beam.energy.cpu().numpy(), parray.E * 1e9)
    assert np.allclose(beam.particle_charges.cpu().numpy(), parray.q_array)
