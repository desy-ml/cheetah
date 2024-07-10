import cheetah

from .resources import ARESlatticeStage3v1_9 as ares


def test_twiss_plot():
    """
    Test that the Twiss plot does not raise an exception using the ARES EA as an
    example.
    """
    cell = cheetah.converters.ocelot.subcell_of_ocelot(
        ares.cell, "AREASOLA1", "AREABSCR1"
    )
    ares.areamqzm1.k1 = 5.0
    ares.areamqzm2.k1 = -5.0
    ares.areamcvm1.k1 = 1e-3
    ares.areamqzm3.k1 = 5.0
    ares.areamchm1.k1 = -2e-3

    incoming_beam = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    segment = cheetah.Segment.from_ocelot(cell)

    # Do the actual plotting
    segment.plot_twiss(incoming_beam)
