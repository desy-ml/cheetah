import torch

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

    # Run the plotting to see if it raises an exception
    segment.plot_twiss(incoming_beam)


def test_reference_particle_plot():
    """
    Test that the reference particle plot does not raise an exception using the example
    from the `simple.ipynb` example notebook from the documentation.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.BPM(name="BPM1SMATCH"),
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.BPM(name="BPM6SMATCH"),
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.VerticalCorrector(length=torch.tensor(0.3), name="V7SMATCH"),
            cheetah.Drift(length=torch.tensor(0.2)),
            cheetah.HorizontalCorrector(length=torch.tensor(0.3), name="H10SMATCH"),
            cheetah.Drift(length=torch.tensor(7.0)),
            cheetah.HorizontalCorrector(length=torch.tensor(0.3), name="H12SMATCH"),
            cheetah.Drift(length=torch.tensor(0.05)),
            cheetah.BPM(name="BPM13SMATCH"),
        ]
    )

    segment.V7SMATCH.angle = torch.tensor(3.142e-3)

    incoming = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    # Run the plotting to see if it raises an exception
    segment.plot_overview(beam=incoming)
