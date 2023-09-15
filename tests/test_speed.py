import time

import torch

import cheetah

from .resources import ARESlatticeStage3v1_9 as ares


# TODO: Test that Cheeath tracks faster than Ocelot
def test_tracking_speed():
    """Really only tests that Cheetah isn't super slow."""
    cell = cheetah.converters.nocelot.subcell_of_ocelot(
        ares.cell, "AREASOLA1", "AREABSCR1"
    )
    segment = cheetah.Segment.from_ocelot(cell)
    segment.AREABSCR1.is_active = True  # Turn screen on and off

    particles = cheetah.ParticleBeam.from_parameters(
        num_particles=torch.tensor(int(1e5)),
        sigma_x=torch.tensor(175e-6),
        sigma_y=torch.tensor(175e-6),
    )

    t1 = time.time()

    _ = segment.track(particles)
    _ = segment.AREABSCR1.reading

    t2 = time.time()

    assert (t2 - t1) < 0.1
