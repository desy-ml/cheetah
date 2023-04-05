import test.ARESlatticeStage3v1_9 as ares
import time

import cheetah


def test_tracking_speed():
    cell = cheetah.utils.subcell_of_ocelot(ares.cell, "AREASOLA1", "AREABSCR1")
    segment = cheetah.Segment.from_ocelot(cell)
    segment.AREABSCR1.is_active = True  # Turn screen on and off

    particles = cheetah.ParticleBeam.from_parameters(
        n=int(1e5), sigma_x=175e-6, sigma_y=175e-6
    )

    t1 = time.time()

    _ = segment(particles)
    _ = segment.AREABSCR1.reading

    t2 = time.time()

    assert (t2 - t1) < 0.1
