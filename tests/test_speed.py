import time

import pytest
import torch

import cheetah

from .resources import ARESlatticeStage3v1_9 as ares

# TODO: Test that Cheeath tracks faster than Ocelot


@pytest.mark.filterwarnings("ignore::cheetah.utils.DefaultParameterWarning")
def test_tracking_speed():
    """Really only tests that Cheetah isn't super slow."""
    cell = cheetah.converters.ocelot.subcell_of_ocelot(
        ares.cell, "AREASOLA1", "AREABSCR1"
    )
    segment = cheetah.Segment.from_ocelot(cell)
    segment.AREABSCR1.is_active = True  # Turn screen on and off

    particles = cheetah.ParticleBeam.from_parameters(
        num_particles=int(1e5),
        sigma_x=torch.tensor(175e-6),
        sigma_y=torch.tensor(175e-6),
    )

    # Use a fixed thread count so thread-pool overhead is comparable across runners
    num_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        start = time.perf_counter()
        _ = segment.track(particles)
        _ = segment.AREABSCR1.reading
        elapsed = time.perf_counter() - start
    finally:
        torch.set_num_threads(num_threads)

    assert elapsed < 0.1
