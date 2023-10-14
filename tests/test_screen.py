import numpy as np
import torch

import cheetah

from .resources import ARESlatticeStage3v1_9 as ocelot_lattice


def test_reading_shows_beam_particle():
    """
    Test that a screen has a reading that shows some sign of the beam having hit it.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.Screen(
                resolution=torch.tensor((100, 100)),
                pixel_size=torch.tensor((1e-5, 1e-5)),
                is_active=True,
                name="my_screen",
            ),
        ],
    )
    beam = cheetah.ParticleBeam.from_astra("benchmark/cheetah/ACHIP_EA1_2021.1351.001")

    assert isinstance(segment.my_screen.reading, torch.Tensor)
    assert segment.my_screen.reading.shape == (100, 100)
    assert np.allclose(segment.my_screen.reading, 0.0)

    _ = segment.track(beam)

    assert isinstance(segment.my_screen.reading, torch.Tensor)
    assert segment.my_screen.reading.shape == (100, 100)
    assert torch.all(segment.my_screen.reading >= 0.0)
    assert torch.any(segment.my_screen.reading > 0.0)


def test_reading_shows_beam_parameter():
    """
    Test that a screen has a reading that shows some sign of the beam having hit it.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.Screen(
                resolution=torch.tensor((100, 100)),
                pixel_size=torch.tensor((1e-5, 1e-5)),
                is_active=True,
                name="my_screen",
            ),
        ],
        name="my_segment",
    )
    beam = cheetah.ParameterBeam.from_astra("benchmark/cheetah/ACHIP_EA1_2021.1351.001")

    assert isinstance(segment.my_screen.reading, torch.Tensor)
    assert segment.my_screen.reading.shape == (100, 100)
    assert np.allclose(segment.my_screen.reading, 0.0)

    _ = segment.track(beam)

    assert isinstance(segment.my_screen.reading, torch.Tensor)
    assert segment.my_screen.reading.shape == (100, 100)
    assert torch.all(segment.my_screen.reading >= 0.0)
    assert torch.any(segment.my_screen.reading > 0.0)


def test_reading_shows_beam_ares():
    """
    Test that a screen has a reading that shows some sign of the beam having hit it.
    """
    segment = cheetah.Segment.from_ocelot(
        ocelot_lattice.cell, warnings=False, device="cpu"
    ).subcell("AREASOLA1", "AREABSCR1")
    beam = cheetah.ParticleBeam.from_astra("benchmark/cheetah/ACHIP_EA1_2021.1351.001")

    segment.AREABSCR1.resolution = torch.tensor((2448, 2040))
    segment.AREABSCR1.pixel_size = torch.tensor((3.3198e-6, 2.4469e-6))
    segment.AREABSCR1.binning = torch.tensor(1)
    segment.AREABSCR1.is_active = True

    assert isinstance(segment.AREABSCR1.reading, torch.Tensor)
    assert segment.AREABSCR1.reading.shape == (2040, 2448)
    assert np.allclose(segment.AREABSCR1.reading, 0.0)

    _ = segment.track(beam)

    assert isinstance(segment.AREABSCR1.reading, torch.Tensor)
    assert segment.AREABSCR1.reading.shape == (2040, 2448)
    assert torch.all(segment.AREABSCR1.reading >= 0.0)
    assert torch.any(segment.AREABSCR1.reading > 0.0)
