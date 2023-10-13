import numpy as np
import torch

import cheetah


def test_reading_shows_beam():
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
