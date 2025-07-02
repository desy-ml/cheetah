import pytest
import torch

import cheetah


@pytest.mark.for_every_element("original")
def test_element_end(original):
    """
    Test that at the end of a split element the result is the same as at the end of the
    original element.
    """

    split = cheetah.Segment(original.split(resolution=torch.tensor(0.015)))

    incoming_beam = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    outgoing_beam_original = original.track(incoming_beam)
    outgoing_beam_split = split.track(incoming_beam)

    assert torch.allclose(
        outgoing_beam_original.particles, outgoing_beam_split.particles
    )


@pytest.mark.for_every_element("original")
def test_split_preserves_dtype(original):
    """
    Test that the dtype of an element's splits is the same as the original element's
    dtype.
    """
    original.to(torch.float64)
    splits = original.split(resolution=torch.tensor(0.1))

    for split in splits:
        assert original.length.dtype == split.length.dtype
