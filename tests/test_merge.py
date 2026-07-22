import pytest
import torch

import cheetah


@pytest.mark.for_every_element("original")
def test_element_end(original):
    """
    Test that at the end of a merged element (re-merged from its splits) the result is
    the same as at the end of the original element.
    """
    original.to(torch.float64)
    split = cheetah.Segment(
        original.split(resolution=torch.tensor(0.015, dtype=torch.float64))
    )
    merged = split.with_consecutive_elements_merged()

    incoming_beam = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001", dtype=torch.float64
    )

    outgoing_beam_original = original.track(incoming_beam)
    outgoing_beam_split = split.track(incoming_beam)
    outgoing_beam_merged = merged.track(incoming_beam)

    assert torch.allclose(
        outgoing_beam_original.particles,
        outgoing_beam_merged.particles,
        rtol=1e-2 if original.tracking_method == "second_order" else 1e-5,
    )
    assert torch.allclose(
        outgoing_beam_split.particles,
        outgoing_beam_merged.particles,
        rtol=1e-2 if original.tracking_method == "second_order" else 1e-5,
    )


@pytest.mark.for_every_element("original")
def test_merge_preserves_dtype(original):
    """
    Test that the dtype of an element merged from its splits is the same as the original
    element's dtype.
    """
    original.to(torch.float64)
    splits = original.split(resolution=torch.tensor(0.1))
    segment_splits = cheetah.Segment(splits)
    merged_segment = segment_splits.with_consecutive_elements_merged()

    for merged in merged_segment.elements:
        assert original.length.dtype == merged.length.dtype
