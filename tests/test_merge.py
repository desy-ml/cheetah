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


def test_with_consecutive_elements_merged_except_for():
    """
    Test that Segment.with_consecutive_elements_merged respects the `except_for` list.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5), name=f"d{i}") for i in range(4)
        ]
    )

    merged_segment = segment.with_consecutive_elements_merged(except_for=["d2"])

    assert len(merged_segment.elements) == 3
    assert [element.name for element in merged_segment.elements] == ["d", "d2", "d3"]


def test_with_consecutive_elements_merged_nested_segments():
    """
    Test that Segment.with_consecutive_elements_merged correctly processes nested
    segments.
    """
    d1 = cheetah.Drift(length=torch.tensor(0.5), name="drift_1")
    d2 = cheetah.Drift(length=torch.tensor(0.5), name="drift_2")
    d3 = cheetah.Drift(length=torch.tensor(0.5), name="drift_3")
    d4 = cheetah.Drift(length=torch.tensor(0.5), name="drift_4")
    d5 = cheetah.Drift(length=torch.tensor(0.5), name="drift_5")

    sub1 = cheetah.Segment(elements=[d1, d2], name="sub1")
    sub2 = cheetah.Segment(elements=[d3, d4], name="sub2")

    parent = cheetah.Segment(elements=[sub1, sub2, d5], name="parent")
    merged_parent = parent.with_consecutive_elements_merged()

    # sub1 and sub2 are nested segments whose inner elements are merged
    merged_sub1 = merged_parent.elements[0]
    assert isinstance(merged_sub1, cheetah.Segment)
    assert len(merged_sub1.elements) == 1

    merged_sub2 = merged_parent.elements[1]
    assert isinstance(merged_sub2, cheetah.Segment)
    assert len(merged_sub2.elements) == 1


def test_with_consecutive_elements_merged_different_in_middle():
    """
    Test that merging consecutive elements in a `Segment` correctly handles when a
    different type of element is in the middle of mergeable elements.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5), name="d1"),
            cheetah.Drift(length=torch.tensor(0.5), name="d2"),
            cheetah.Quadrupole(length=torch.tensor(0.2), name="q1"),
            cheetah.Drift(length=torch.tensor(0.5), name="d3"),
            cheetah.Drift(length=torch.tensor(0.5), name="d4"),
        ]
    )

    merged_segment = segment.with_consecutive_elements_merged()

    assert len(merged_segment.elements) == 3
    assert [element.name for element in merged_segment.elements] == ["d", "q1", "d"]
