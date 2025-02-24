import torch

import cheetah


def test_subcell_start_end():
    """
    Test that `start` and `end` have proper defaults and fallbacks if the named element
    does not exist in the original segment.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5), name=f"drift_{i}")
            for i in range(10)
        ]
    )

    # Test standard behaviour
    assert len(segment.subcell().elements) == 10
    assert len(segment.subcell(start="drift_3").elements) == 7
    assert len(segment.subcell(end="drift_4").elements) == 5
    assert len(segment.subcell(start="drift_3", end="drift_4").elements) == 2

    # Test with invalid start or end element
    assert len(segment.subcell(start="drift_42").elements) == 0
    assert len(segment.subcell(end="drift_42").elements) == 10
    assert len(segment.subcell(start="drift_5", end="drift_42").elements) == 5
    assert len(segment.subcell(start="drift_3", end="drift_2").elements) == 0


def test_subcell_endpoint():
    """Test that the subcell method properly determines the new end element."""
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5), name=f"drift_{i}")
            for i in range(10)
        ]
    )

    # Test with fixed end
    assert len(segment.subcell("drift_2", "drift_7").elements) == 6
    assert len(segment.subcell("drift_2", "drift_7", endpoint=True).elements) == 6
    assert len(segment.subcell("drift_2", "drift_7", endpoint=False).elements) == 5

    # Test with open end
    assert len(segment.subcell("drift_2").elements) == 8
    assert len(segment.subcell("drift_2", endpoint=True).elements) == 8
    assert len(segment.subcell("drift_2", endpoint=False).elements) == 8
