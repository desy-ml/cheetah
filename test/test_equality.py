from cheetah.accelerator import Drift, Quadrupole, Segment


def test_drift_equal():
    """Test that equal drift sections are recognised as such."""
    drift_1 = Drift(name="drift_eq", length=0.1)
    drift_2 = Drift(name="drift_eq", length=0.1)
    assert drift_1 == drift_2


def test_drift_not_equal():
    """Test that unequal drift sections are recognised as such."""
    drift_1 = Drift(name="drift_1", length=0.1)
    drift_2 = Drift(name="drift_2", length=0.2)
    assert drift_1 != drift_2


def test_quadrupole_equal():
    """Test that equal quadrupoles are recognised as such."""
    quadrupole_1 = Quadrupole(name="quadrupole_eq", length=0.1, k1=0.2)
    quadrupole_2 = Quadrupole(name="quadrupole_eq", length=0.1, k1=0.2)
    assert quadrupole_1 == quadrupole_2


def test_quadrupole_not_equal():
    """Test that unequal quadrupoles are recognised as such."""
    quadrupole_1 = Quadrupole(name="quadrupole_1", length=0.1, k1=0.2)
    quadrupole_2 = Quadrupole(name="quadrupole_2", length=0.2, k1=0.2)
    assert quadrupole_1 != quadrupole_2


def test_segment_equal():
    """Test that equal segments are recognised as such."""
    segment_1 = Segment(
        cell=[Drift(name="d", length=0.1), Quadrupole(name="q", length=0.1, k1=0.2)],
        name="test_segment",
    )
    segment_2 = Segment(
        cell=[Drift(name="d", length=0.1), Quadrupole(name="q", length=0.1, k1=0.2)],
        name="test_segment",
    )
    assert segment_1 == segment_2


def test_segment_not_equal():
    """Test that unequal segments are recognised as such."""
    segment_1 = Segment(
        cell=[Drift(name="d", length=0.1), Quadrupole(name="q", length=0.1, k1=0.2)],
        name="test_segment",
    )
    segment_2 = Segment(
        cell=[Quadrupole(name="q", length=0.2, k1=0.2), Drift(name="d", length=0.1)],
        name="test_segment",
    )
    assert segment_1 != segment_2
