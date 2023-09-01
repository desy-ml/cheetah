from cheetah.accelerator import Dipole, Drift, Quadrupole, Segment

Dipole1 = Dipole(
    name="dipole1",
    length=1.0,
    angle=0.1,
    e1=0.2,
    e2=0.3,
    gap=0.4,
    tilt=0.5,
    fint=0.6,
    fintx=0.7,
)

Dipole2 = Dipole(
    name="dipole2",
    length=2.0,
    angle=0.1,
    e1=0.2,
    e2=0.3,
    gap=0.4,
    tilt=0.5,
    fint=0.6,
    fintx=0.7,
)

Drift1 = Drift(name="Drift1", length=0.1)
Drift2 = Drift(name="Drift1", length=0.1)


Quadrupole1 = Quadrupole(name="Q1", length=0.1, k1=0.2)
Quadrupole2 = Quadrupole(name="Q1", length=0.1, k1=0.2)

segment1 = Segment(cell=[Dipole1, Drift1, Quadrupole1], name="TestSegment")
segment2 = Segment(cell=[Dipole2, Drift1, Quadrupole1], name="TestSegment")
segment3 = Segment(cell=[Dipole1, Drift2, Quadrupole2], name="TestSegment")


def test_elements_equal():
    assert Dipole1 != Dipole2
    assert Drift1 == Drift2
    assert Quadrupole1 == Quadrupole2


def test_segment_equal():
    assert segment1 != segment2
    assert segment1 == segment3
