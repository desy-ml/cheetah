import pytest

from cheetah import Screen, Segment

from .resources import ARESlatticeStage3v1_9 as ares


@pytest.mark.parametrize(
    "name",
    [
        "ARLIBSCX1",
        "ARLIBSCR1",
        "ARLIBSCR2",
        "ARLIBSCR3",
        "AREABSCR1",
        "ARMRBSCR1",
        "ARMRBSCR2",
        "ARMRBSCR3",
        "ARBCBSCE1",
        "ARDLBSCR1",
        "ARDLBSCE1",
        "ARSHBSCE2",
        "ARSHBSCE1",
    ],
)
def test_screen_conversion(name: str):
    """
    Test on the example of the ARES lattice that all screens are correctly converted to
    `cheetah.Screen`.
    ˚"""
    segment = Segment.from_ocelot(ares.cell)
    screen = getattr(segment, name)
    assert isinstance(screen, Screen)
