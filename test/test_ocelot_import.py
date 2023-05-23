import test.ARESlatticeStage3v1_9 as ares

from cheetah import Screen, Segment


def test_screen_conversion():
    """
    Test on the example of the ARES lattice that all screens are correctly converted to
    `cheetah.Screen`.
    ˚"""
    segment = Segment.from_ocelot(ares.cell)

    assert isinstance(segment.ARLIBSCR1, Screen)
    assert isinstance(segment.ARLIBSCR2, Screen)
    assert isinstance(segment.ARLIBSCR3, Screen)
    assert isinstance(segment.AREABSCR1, Screen)
    assert isinstance(segment.ARMRBSCR1, Screen)
    assert isinstance(segment.ARMRBSCR2, Screen)
    assert isinstance(segment.ARMRBSCR3, Screen)
    assert isinstance(segment.ARDLBSCR1, Screen)
