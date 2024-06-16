import lynx


def test_no_error():
    """
    Just test that reading ARES from NX Tables doesn't throw an error and produces
    something somewhat plausbile.
    """
    segment = lynx.Segment.from_nx_tables("tests/resources/Stage4v3_9.txt")

    assert isinstance(segment, lynx.Segment)
    assert len(segment.elements) > 1
    assert 40.0 < segment.length < 50.0


def test_length():
    """
    When first designing NX Table read, the lattice looked plausbile and the length came
    out as 44.2215 m. This test is to make sure that reading from NX Tables still
    produces this length. NOTE, that this not to say that that is actually the correct
    length.
    """
    segment = lynx.Segment.from_nx_tables("tests/resources/Stage4v3_9.txt")

    assert segment.length == 44.2215
