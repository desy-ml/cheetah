import cheetah
from cheetah.converters import madx


def test_madx_tfs():
    madx_tfs_file_path = "tests/resources/twiss_tt43_nom.tfs"
    converted_segment = madx.convert_madx_lattice(madx_tfs_file_path)
    print(converted_segment)
    assert isinstance(converted_segment.elements[7], cheetah.Quadrupole)  # the elements
    assert isinstance(converted_segment.elements[19], cheetah.RBend)  # the elements


# test_madx_tfs()
