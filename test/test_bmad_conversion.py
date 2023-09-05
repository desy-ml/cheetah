import cheetah


def test_bmad_tutorial():
    """Test importing the lattice example file from the Bmad and Tao tutorial."""
    file_path = "test/bmad/bmad_tutorial_lattice.bmad"
    converted = cheetah.Segment.from_bmad(file_path)
    converted.name = "bmad_tutorial"

    correct = cheetah.Segment(
        [
            cheetah.Drift(length=0.5, name="d"),
            cheetah.Dipole(length=0.5, e1=0.1, name="b"),  # TODO: What are g and dg?
            cheetah.Quadrupole(length=0.6, k1=0.23, name="q"),
        ],
        name="bmad_tutorial",
    )

    assert converted == correct
