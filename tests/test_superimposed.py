import pytest
import torch

import cheetah


def test_superimposed_base_split_length():
    """
    Test that the base element of a superimposed segment is correctly split into two
    halfs, each half the length of the original base element.
    """
    superimposed = cheetah.Superimposed(
        base_element=cheetah.Quadrupole(length=torch.tensor(1.0)),
        superimposed_element=cheetah.BPM(),
    )

    assert len(superimposed._segment.elements) == 3
    assert isinstance(superimposed._segment.elements[0], cheetah.Quadrupole)
    assert isinstance(superimposed._segment.elements[1], cheetah.BPM)
    assert isinstance(superimposed._segment.elements[2], cheetah.Quadrupole)
    assert superimposed._segment.elements[0].length == torch.tensor(0.5)
    assert superimposed._segment.elements[2].length == torch.tensor(0.5)

    assert superimposed.length == torch.tensor(1.0)


def test_superimposed_first_order_transfer_map():
    """
    Test that the first order transfer map of a superimposed segment is the same as the
    first order transfer map of the base element.
    """
    quadrupole = cheetah.Quadrupole(length=torch.tensor(1.0), k1=torch.tensor(4.2))
    superimposed = cheetah.Superimposed(
        base_element=quadrupole, superimposed_element=cheetah.BPM()
    )

    energy = torch.tensor(1.0e9)
    species = cheetah.Species("electron")

    tm_superimposed = superimposed.first_order_transfer_map(energy, species)
    tm_quadrupole = quadrupole.first_order_transfer_map(energy, species)

    assert torch.allclose(tm_superimposed, tm_quadrupole)


def test_not_flattening():
    """
    Test that a `Superimposed` element is also flattened when `.flattened()` is called
    on a `Segment` containing it.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.Superimposed(
                base_element=cheetah.Quadrupole(
                    length=torch.tensor(1.0), k1=torch.tensor(1.0)
                ),
                superimposed_element=cheetah.BPM(),
            ),
            cheetah.Drift(length=torch.tensor(1.0)),
        ]
    )
    flattened = segment.flattened()

    assert len(flattened.elements) == 5
    assert isinstance(flattened.elements[0], cheetah.Drift)
    assert isinstance(flattened.elements[1], cheetah.Quadrupole)
    assert isinstance(flattened.elements[2], cheetah.BPM)
    assert isinstance(flattened.elements[3], cheetah.Quadrupole)
    assert isinstance(flattened.elements[4], cheetah.Drift)


def test_superimposed_element_rejects_nonzero_length():
    """
    Test that an error is raised when attempting to superimpose a non-zero length
    element.
    """
    with pytest.raises(
        AssertionError, match="The superimposed element must have zero length."
    ):
        _ = cheetah.Superimposed(
            base_element=cheetah.Quadrupole(length=torch.tensor(1.0)),
            superimposed_element=cheetah.Dipole(length=torch.tensor(0.5)),
        )


def test_superimposed_serialization(tmp_path):
    """
    Test that a `Superimposed` element can be serialized to and deserialized from JSON.
    """

    single_element_json = tmp_path / "superimposed_test.json"
    segment_element_json = tmp_path / "superimposed_segment_test.json"

    # test case where the superimposed element is a BPM
    superimposed = cheetah.Superimposed(
        base_element=cheetah.Quadrupole(length=torch.tensor(1.0), k1=torch.tensor(2.0)),
        superimposed_element=cheetah.BPM(),
        name="superimposed_test",
    )
    segment = cheetah.Segment(elements=[superimposed], name="test_segment")

    segment.to_lattice_json(str(single_element_json))
    deserialized = cheetah.Segment.from_lattice_json(str(single_element_json))

    assert isinstance(deserialized.elements[0], cheetah.Superimposed)
    superimposed_deserialized = deserialized.elements[0]
    assert superimposed_deserialized.name == "superimposed_test"
    assert isinstance(superimposed_deserialized.base_element, cheetah.Quadrupole)
    assert superimposed_deserialized.base_element.k1 == torch.tensor(2.0)
    assert isinstance(superimposed_deserialized.superimposed_element, cheetah.BPM)

    # test case where the superimposed element is a Segment
    superimposed_segment = cheetah.Segment(
        elements=[
            cheetah.BPM(name="bpm1"),
            cheetah.Marker(name="marker1"),
        ],
        name="superimposed_segment",
    )

    superimposed = cheetah.Superimposed(
        base_element=cheetah.Quadrupole(length=torch.tensor(1.0), k1=torch.tensor(2.0)),
        superimposed_element=superimposed_segment,
        name="superimposed_segment_test",
    )
    segment = cheetah.Segment(elements=[superimposed], name="test_segment_2")

    segment.to_lattice_json(str(segment_element_json))
    deserialized = cheetah.Segment.from_lattice_json(str(segment_element_json))

    assert isinstance(deserialized.elements[0], cheetah.Superimposed)
    superimposed_deserialized = deserialized.elements[0]
    assert superimposed_deserialized.name == "superimposed_segment_test"
    assert isinstance(superimposed_deserialized.base_element, cheetah.Quadrupole)
    assert superimposed_deserialized.base_element.k1 == torch.tensor(2.0)
    assert isinstance(superimposed_deserialized.superimposed_element, cheetah.Segment)
