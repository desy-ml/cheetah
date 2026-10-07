import pytest
import torch

import cheetah


def test_superimposed_base_split_length():
    """
    Test that the base element of a superimposed segment is correctly split into two
    halves, each half the length of the original base element.
    """
    base_quad = cheetah.Quadrupole(name="q1", length=torch.tensor(1.0))
    superimposed = cheetah.Superimposed(
        base_element=base_quad,
        superimposed_element=cheetah.BPM(name="bpm1"),
        name="super1",
    )

    # Base element name must not be mutated
    assert base_quad.name == "q1"
    assert superimposed.base_element.name == "q1"

    flattened = superimposed.flattened()
    assert len(flattened.elements) == 3
    assert isinstance(flattened.elements[0], cheetah.Quadrupole)
    assert isinstance(flattened.elements[1], cheetah.BPM)
    assert isinstance(flattened.elements[2], cheetah.Quadrupole)
    assert flattened.elements[0].length == torch.tensor(0.5)
    assert flattened.elements[2].length == torch.tensor(0.5)
    assert flattened.elements[0].name == "super1_1"
    assert flattened.elements[2].name == "super1_2"

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
        ValueError, match="The superimposed element must have zero length."
    ):
        _ = cheetah.Superimposed(
            base_element=cheetah.Quadrupole(length=torch.tensor(1.0)),
            superimposed_element=cheetah.Dipole(length=torch.tensor(0.5)),
        )


def test_superimposed_serialization(tmp_path):
    """
    Test that a `Superimposed` element can be serialised to and deserialised from JSON.

    The base elements share their name with the `Superimposed` element, as is the case
    for elements coming from the Bmad importer. This means they are renamed internally,
    so this also tests that the names of the base element halves are unchanged by the
    round trip.
    """
    # Test case where the superimposed element is a `BPM`
    superimposed = cheetah.Superimposed(
        base_element=cheetah.Quadrupole(
            length=torch.tensor(1.0), k1=torch.tensor(2.0), name="superimposed_test"
        ),
        superimposed_element=cheetah.BPM(name="bpm0"),
        name="superimposed_test",
    )
    segment = cheetah.Segment(elements=[superimposed], name="test_segment")

    assert segment.flattened().element_names == [
        "superimposed_test_1",
        "bpm0",
        "superimposed_test_2",
    ]

    segment.to_lattice_json(str(tmp_path / "superimposed_test.json"))
    deserialized = cheetah.Segment.from_lattice_json(
        str(tmp_path / "superimposed_test.json")
    )

    assert isinstance(deserialized.elements[0], cheetah.Superimposed)
    superimposed_deserialized = deserialized.elements[0]
    assert superimposed_deserialized.name == "superimposed_test"
    assert isinstance(superimposed_deserialized.base_element, cheetah.Quadrupole)
    assert (
        superimposed_deserialized.base_element.name == "superimposed_test_base_element"
    )
    assert superimposed_deserialized.base_element.k1 == torch.tensor(2.0)
    assert isinstance(superimposed_deserialized.superimposed_element, cheetah.BPM)
    assert deserialized.flattened().element_names == [
        "superimposed_test_1",
        "bpm0",
        "superimposed_test_2",
    ]

    # Test case where the superimposed element is a `Segment`
    superimposed_segment = cheetah.Segment(
        elements=[
            cheetah.BPM(name="bpm1"),
            cheetah.Marker(name="marker1"),
        ],
        name="superimposed_segment",
    )

    superimposed = cheetah.Superimposed(
        base_element=cheetah.Quadrupole(
            length=torch.tensor(1.0), k1=torch.tensor(2.0), name="q1"
        ),
        superimposed_element=superimposed_segment,
        name="q1",
    )
    segment = cheetah.Segment(elements=[superimposed], name="test_segment_2")

    assert segment.flattened().element_names == ["q1_1", "bpm1", "marker1", "q1_2"]

    segment.to_lattice_json(str(tmp_path / "superimposed_segment_test.json"))
    deserialized = cheetah.Segment.from_lattice_json(
        str(tmp_path / "superimposed_segment_test.json")
    )

    assert isinstance(deserialized.elements[0], cheetah.Superimposed)
    superimposed_deserialized = deserialized.elements[0]
    assert superimposed_deserialized.name == "q1"
    assert isinstance(superimposed_deserialized.base_element, cheetah.Quadrupole)
    assert superimposed_deserialized.base_element.name == "q1_base_element"
    assert superimposed_deserialized.base_element.k1 == torch.tensor(2.0)
    assert isinstance(superimposed_deserialized.superimposed_element, cheetah.Segment)
    assert deserialized.flattened().element_names == ["q1_1", "bpm1", "marker1", "q1_2"]
