import pytest
import torch

import cheetah


def test_superimposed_base_split_length():
    """
    Test that the base element of a superimposed segment is correctly split into two
    halfs, each half the length of the original base element.
    """
    superimposed_segment = cheetah.Superimposed(
        base_element=cheetah.Quadrupole(length=torch.tensor(1.0)),
        superimposed_element=cheetah.BPM(),
    )

    assert len(superimposed_segment.subelements) == 3
    assert isinstance(superimposed_segment.subelements[0], cheetah.Quadrupole)
    assert isinstance(superimposed_segment.subelements[1], cheetah.BPM)
    assert isinstance(superimposed_segment.subelements[2], cheetah.Quadrupole)
    assert superimposed_segment.subelements[0].length == torch.tensor(0.5)
    assert superimposed_segment.subelements[2].length == torch.tensor(0.5)


def test_superimposed_first_order_transfer_map():
    """
    Test that the first order transfer map of a superimposed segment is the same as the
    first order transfer map of the base element.
    """
    quadrupole = cheetah.Quadrupole(length=torch.tensor(1.0), k1=torch.tensor(4.2))
    superimposed_segment = cheetah.Superimposed(
        base_element=quadrupole, superimposed_element=cheetah.BPM()
    )

    energy = torch.tensor(1.0e9)
    species = cheetah.Species("electron")

    tm_superimposed = superimposed_segment.first_order_transfer_map(energy, species)
    tm_quadrupole = quadrupole.first_order_transfer_map(energy, species)

    assert torch.allclose(tm_superimposed, tm_quadrupole)


def test_not_flattening():
    """
    Test that a `Superimposed` element is not flattened when `.flattened()` is called on
    a `Segment`, but remains as a `Superimposed` element with the correct subelements.
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

    assert len(flattened.elements) == 3
    assert isinstance(flattened.elements[1], cheetah.Superimposed)
    assert len(flattened.elements[1].subelements) == 3


def test_superimposed_element_rejects_nonzero_length():
    """
    Test that an error is raised when attempting to superimpose a non-zero length
    element.
    """
    with pytest.raises(ValueError):
        _ = cheetah.Superimposed(
            base_element=cheetah.Quadrupole(length=torch.tensor(1.0)),
            superimposed_element=cheetah.Dipole(length=torch.tensor(0.5)),
        )
