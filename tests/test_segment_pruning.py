import numpy as np
import torch

import cheetah


def test_inactive_elements_as_drifts():
    """Test that the conversion into drifts works properly."""
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor([1.0, 2.0])),
            cheetah.Dipole(
                length=torch.tensor([0.1, 0.0]),
                angle=torch.tensor([0.0, 0.0]),
                name="Dipole1",
            ),
            cheetah.Dipole(
                length=torch.tensor([0.0, 0.1]),
                angle=torch.tensor([0.0, 0.0]),
                name="Dipole2",
            ),
            cheetah.Dipole(
                length=torch.tensor([0.2, 0.1]),
                angle=torch.tensor([0.5, 0.0]),
                name="Dipole3",
            ),
            cheetah.Dipole(
                length=torch.tensor([0.0, 0.0]),
                angle=torch.tensor([0.0, 0.0]),
                name="Dipole4",
            ),
            cheetah.Drift(length=torch.tensor([0.0, 2.0])),
            cheetah.BPM(is_active=torch.tensor([False, False]), name="Bpm").broadcast(
                (2,)
            ),
        ]
    )

    pruned = segment.inactive_elements_as_drifts()
    pruned_except = segment.inactive_elements_as_drifts(except_for=["Dipole2"])

    assert len(segment.elements) == len(pruned.elements)
    assert len(segment.elements) == len(pruned_except.elements)
    assert np.allclose(segment.length, pruned.length)
    assert np.allclose(segment.length, pruned_except.length)

    assert isinstance(pruned.Dipole1, cheetah.Drift)
    assert isinstance(pruned.Dipole2, cheetah.Drift)
    assert isinstance(pruned.Dipole3, cheetah.Dipole)
    assert isinstance(pruned.Dipole4, cheetah.Dipole)
    assert isinstance(pruned.Bpm, cheetah.BPM)
    assert isinstance(pruned_except.Dipole1, cheetah.Drift)
    assert isinstance(pruned_except.Dipole2, cheetah.Dipole)
    assert isinstance(pruned_except.Dipole3, cheetah.Dipole)
    assert isinstance(pruned_except.Dipole4, cheetah.Dipole)
    assert isinstance(pruned_except.Bpm, cheetah.BPM)


def test_without_zerolength_elements():
    """Test that zerolength elements are properly recognized and removed."""
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor([1.0, 2.0])),
            cheetah.Dipole(
                length=torch.tensor([0.0, 0.0]),
                angle=torch.tensor([0.0, 0.0]),
                name="Dipole1",
            ),
            cheetah.Drift(length=torch.tensor([1.0, 0.0])),
            cheetah.Dipole(
                length=torch.tensor([0.0, 0.0]),
                angle=torch.tensor([0.0, 0.0]),
                name="Dipole2",
            ),
            cheetah.Drift(length=torch.tensor([0.0, 2.0])),
            cheetah.Dipole(
                length=torch.tensor([0.0, 0.1]),
                angle=torch.tensor([0.0, 0.0]),
                name="Dipole3",
            ),
            cheetah.Drift(length=torch.tensor([0.0, 0.0])),
            cheetah.Dipole(
                length=torch.tensor([0.0, 0.0]),
                angle=torch.tensor([0.5, 0.0]),
                name="Dipole4",
            ),
        ]
    )

    pruned = segment.without_inactive_zero_length_elements()
    pruned_except = segment.without_inactive_zero_length_elements(
        except_for=["Dipole2"]
    )

    assert len(segment.elements) == 8
    assert len(pruned.elements) == 5
    assert len(pruned_except.elements) == 6
    assert np.allclose(segment.length, pruned.length)
    assert np.allclose(segment.length, pruned_except.length)
    assert not torch.all(pruned_except.Dipole2.is_active)
