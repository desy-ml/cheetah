import numpy as np
import pytest
import torch
import trimesh

import cheetah


def test_run_and_check_return():
    """
    Test that the `to_mesh` method of the `Segment` class returns a `trimesh.Scene` and
    correctly shaped output transform matrix without raising an error.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.3)),
            cheetah.Quadrupole(length=torch.tensor(0.2)),
            cheetah.Drift(length=torch.tensor(0.1)),
            cheetah.HorizontalCorrector(length=torch.tensor(0.1)),
            cheetah.Drift(length=torch.tensor(0.3)),
        ],
    )
    mesh, output_transform = segment.to_mesh()

    assert isinstance(mesh, trimesh.Scene)
    assert isinstance(output_transform, np.ndarray)
    assert output_transform.shape == (4, 4)


def test_no_mesh_warning():
    """
    Test that a warning is raised when an element's mesh cannot be loaded, and that the
    warning message is correct.
    """
    element = cheetah.BPM(name="bpm1")

    with pytest.warns(cheetah.VisualizationWarning) as record:
        _, _ = element.to_mesh()

        assert len(record) == 1
        assert (
            record[0].message.args[0]
            == "Could not load 3D mesh for element bpm1 of type BPM. The element will "
            "not be visualised."
        )


def test_zero_length_warning():
    """
    Test that a warning is raised when an element has a length of zero, and that the
    warning message is correct.
    """
    element = cheetah.HorizontalCorrector(length=torch.tensor(0.0), name="hcorr1")

    with pytest.warns(cheetah.VisualizationWarning) as record:
        _, _ = element.to_mesh()

        assert len(record) == 1
        assert (
            record[0].message.args[0]
            == "Element hcorr1 of type HorizontalCorrector has a length of zero. The "
            "mesh is therefore scaled to a default size and does not accurately "
            "represent the element's length. If this is intentional, you can ignore "
            "this warning."
        )
