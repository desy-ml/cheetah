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
            cheetah.HorizontalCorrector(name="hcorr1", length=torch.tensor(0)),
            cheetah.Drift(length=torch.tensor(0.3)),
            cheetah.BPM(name="bpm1"),
            cheetah.Screen(),
        ],
    )

    with pytest.warns(cheetah.utils.BadVisualizationWarning) as record:
        mesh, output_transform = segment.to_mesh()

    assert len(record) == 2
    assert record[0].message.args[0] =="Element hcorr1 of type HorizontalCorrector"\
    " has zero length. The mesh will not be scaled to the correct length."
    assert record[1].message.args[0] == "Could not load 3D mesh for element bpm1 of "\
    "type BPM. The element will not be visualised."

    assert isinstance(mesh, trimesh.Scene)
    assert isinstance(output_transform, np.ndarray)
    assert output_transform.shape == (4, 4)
