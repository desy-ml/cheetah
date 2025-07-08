import numpy as np
import torch
import trimesh

import cheetah


def test_segment_method():
    """
    Test that the `to_mesh` method of the `Segment` class returns a trimesh.Scene
    without raising an error.

    TODO: Rename test once `Segment3DBuilder` is phased out and refine test depth.
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

    assert isinstance(mesh, trimesh.Trimesh)
    assert isinstance(output_transform, np.ndarray)
