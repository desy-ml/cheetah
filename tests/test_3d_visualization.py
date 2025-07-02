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
    mesh = segment.to_mesh(s=0.0)

    assert isinstance(mesh, trimesh.Trimesh)
