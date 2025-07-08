import numpy as np
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

    assert isinstance(mesh, trimesh.Trimesh)
    assert isinstance(output_transform, np.ndarray)
    assert output_transform.shape == (4, 4)
