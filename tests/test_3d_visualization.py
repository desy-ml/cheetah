import os

import pytest
import torch
import trimesh

import cheetah


@pytest.fixture
def segment():
    # Mock or create a Segment instance to be used in tests
    return cheetah.Segment.from_lattice_json(filepath="tests/mock_config.json")


def test_builder_initialization(segment):
    builder = cheetah.utils.segment_3d_builder.Segment3DBuilder(segment)
    assert builder.segment == segment


def test_component_positions(segment):
    builder = cheetah.utils.segment_3d_builder.Segment3DBuilder(segment)
    assert isinstance(builder.component_positions, dict)


def test_export_function(segment):
    builder = cheetah.utils.segment_3d_builder.Segment3DBuilder(segment)
    output_filename = "test_scene.glb"
    builder.build_segment(output_filename=output_filename, is_export_enabled=True)
    assert os.path.exists(output_filename)


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

    assert isinstance(mesh, trimesh.Scene)
