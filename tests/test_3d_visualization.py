import os

import pytest

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
