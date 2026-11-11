from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

import cheetah
from cheetah.utils.blender import find_blender_binary

try:
    find_blender_binary()
    HAS_BLENDER = True
except FileNotFoundError:
    HAS_BLENDER = False

blender_required = pytest.mark.skipif(
    not HAS_BLENDER, reason="Blender is not installed on this system"
)


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


@blender_required
def test_to_blender_segment(tmp_path: Path):
    """
    Test that the `to_blender` method of `Segment` generates a valid .blend file.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.3)),
            cheetah.Quadrupole(length=torch.tensor(0.2), name="quad1"),
            cheetah.Drift(length=torch.tensor(0.1)),
            cheetah.HorizontalCorrector(length=torch.tensor(0.1), name="hcorr1"),
            cheetah.Drift(length=torch.tensor(0.3)),
        ],
    )
    dest_path = tmp_path / "lattice.blend"
    returned_path = segment.to_blender(filepath=dest_path)

    assert returned_path == dest_path.resolve()
    assert dest_path.exists()
    assert dest_path.stat().st_size > 0


@blender_required
def test_to_blender_single_element(tmp_path: Path):
    """
    Test that the `to_blender` method of `Element` generates a valid .blend file.
    """
    element = cheetah.Quadrupole(length=torch.tensor(0.2), name="quad1")
    dest_path = tmp_path / "quad.blend"
    returned_path = element.to_blender(filepath=dest_path)

    assert returned_path == dest_path.resolve()
    assert dest_path.exists()
    assert dest_path.stat().st_size > 0


@blender_required
def test_to_blender_no_mesh_warning(tmp_path: Path):
    """
    Test that a warning is raised when an element's 3D asset cannot be loaded in
    `to_blender`, and that the warning message is correct.
    """
    element = cheetah.BPM(name="bpm1")
    dest_path = tmp_path / "bpm.blend"

    with pytest.warns(cheetah.VisualizationWarning) as record:
        _ = element.to_blender(filepath=dest_path)

        assert len(record) == 1
        assert (
            record[0].message.args[0]
            == "Could not load 3D mesh for element bpm1 of type BPM. The element will "
            "not be visualised."
        )


@blender_required
def test_to_blender_zero_length_warning(tmp_path: Path):
    """
    Test that a warning is raised when an element has zero length in `to_blender`.
    """
    element = cheetah.HorizontalCorrector(length=torch.tensor(0.0), name="hcorr1")
    dest_path = tmp_path / "hcorr0.blend"

    with pytest.warns(cheetah.VisualizationWarning) as record:
        _ = element.to_blender(filepath=dest_path)

        assert len(record) == 1
        assert (
            record[0].message.args[0]
            == "Element hcorr1 of type HorizontalCorrector has a length of zero. The "
            "mesh is therefore scaled to a default size and does not accurately "
            "represent the element's length. If this is intentional, you can ignore "
            "this warning."
        )


def test_to_blender_binary_not_found(tmp_path: Path):
    """
    Test that `FileNotFoundError` is raised when the Blender executable cannot be found.
    """
    segment = cheetah.Segment(
        elements=[cheetah.Quadrupole(length=torch.tensor(0.2), name="quad1")]
    )
    with pytest.raises(FileNotFoundError, match="does not exist or is not executable"):
        segment.to_blender(
            filepath=tmp_path / "test.blend",
            blender_path="/nonexistent/path/to/blender",
        )
