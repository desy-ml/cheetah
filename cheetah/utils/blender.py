import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

BLENDER_ASSEMBLY_SCRIPT = """
import json
import sys
import bpy
from mathutils import Matrix, Vector

data_filepath = sys.argv[-2]
output_filepath = sys.argv[-1]

with open(data_filepath, "r", encoding="utf-8") as f:
    elements_data = json.load(f)

# Clear default scene to start completely fresh
bpy.ops.wm.read_factory_settings(use_empty=True)

# Create main Lattice collection
lattice_col = bpy.data.collections.new("Lattice")
bpy.context.scene.collection.children.link(lattice_col)

# Create hidden assets collection to store linked asset collections
assets_col = bpy.data.collections.new("_Assets")
bpy.context.scene.collection.children.link(assets_col)
assets_col.hide_viewport = True
assets_col.hide_render = True

loaded_assets = {}

for el in elements_data:
    asset_path = el["asset_path"]
    el_type = el["type"]
    el_name = el["name"]
    matrix_data = el["matrix"]
    cuteness = el.get("cuteness", 1.0)
    length = el.get("length", 0.0)

    if el_type not in loaded_assets:
        with bpy.data.libraries.load(asset_path, link=False) as (data_from, data_to):
            data_to.collections = data_from.collections

        master_col = bpy.data.collections.new(f"Asset_{el_type}")
        assets_col.children.link(master_col)
        for col in data_to.collections:
            master_col.children.link(col)

        # Ensure hidden render objects are also hidden in viewport
        for obj in master_col.all_objects:
            if getattr(obj, "hide_render", False):
                obj.hide_viewport = True

        # Measure base length along Y axis for renderable meshes
        renderable_meshes = [
            obj
            for obj in master_col.all_objects
            if obj.type == "MESH" and not obj.hide_render
        ]
        if renderable_meshes:
            min_y = min(
                min((obj.matrix_world @ Vector(pt))[1] for pt in obj.bound_box)
                for obj in renderable_meshes
            )
            max_y = max(
                max((obj.matrix_world @ Vector(pt))[1] for pt in obj.bound_box)
                for obj in renderable_meshes
            )
            base_length = max_y - min_y
        else:
            base_length = 1.0

        loaded_assets[el_type] = (master_col, base_length)

    master_col, base_length = loaded_assets[el_type]

    # Compute scaling factor
    scale = cuteness
    if length > 0.0 and base_length > 0.0:
        scale *= (length / base_length)

    # Create empty collection instance
    inst = bpy.data.objects.new(el_name, None)
    inst.instance_type = "COLLECTION"
    inst.instance_collection = master_col

    # Apply rigid transformation and uniform scale
    m_rigid = Matrix(matrix_data)
    m_scale = Matrix.Diagonal((scale, scale, scale, 1.0))
    inst.matrix_world = m_rigid @ m_scale
    lattice_col.objects.link(inst)

# Add basic lighting and camera
sun_data = bpy.data.lights.new(name="Sun", type="SUN")
sun_data.energy = 3.0
sun_obj = bpy.data.objects.new(name="Sun", object_data=sun_data)
sun_obj.location = (5.0, 5.0, 10.0)
sun_obj.rotation_euler = (0.785, 0.0, 0.785)
bpy.context.scene.collection.objects.link(sun_obj)

cam_data = bpy.data.cameras.new(name="Camera")
cam_obj = bpy.data.objects.new(name="Camera", object_data=cam_data)
cam_obj.location = (5.0, 5.0, 5.0)
cam_obj.rotation_euler = (1.1, 0.0, 2.3)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# Save final .blend file
bpy.ops.wm.save_as_mainfile(filepath=output_filepath)
"""


def find_blender_binary(blender_path: str | Path | None = None) -> Path:
    """
    Find the Blender executable binary on the system.

    :param blender_path: Path to the Blender executable. If provided, checks that it
        exists and is executable.
    :return: The `Path` to the Blender executable.
    :raises FileNotFoundError: If the Blender executable could not be found.
    """
    if blender_path is not None:
        path = Path(blender_path)
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise FileNotFoundError(
            f"The specified Blender executable does not exist or is not executable: "
            f"{blender_path}"
        )

    # Check environment variable
    env_path = os.getenv("BLENDER_PATH")
    if env_path:
        path = Path(env_path)
        if path.is_file() and os.access(path, os.X_OK):
            return path

    # Check PATH
    which_path = shutil.which("blender")
    if which_path:
        return Path(which_path)

    # Check common system paths
    candidate_paths = [
        Path("/Applications/Blender.app/Contents/MacOS/Blender"),
        Path("/opt/homebrew/bin/blender"),
        Path.home() / "Applications/Blender.app/Contents/MacOS/Blender",
        Path("/usr/bin/blender"),
        Path("/usr/local/bin/blender"),
        Path("/snap/bin/blender"),
    ]
    for candidate in candidate_paths:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    raise FileNotFoundError(
        "Could not find the Blender executable. Please ensure Blender is installed "
        "and in your PATH, set the BLENDER_PATH environment variable, or pass "
        "'blender_path' explicitly."
    )


def assemble_blender_scene(
    elements_data: list[dict], output_filepath: Path, blender_path: Path
) -> None:
    """
    Assemble the elements into a Blender scene and save as a .blend file.

    :param elements_data: List of element metadata dictionaries containing names,
        asset paths, transforms, scaling, etc.
    :param output_filepath: Target path for the output .blend file.
    :param blender_path: Path to the Blender executable binary.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        data_file = tmp_dir_path / "elements_data.json"
        script_file = tmp_dir_path / "assemble_scene.py"

        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(elements_data, f)

        with open(script_file, "w", encoding="utf-8") as f:
            f.write(BLENDER_ASSEMBLY_SCRIPT)

        cmd = [
            str(blender_path),
            "-b",
            "-P",
            str(script_file),
            "--",
            str(data_file),
            str(output_filepath),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Blender scene assembly failed with exit code {result.returncode}:\n"
                f"{result.stderr or result.stdout}"
            )
