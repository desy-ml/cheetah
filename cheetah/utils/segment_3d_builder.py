"""
A 3D builder class for creating and exporting accelerator lattice segments.
Handles the construction and visualization of beam line elements like dipoles,
quadrupoles, monitors etc. as 3D models in a unified scene.

This module handles:
- Configuration parsing for accelerator components using cheetah-accelerator
- GLB scene assembly and export
- JSON configuration

Usage Example:
    builder = Segment3DBuilder(config)
    builder.build_segment("scene.glb")
"""

import argparse
import logging
import math
import os
from importlib.resources import files
from typing import Any

import torch
import trimesh

from cheetah import (
    Cavity,
    Dipole,
    HorizontalCorrector,
    Quadrupole,
    Screen,
    Segment,
    Undulator,
    VerticalCorrector,
    _assets,
)

# Set logging level based on environment
debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Setup logging
logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO)
logger = logging.getLogger(__name__)


# Constants
DEFAULT_SCALE_FACTOR = 0.20
DEFAULT_ROTATION_ANGLE = 2 * math.pi
DEFAULT_ROTATION_AXIS = [0, 1, 0]


class MeshTransformer:
    """Helper class for 3D mesh transformations."""

    def __init__(
        self, scale_factor: float, rotation_angle: float, rotation_axis: list[float]
    ):
        """
        Initializes the object with scaling and rotation parameters.

        :param scale_factor: The factor by which to scale an object.
        :param rotation_angle: The angle (in degrees or radians)
                               by which to rotate the object.
        :param rotation_axis: A 3D vector representing the axis of rotation.
        """
        self.scale_factor = scale_factor
        self.rotation_angle = rotation_angle
        self.rotation_axis = rotation_axis

    def transform_mesh(
        self, mesh: trimesh.Trimesh, translation_vector: list[float]
    ) -> None:
        """Apply transformations to mesh."""
        rotation_matrix = trimesh.transformations.rotation_matrix(
            self.rotation_angle, self.rotation_axis
        )
        mesh.apply_scale(self.scale_factor)
        mesh.apply_transform(rotation_matrix)
        mesh.apply_translation(translation_vector)


class Segment3DBuilder:
    """
    Builds 3D representations of accelerator lattice segments
    by creating and positioning detailed models of beamline components
    in a virtual scene.

    These models represent various beamline elements used in particle accelerators,
    including:
    - Dipole magnets (used for beam bending)
    - Vertical and horizontal correctors (for beam steering)
    - Quadrupole magnets (for beam focusing)
    - Screen (for beam diagnostics)
    - RF Cavities (for accelerating particles)
    - Undulators (for generating synchrotron radiation)

    The builder arranges these components in 3D space based on their sequence
    and configuration parameters.
    """

    def __init__(self, segment: Segment):
        """
        Initialize the 3D segment builder.

        Args:
            segment: Lattice element specification.
        """
        # Segment object representing a sequence of lattice elements
        # (e.g., Dipole, Quadrupole) that define the accelerator beamline.
        self.segment = segment

        # Maps each accelerator component type to its corresponding 3D model
        # asset file (.glb).
        # This dictionary allows dynamic lookup of the appropriate 3D model for
        # each element in the scene.
        self.asset_map = {
            VerticalCorrector: "vertical_corrector.glb",
            HorizontalCorrector: "horizontal_corrector.glb",
            Quadrupole: "quadrupole.glb",
            Screen: "screen.glb",
            Cavity: "cavity.glb",
            Undulator: "undulator.glb",
        }

        # Default transformation settings for scaling and rotation of 3D models
        config = {
            "scale_factor": DEFAULT_SCALE_FACTOR,
            "rotation_angle": DEFAULT_ROTATION_ANGLE,
            "rotation_axis": DEFAULT_ROTATION_AXIS,
        }

        # Handles scaling, rotation, and positioning of 3D models
        self.transformer = MeshTransformer(
            scale_factor=config["scale_factor"],
            rotation_angle=config["rotation_angle"],
            rotation_axis=config["rotation_axis"],
        )

        # Track the current longitudinal position along the segment
        self.current_position = 0.0

        # Determine the base directory for assets (or for storing outputs)
        self.assets_dir = os.path.dirname(_assets.__file__)

        # Creates a visualization scene using triangular meshes with an
        # automatically generated camera and lighting
        self.scene = trimesh.Scene()

        # Track lattice component positions
        self._component_positions: dict[str, float] = {}

    @property
    def current_position(self) -> float:
        """Current longitudinal position along the segment."""
        return self._current_position

    @current_position.setter
    def current_position(self, value: float) -> None:
        """Set current longitudinal position."""
        self._current_position = value

    @property
    def component_positions(self) -> dict:
        """Return a copy of component positions to prevent external modification."""
        return self._component_positions.copy()

    @component_positions.setter
    def component_positions(self, positions: dict) -> None:
        """
        Set component positions safely, ensuring correct format.

        :param positions: A dictionary containing the component positions.
                          The keys are the component identifiers and
                          the values are their respective positions.
        :raises ValueError: If the input is not a dictionary.
        """
        if not isinstance(positions, dict):
            raise ValueError("component_positions must be a dictionary.")
        self._component_positions = positions

    def build_segment(
        self, output_filename: str | None = None, is_export_enabled: bool = True
    ) -> None:
        """
        Build a complete 3D segment scene by iterating through the segment
        and adding the elements (Dipole/Quadrupole) to the 3D scene.
        Other elements (like Drift) only advance the current position.

        :param output_filename: Optional; the output GLB/GLTF filename
                                for the exported 3D scene. If not provided,
                                the scene will not be exported.
        :param is_export_enabled: A boolean flag to enable or disable the export.
                                  Default is True, meaning the scene will
                                  be exported if an output filename is provided.
        """
        # For Drift, or other elements, we do not add geometry.
        for element in self.segment.elements:
            self.add_element_to_scene(element)

            length = (
                element.length.item()
                if isinstance(element.length, torch.Tensor)
                else float(element.length)
            )

            self.current_position += length

        # Represents the total length after processing last element
        logger.info("Final lattice segment length: %s", self.current_position)

        # Export the final scene to a GLTF file
        if is_export_enabled:
            self.export(output_filename)

    def add_element_to_scene(
        self,
        element: (
            Cavity
            | Dipole
            | HorizontalCorrector
            | Quadrupole
            | Screen
            | Undulator
            | VerticalCorrector
        ),
    ) -> None:
        """
        Add an element to the scene by loading its mesh, transforming it,
        and tracking its position.

        :param element: The element to be added to the scene.
                        It can be one of the following types:
                            Cavity, Dipole, HorizontalCorrector, Quadrupole,
                            Screen, Undulator, or VerticalCorrector.
        :raises: If the element type is not recognized, a warning is logged.
        """
        if type(element) in self.asset_map:
            self._load_and_transform_mesh(element)
            self._track_component_position(element.name)
            logger.info(
                "Added %s: %s at position %s",
                element.__class__.__name__,
                element.name,
                self.current_position,
            )
        else:
            logger.warning("Element type %s not recognized.", type(element).__name__)

    def export(self, output_file: str | None = None) -> None:
        """
        Export the 3D scene to a file in the GLB format.

        :param output_file: Optional; the output filename for the exported scene.
                            If not provided, the scene will be saved as "scene.glb"
                            in the assets directory.
        :raises Exception: If there is an error during the export process,
                           an exception is raised and logged.
        """
        # Optionally, store the exported .glb file in the assets directory,
        # using a default filename if none is provided.
        if output_file is None:
            output_file = "scene.glb"
            output_path = os.path.join(self.assets_dir, output_file)
        else:
            output_path = output_file  # Use provided path

        try:
            self.scene.export(output_path, include_normals=True)
            logger.info("Exported 3D scene to %s", output_path)
        except Exception as e:
            logger.error("Failed to export scene: %s", e)
            raise

    def _load_and_transform_mesh(self, element: Any) -> None:
        """
        Load and transform the 3D model mesh based on the element type to the scene.

        :param element: An accelerator lattice component (e.g., Quadrupole, Dipole).
                        This represents a physical element to be visualized
                        in the 3D scene.
        :raises ValueError: If the asset for the element type is not found
                            in the asset_map.
        :raises Exception: If there is an error while loading or transforming the mesh.
        """
        asset_filename = self.asset_map.get(type(element))

        if asset_filename is None:
            raise ValueError(
                f"Asset for element type '{type(element).__name__}' not found."
            )

        try:
            # Use importlib.resources to access the asset file.
            asset_path = files(_assets) / asset_filename

            # Force loading 3D model as a scene to ensure multiple geometries are
            # handled properly. Additioanlly, try to coerce everything into a scene
            # instead of a single mesh
            scene = trimesh.load(str(asset_path), file_type="glb", force="scene")

            for mesh in scene.geometry.values():
                # Translation vector [x, y, z] defining the model's position
                # in 3D space. Determines where the component is placed within the scene
                translation_vector = [0, 0, self.current_position]
                self.transformer.transform_mesh(mesh, translation_vector)
                self.scene.add_geometry(mesh)

        except Exception as e:
            logger.error("Failed to load mesh for asset key %s: %s", self.asset_map, e)
            raise

    def _track_component_position(self, component_name: str) -> None:
        """
        Store the current position of a component dynamically.

        :param component_name: The name of the component whose position
                               is being tracked. The name is used as a key
                               to store the position in the component positions
                               dictionary.
        """
        self._component_positions[component_name] = self.current_position

    def __repr__(self) -> str:
        """String representation."""
        return f"Segment3DBuilder(elements={len(self.segment.elements)},\
        position={self.current_position})"


def main():
    """
    Main function to parse arguments, build the 3D segment, and export it to a file.
    """
    parser = argparse.ArgumentParser(
        description="Build and export a 3D accelerator lattice scene."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.json",
        help="Path to the configuration file (default: config.json)",
    )
    parser.add_argument(
        "-o",
        "--output-scene",
        default=None,
        help="Output GLB filename (default: None)",
    )
    args = parser.parse_args()

    # Create a cheetah segment with a mixture of elements from JSON file
    segment = Segment.from_lattice_json(filepath=args.config)

    # Setup 3D lattice segment
    builder = Segment3DBuilder(segment)

    # Build and export the 3D scene
    builder.build_segment(output_filename=args.output_scene)

    # Show rendered 3D scene (if desired)
    builder.scene.show()


if __name__ == "__main__":
    main()
