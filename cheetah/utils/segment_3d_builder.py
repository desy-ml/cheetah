from typing import Dict, List, Optional, Union, Any
import os
import math
import json
import logging
from pathlib import Path

import torch
import trimesh
import importlib.resources as pkg_resources
import cheetah._assets
from cheetah import _assets  # note the underscore to indicate a private module
from cheetah import Segment, Dipole, Quadrupole, BPM, Drift, Cavity, Undulator

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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SCALE_FACTOR = 0.25
DEFAULT_ROTATION_ANGLE = math.pi
DEFAULT_ROTATION_AXIS = [0, 1, 0]


class MeshTransformer:
    """Helper class for 3D mesh transformations."""

    def __init__(self, scale_factor: float, rotation_angle: float, rotation_axis: List[float]):
        self.scale_factor = scale_factor
        self.rotation_angle = rotation_angle
        self.rotation_axis = rotation_axis

    def transform_mesh(self, mesh: trimesh.Trimesh, translation_vector: List[float]) -> None:
        """Apply transformations to mesh."""
        rotation_matrix = trimesh.transformations.rotation_matrix(
            self.rotation_angle, self.rotation_axis
        )
        mesh.apply_scale(self.scale_factor)
        mesh.apply_transform(rotation_matrix)
        mesh.apply_translation(translation_vector)


class Segment3DBuilder:
    """
    Builds 3D representations of accelerator lattice segments.

    Creates and manages 3D models of accelerator components including:
    - Dipole magnets
    - Quadrupole magnets
    - Beam Position Monitors (BPMs)
    - RF Cavities
    - Undulators

    The builder places components in 3D space according to their sequence
    and parameters defined in the configuration.
    """
    def __init__(self, segment: Segment):
        """
        Initialize the 3D segment builder.
        
        Args:
            segment: Lattice element definitions
        """
        # Segment object containing lattice elements (e.g., Dipole, Quadrupole)
        self.segment = segment

        # Initialize transformer
        config = {
            "scale_factor": DEFAULT_SCALE_FACTOR,
            "rotation_angle": DEFAULT_ROTATION_ANGLE,
            "rotation_axis": DEFAULT_ROTATION_AXIS
        }

        self.transformer = MeshTransformer(
            scale_factor=config["scale_factor"],
            rotation_angle=config["rotation_angle"],
            rotation_axis=config["rotation_axis"]
        )

        # Track current longitudinal position along the segment
        self.current_position = 0.0

        # Determine the base directory for assets (or for storing outputs)
        self.assets_dir = os.path.dirname(cheetah._assets.__file__)

        # Creates a visualization scene using triangular meshes with an automatically generated camera and lighting
        self.scene = trimesh.Scene()

        # Track lattice component positions
        self.component_positions = {}

    @property
    def current_position(self) -> float:
        """Current longitudinal position along the segment."""
        return self._current_position

    @current_position.setter
    def current_position(self, value: float) -> None:
        """Set current longitudinal position."""
        self._current_position = value

    def _load_and_transform_mesh(self, asset_key: str, translation_vector: List[float]) -> None:
        """
        Load and transform 3D model mesh and adds it to the scene
        
        Args:
            filename: Path to GLB/GLTF file (3D model)
            translation_vector: [x, y, z] translation coordinates to place the model in the correct position
        """
        # Map the asset key to the corresponding resource file name.
        asset_map = {
            "dipole_vcor": "ares_ea_small_steerer_vcor.glb",
            "dipole_hcor": "ares_ea_small_steerer_hcor.glb",
            "quadrupole": "ares_ea_quadrupole.glb",
            "monitor": "ares_ea_screen_station.glb",
            "cavity": "ares_ea_cavity.glb",
            "undulator": "ares_ea_undulator.glb"
        }

        filename = asset_map.get(asset_key)
        if filename is None:
            raise ValueError(f"Asset for key '{asset_key}' not found.")

        try:
            # Use importlib.resources to access the asset file.
            with pkg_resources.path(_assets, filename) as asset_path:
                # Force loading 3D model as a scene to ensure multiple geometries are handled properly.
                scene = trimesh.load(str(asset_path), file_type="glb", force="scene")  # try to coerce everything into a scene instead of a single mesh

                for mesh in scene.geometry.values():
                    self.transformer.transform_mesh(mesh, translation_vector)
                    self.scene.add_geometry(mesh)

        except Exception as e:
            logger.error(f"Failed to load mesh for asset key {asset_key}: {e}")
            raise

    def _add_element(self, element: Union[Dipole, Quadrupole, BPM, Undulator, Cavity],
                    component_key: str) -> None:
        """Add element to scene."""
        translation_vector = [0, 0, self.current_position]

        self._load_and_transform_mesh(component_key, translation_vector)
        self._track_component_position(element.name)

        logger.info(f"Added {element.__class__.__name__}: {element.name} at position {self.current_position}")

    def add_dipole(self, element: Dipole) -> None:
        """Add dipole magnet to scene."""
        component_key = "dipole_vcor" if "vcor" in element.name else "dipole_hcor"
        self._add_element(element, component_key)

    def add_quadrupole(self, element: Quadrupole) -> None:
        """Add quadrupole magnet to scene."""
        self._add_element(element, "quadrupole")

    def add_monitor(self, element: BPM) -> None:
        """Add beam position monitor to scene."""
        self._add_element(element, "monitor")

    def add_undulator(self, element: Undulator) -> None:
        """Add undulator to scene."""
        self._add_element(element, "undulator")

    def add_cavity(self, element: Cavity) -> None:
        """Add RF cavity to scene."""
        self._add_element(element, "cavity")

    def _track_component_position(self, component_name: str) -> None:
        """Store the component position dynamically."""
        self.component_positions[component_name] = self.current_position

    def build_segment(self, output_filename: Optional[str] = None) -> None:
        """
        Build complete 3D segment scene. Iterates through the segment and adds each element (Dipole/Quadrupole) to the 3D scene.
        Other elements (like Drift or BPM) only advance the current position.
        
        Args:
            output_filename: Output GLB/GLTF filename for the exported 3D scene
        """
        # For Drift, or other elements, we do not add geometry.
        for element in self.segment.elements:
            length = element.length.item() if isinstance(element.length, torch.Tensor) else float(element.length)

            element_handlers = {
                Dipole: self.add_dipole,
                Quadrupole: self.add_quadrupole,
                BPM: self.add_monitor,
                Undulator: self.add_undulator,
                Cavity: self.add_cavity
            }

            handler = element_handlers.get(type(element))
            if handler:
                handler(element)
            
            self.current_position += length

        # Represents the total length after processing last element
        logger.info(f"Final lattice segment length: {self.current_position}")

        # Export the final scene to a GLTF file
        self.export(output_filename)

    def export(self, output_file: Optional[str] = None) -> None:
        """
        Export scene to file.
        
        Args:
            output_file: Output filename
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
            logger.info(f"Exported 3D scene to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export scene: {e}")
            raise

    def __repr__(self) -> str:
        """String representation."""
        return f"Segment3DBuilder(elements={len(self.segment.elements)}, position={self.current_position})"


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build and export a 3D accelerator lattice scene."
    )
    parser.add_argument(
        "-c", "--config", default="config.json", help="Path to the configuration file (default: config.json)"
    )
    parser.add_argument(
        "-o", "--output-scene", default=None, help="Output GLB filename (default: None)"
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
