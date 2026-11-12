import warnings
from pathlib import Path

import pytest
import torch

import cheetah
from cheetah.utils import DirtyNameWarning

from .resources import ARESlatticeStage3v1_9 as ares


@pytest.mark.filterwarnings("ignore::cheetah.utils.DefaultParameterWarning")
def test_save_and_reload_ares_example(tmp_path):
    """
    Test that saving Cheetah `Segment` to LatticeJSON works and that it can be reloaded
    correctly at the example of the full ARES lattice.
    """
    original_segment = cheetah.Segment.from_ocelot(ares.cell, name="ARES_Segment")

    original_segment.to_lattice_json(
        str(tmp_path / "ares_lattice.json"),
        title="ARES LatticeJSON",
        info="Save and reload test for Cheetah using the ARES lattice",
    )

    reloaded_segment = cheetah.Segment.from_lattice_json(
        str(tmp_path / "ares_lattice.json")
    )

    assert original_segment.name == reloaded_segment.name
    assert len(original_segment.elements) == len(reloaded_segment.elements)
    assert original_segment.length == reloaded_segment.length

    # TODO: Improve element comparison when equality is working again
    for original_element, reloaded_element in zip(
        original_segment.elements, reloaded_segment.elements
    ):
        assert original_element.name == reloaded_element.name
        assert original_element.__class__ == reloaded_element.__class__


def test_save_and_reload_custom_transfer_map(tmp_path):
    """
    Test that saving and reloading a `CustomTransferMap` works. `CustomTransferMap`
    never appears in the ARES lattice and must therefore be tested separately.
    """
    custom_transfer_map_element = cheetah.CustomTransferMap(
        predefined_transfer_map=torch.eye(7, 7),
        length=torch.tensor(1.0),
        name="my_custom_transfer_map_element",
    )
    segment = cheetah.Segment([custom_transfer_map_element], name="test_segment")

    segment.to_lattice_json(
        str(tmp_path / "custom_transfer_map_lattice.json"),
        title="Custom Transfer Map LatticeJSON",
        info="Save and reload test for Cheetah using a custom transfer map",
    )

    reloaded_segment = cheetah.Segment.from_lattice_json(
        str(tmp_path / "custom_transfer_map_lattice.json")
    )

    # I really only care that the transfer map element is recovered correctly, the
    # segment was tested in a different test.
    reloaded_custom_transfer_map_element = reloaded_segment.elements[0]

    torch.testing.assert_close(
        custom_transfer_map_element.predefined_transfer_map,
        reloaded_custom_transfer_map_element.predefined_transfer_map,
    )
    torch.testing.assert_close(
        custom_transfer_map_element.length, reloaded_custom_transfer_map_element.length
    )
    assert custom_transfer_map_element.name == reloaded_custom_transfer_map_element.name


def test_save_and_reload_metadata(tmp_path):
    """
    Test that saving and reloading an element with metadata correctly preserves the
    metadata.
    """
    quadrupole = cheetah.Quadrupole(
        length=torch.tensor(0.3),
        k1=torch.tensor(4.2),
        name="my_quadrupole",
        metadata={"control_system": {"pv_base": "A:Q1:", "readbacks": ["MeasCurrent"]}},
    )
    segment = cheetah.Segment(elements=[quadrupole], name="test_segment")

    segment.to_lattice_json(str(tmp_path / "metadata_lattice.json"))

    reloaded_segment = cheetah.Segment.from_lattice_json(
        str(tmp_path / "metadata_lattice.json")
    )
    reloaded_quadrupole = reloaded_segment.my_quadrupole

    assert reloaded_quadrupole.metadata == quadrupole.metadata


@pytest.mark.parametrize(
    "desired_dtype", [torch.float32, torch.float64], ids=["float32", "float64"]
)
def test_desired_dtype(tmp_path, desired_dtype: torch.dtype):
    """
    Test that the lattice JSON import correctly interprets its optional dtype parameter.
    """
    original_segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.Dipole(length=torch.tensor(0.5), angle=torch.tensor(1e-3)),
            cheetah.Quadrupole(length=torch.tensor(0.3), k1=torch.tensor(31.415)),
        ]
    )
    original_segment.to_lattice_json(str(tmp_path / "dummy_lattice.json"))

    reloaded_segment = cheetah.Segment.from_lattice_json(
        str(tmp_path / "dummy_lattice.json"), dtype=desired_dtype
    )

    assert all(
        getattr(element, feature).dtype == desired_dtype
        for element in reloaded_segment.elements
        for feature in element.defining_tensors
    )


@pytest.mark.parametrize("sanitize_names", [None, True, False])
def test_lattice_json_sanitize_name_warning(tmp_path, sanitize_names):
    """
    Test that the `DirtyNameWarning` is suppressed by the `sanitize_name` argument.
    """
    original_segment = cheetah.Segment(
        elements=[
            cheetah.Drift(
                length=torch.tensor(1.0), name="dirty:drift#1", sanitize_name=False
            )
        ],
        name="dirty!segment",
        sanitize_name=False,
    )
    original_segment.to_lattice_json(tmp_path / "dirty_lattice.json")

    if sanitize_names is None:
        with pytest.warns(DirtyNameWarning):
            _ = cheetah.Segment.from_lattice_json(
                tmp_path / "dirty_lattice.json", sanitize_names=sanitize_names
            )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DirtyNameWarning)
            _ = cheetah.Segment.from_lattice_json(
                tmp_path / "dirty_lattice.json", sanitize_names=sanitize_names
            )


def test_save_and_reload_overlay_with_simple_rules(tmp_path):
    """Test that saving and reloading a segment with overlay works correctly."""
    # Define rules as lambda expressions
    rule1 = lambda vars: vars["base_length"]
    rule2 = lambda vars: vars["base_length"] * 1.5

    # Create elements
    drift1 = cheetah.Drift(length=torch.tensor(1.0), name="drift1")
    drift2 = cheetah.Drift(length=torch.tensor(2.0), name="drift2")

    # Create overlay
    overlay = cheetah.Overlay(
        name="length_control",
        variables={"base_length": 2.0},
        targets={
            ("drift1", "length"): rule1,
            ("drift2", "length"): rule2,
        },
    )

    # Create segment with overlay
    original_segment = cheetah.Segment(
        [drift1, drift2], name="test_segment", overlays=[overlay]
    )

    # Verify overlay works before saving
    assert original_segment.drift1.length == torch.tensor(2.0)
    assert original_segment.drift2.length == torch.tensor(3.0)

    # Save to JSON
    original_segment.to_lattice_json(str("overlay_lattice.json"))

    # Reload from JSON
    reloaded_segment = cheetah.Segment.from_lattice_json(
        str("overlay_lattice.json")
    )

    # Verify structure
    assert reloaded_segment.name == original_segment.name
    assert len(reloaded_segment.elements) == len(original_segment.elements)
    assert "length_control" in reloaded_segment.overlay_names

    # Verify overlay is functional
    assert reloaded_segment.drift1.length == torch.tensor(2.0)
    assert reloaded_segment.drift2.length == torch.tensor(3.0)

    # Test updating overlay variable
    reloaded_segment.length_control.base_length = 3.0
    assert torch.allclose(reloaded_segment.drift1.length, torch.tensor(3.0))
    assert torch.allclose(reloaded_segment.drift2.length, torch.tensor(4.5))


def test_overlay_json_format(tmp_path):
    """Verify the JSON format for overlays is correct."""
    drift = cheetah.Drift(length=torch.tensor(1.0), name="drift1")

    overlay = cheetah.Overlay(
        name="test_overlay",
        variables={"var1": 2.5},
        targets={("drift1", "length"): lambda vars: vars["var1"] * 2},
    )
    segment = cheetah.Segment([drift], name="seg", overlays=[overlay])
    filepath = str("format_check.json")
    segment.to_lattice_json(filepath)

    # Load JSON and verify structure
    import json

    with open(filepath) as f:
        data = json.load(f)

    # Verify overlays section exists
    assert "overlays" in data
    assert "test_overlay" in data["overlays"]

    overlay_data = data["overlays"]["test_overlay"]
    assert "variables" in overlay_data
    assert "targets" in overlay_data
    assert overlay_data["variables"]["var1"] == 2.5
    assert "drift1.length" in overlay_data["targets"]
    # Rule expression should be serialized as string
    assert isinstance(overlay_data["targets"]["drift1.length"], str)


def test_save_and_reload_nested_overlays(tmp_path):
    """Test segment with nested overlays that reference each other's variables."""
    # Define rules that reference each other's overlays
    rule1 = lambda vars: vars["base_value"]
    rule2 = lambda vars: vars["overlay1_base_value"] * 2

    # Create elements
    drift1 = cheetah.Drift(length=torch.tensor(1.0), name="drift1")
    drift2 = cheetah.Drift(length=torch.tensor(2.0), name="drift2")

    # First overlay sets a base value
    overlay1 = cheetah.Overlay(
        name="overlay1",
        variables={"base_value": 2.0},
        targets={("drift1", "length"): rule1},
    )

    # Second overlay references first overlay's variables
    overlay2 = cheetah.Overlay(
        name="overlay2",
        variables={"dummy": 1.0},
        targets={("drift2", "length"): rule2},
    )

    # Create segment with nested overlays
    original_segment = cheetah.Segment(
        [drift1, drift2], name="nested_segment", overlays=[overlay1, overlay2]
    )

    # Verify nested overlays work before saving
    assert torch.allclose(original_segment.drift1.length, torch.tensor(2.0))
    assert torch.allclose(original_segment.drift2.length, torch.tensor(4.0))

    # Save to JSON
    original_segment.to_lattice_json(str(tmp_path / "nested_overlay.json"))

    # Reload from JSON
    reloaded_segment = cheetah.Segment.from_lattice_json(
        str(tmp_path / "nested_overlay.json")
    )

    # Verify both overlays exist
    assert len(reloaded_segment.overlay_names) == 2
    assert "overlay1" in reloaded_segment.overlay_names
    assert "overlay2" in reloaded_segment.overlay_names

    # Verify nested overlays still work after reload
    assert torch.allclose(reloaded_segment.drift1.length, torch.tensor(2.0))
    assert torch.allclose(reloaded_segment.drift2.length, torch.tensor(4.0))

    # Test updating overlay1's variable updates its own target
    reloaded_segment.overlay1.base_value = 3.0
    assert torch.allclose(reloaded_segment.drift1.length, torch.tensor(3.0))
    # Note: overlay2 won't automatically update since it's a separate overlay
    # The nested reference is evaluated when overlay2's variables change, not when overlay1 changes

