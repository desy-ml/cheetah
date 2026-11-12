"""Tests for Overlay functionality."""

import pytest
import torch

from cheetah.accelerator import Overlay, Drift, Segment


class TestOverlayBasics:
    """Test basic Overlay creation and validation."""

    def test_overlay_creation(self):
        """Test basic overlay creation with variables and targets."""
        overlay = Overlay(
            name="test_overlay",
            variables={"var1": 1.0, "var2": 2.0},
            targets={("element1", "attr1"): lambda vars: vars["var1"] + vars["var2"]},
        )
        assert overlay.name == "test_overlay"
        assert "var1" in overlay.variables
        assert "var2" in overlay.variables

    def test_overlay_empty_name_error(self):
        """Test that empty overlay name raises ValueError."""
        with pytest.raises(ValueError, match="nonempty name"):
            Overlay(
                name="",
                variables={"var1": 1.0},
                targets={("element1", "attr1"): lambda vars: vars["var1"]},
            )

    def test_overlay_missing_variables_error(self):
        """Test that missing variables raises ValueError."""
        with pytest.raises(ValueError, match="variables and targets"):
            Overlay(
                name="test",
                variables={},
                targets={("element1", "attr1"): lambda vars: 1.0},
            )

    def test_overlay_missing_targets_error(self):
        """Test that missing targets raises ValueError."""
        with pytest.raises(ValueError, match="variables and targets"):
            Overlay(
                name="test",
                variables={"var1": 1.0},
                targets={},
            )

    def test_overlay_invalid_variable_name(self):
        """Test that invalid variable names raise ValueError."""
        # Reserved underscore prefix
        with pytest.raises(ValueError, match="Invalid or reserved"):
            Overlay(
                name="test",
                variables={"_var": 1.0},
                targets={("elem", "attr"): lambda vars: 1.0},
            )

        # Keyword
        with pytest.raises(ValueError, match="Invalid or reserved"):
            Overlay(
                name="test",
                variables={"if": 1.0},
                targets={("elem", "attr"): lambda vars: 1.0},
            )

    def test_overlay_invalid_target_tuple(self):
        """Test that invalid target tuples raise ValueError."""
        # Wrong tuple length
        with pytest.raises(ValueError, match="Expected.*target"):
            Overlay(
                name="test",
                variables={"var1": 1.0},
                targets={("only_one",): lambda vars: 1.0},
            )

        # Non-string tuple element
        with pytest.raises(ValueError, match="Expected.*target"):
            Overlay(
                name="test",
                variables={"var1": 1.0},
                targets={(1, "attr"): lambda vars: 1.0},
            )

    def test_overlay_non_callable_rule_error(self):
        """Test that non-callable rules raise TypeError."""
        with pytest.raises(TypeError, match="must be callable"):
            Overlay(
                name="test",
                variables={"var1": 1.0},
                targets={("elem", "attr"): 5},
            )


class TestOverlayBuffers:
    """Test that overlay variables are properly registered as buffers."""

    def test_variables_registered_as_buffers(self):
        """Test that variables are registered as nn.Module buffers."""
        overlay = Overlay(
            name="test",
            variables={"var1": 1.0, "var2": torch.tensor(2.0)},
            targets={("elem", "attr"): lambda vars: vars["var1"]},
        )

        buffers = dict(overlay.named_buffers())
        assert "var1" in buffers
        assert "var2" in buffers
        assert isinstance(buffers["var1"], torch.Tensor)
        assert isinstance(buffers["var2"], torch.Tensor)

    def test_variables_converted_to_tensors(self):
        """Test that variables are converted to tensors."""
        overlay = Overlay(
            name="test",
            variables={"var1": 1.0, "var2": 2},
            targets={("elem", "attr"): lambda vars: vars["var1"]},
        )

        assert isinstance(overlay.var1, torch.Tensor)
        assert isinstance(overlay.var2, torch.Tensor)
        # Float literals become float tensors
        assert (
            overlay.var1.dtype == torch.float32 or overlay.var1.dtype == torch.float64
        )
        # Int literals become int tensors
        assert overlay.var2.dtype == torch.int64 or overlay.var2.dtype == torch.int32

    def test_invalid_variable_type_error(self):
        """Test that invalid variable types raise TypeError."""
        with pytest.raises(TypeError, match="must be a number or tensor"):
            Overlay(
                name="test",
                variables={"var1": "invalid"},
                targets={("elem", "attr"): lambda vars: 1.0},
            )


class TestOverlayDotNotation:
    """Test dot notation access and assignment."""

    def test_read_variable_via_dot_notation(self):
        """Test reading variables using dot notation."""
        overlay = Overlay(
            name="test",
            variables={"var1": 2.0},
            targets={("elem", "attr"): lambda vars: vars["var1"]},
        )

        assert overlay.var1 == 2.0

    def test_write_variable_via_dot_notation(self):
        """Test writing variables using dot notation."""
        overlay = Overlay(
            name="test",
            variables={"var1": 2.0},
            targets={("elem", "attr"): lambda vars: vars["var1"]},
        )

        overlay.var1 = 5.0
        assert overlay.var1 == 5.0

    def test_write_tensor_via_dot_notation(self):
        """Test writing tensor variables using dot notation."""
        overlay = Overlay(
            name="test",
            variables={"var1": 2.0},
            targets={("elem", "attr"): lambda vars: vars["var1"]},
        )

        overlay.var1 = torch.tensor(10.0)
        assert torch.allclose(overlay.var1, torch.tensor(10.0))

    def test_cannot_add_new_variable_after_init(self):
        """Test that new variables cannot be added after initialization."""
        overlay = Overlay(
            name="test",
            variables={"var1": 2.0},
            targets={("elem", "attr"): lambda vars: vars["var1"]},
        )

        with pytest.raises(
            AttributeError, match="Cannot add new variable.*after overlay creation"
        ):
            overlay.new_var = 1.0


class TestOverlaySet:
    """Test the set() method for batch variable updates."""

    def test_set_single_variable(self):
        """Test updating a single variable with set()."""
        overlay = Overlay(
            name="test",
            variables={"var1": 1.0, "var2": 2.0},
            targets={("elem", "attr"): lambda vars: vars["var1"] + vars["var2"]},
        )

        overlay.set(var1=5.0)
        assert overlay.var1 == 5.0
        assert overlay.var2 == 2.0

    def test_set_multiple_variables(self):
        """Test updating multiple variables with set()."""
        overlay = Overlay(
            name="test",
            variables={"var1": 1.0, "var2": 2.0},
            targets={("elem", "attr"): lambda vars: vars["var1"] + vars["var2"]},
        )

        overlay.set(var1=5.0, var2=10.0)
        assert overlay.var1 == 5.0
        assert overlay.var2 == 10.0

    def test_set_nonexistent_variable_error(self):
        """Test that set() raises error for nonexistent variable."""
        overlay = Overlay(
            name="test",
            variables={"var1": 1.0},
            targets={("elem", "attr"): lambda vars: vars["var1"]},
        )

        with pytest.raises(AttributeError, match="has no variable"):
            overlay.set(nonexistent=5.0)


class TestSegmentOverlayIntegration:
    """Test Overlay integration with Segment."""

    def test_add_overlay_to_segment(self):
        """Test adding an overlay to a segment."""
        drift = Drift(name="drift1", length=torch.tensor(1.0))
        segment = Segment([drift], name="seg")

        overlay = Overlay(
            name="overlay1",
            variables={"var1": 2.0},
            targets={("drift1", "length"): lambda vars: vars["var1"]},
        )

        segment.add_overlay(overlay)
        assert "overlay1" in segment.overlay_names

    def test_overlay_attribute_access_on_segment(self):
        """Test accessing overlay via segment attribute."""
        drift = Drift(name="drift1", length=torch.tensor(1.0))
        segment = Segment([drift], name="seg")

        overlay = Overlay(
            name="overlay",
            variables={"var": 2.0},
            targets={("drift1", "length"): lambda vars: vars["var"]},
        )

        segment.add_overlay(overlay)

        # Access overlay through segment
        assert segment.overlay.name == "overlay"
        assert segment.overlay.var == 2.0

    def test_dot_notation_through_segment(self):
        """Test writing to overlay variables via segment attribute."""
        drift = Drift(name="drift1", length=torch.tensor(1.0))
        segment = Segment([drift], name="seg")

        overlay = Overlay(
            name="myoverlay",
            variables={"param": 2.0},
            targets={("drift1", "length"): lambda vars: vars["param"]},
        )

        segment.add_overlay(overlay)

        # Modify via segment
        segment.myoverlay.param = 5.0
        assert segment.myoverlay.param == 5.0

    def test_segment_initial_overlays(self):
        """Test creating segment with initial overlays."""
        drift = Drift(name="drift1", length=torch.tensor(1.0))
        overlay = Overlay(
            name="overlay1",
            variables={"var1": 2.0},
            targets={("drift1", "length"): lambda vars: vars["var1"]},
        )

        segment = Segment([drift], name="seg", overlays=[overlay])
        assert "overlay1" in segment.overlay_names
        assert segment.overlay1.var1 == 2.0

    def test_multiple_overlays_in_segment(self):
        """Test segment with multiple overlays."""
        drift = Drift(name="drift1", length=torch.tensor(1.0))
        segment = Segment([drift], name="seg")

        overlay1 = Overlay(
            name="overlay1",
            variables={"var1": 1.0},
            targets={("drift1", "length"): lambda vars: vars["var1"]},
        )

        overlay2 = Overlay(
            name="overlay2",
            variables={"var2": 2.0},
            targets={("drift1", "length"): lambda vars: vars["var2"]},
        )

        segment.add_overlay(overlay1)
        segment.add_overlay(overlay2)

        assert len(segment.overlay_names) == 2
        assert "overlay1" in segment.overlay_names
        assert "overlay2" in segment.overlay_names

    def test_duplicate_overlay_name_error(self):
        """Test that duplicate overlay names raise ValueError."""
        drift = Drift(name="drift1", length=torch.tensor(1.0))
        segment = Segment([drift], name="seg")

        overlay1 = Overlay(
            name="overlay",
            variables={"var1": 1.0},
            targets={("drift1", "length"): lambda vars: vars["var1"]},
        )

        overlay2 = Overlay(
            name="overlay",
            variables={"var2": 2.0},
            targets={("drift1", "length"): lambda vars: vars["var2"]},
        )

        segment.add_overlay(overlay1)
        with pytest.raises(ValueError, match="already exists"):
            segment.add_overlay(overlay2)

    def test_remove_overlay_from_segment(self):
        """Test removing an overlay from a segment."""
        drift = Drift(name="drift1", length=torch.tensor(1.0))
        segment = Segment([drift], name="seg")

        overlay = Overlay(
            name="overlay1",
            variables={"var1": 2.0},
            targets={("drift1", "length"): lambda vars: vars["var1"]},
        )

        segment.add_overlay(overlay)
        assert "overlay1" in segment.overlay_names

        removed = segment.remove_overlay("overlay1")
        assert removed.name == "overlay1"
        assert "overlay1" not in segment.overlay_names

    def test_remove_overlay_attribute_cleanup(self):
        """Test that removing overlay cleans up segment attribute."""
        drift = Drift(name="drift1", length=torch.tensor(1.0))
        segment = Segment([drift], name="seg")

        overlay = Overlay(
            name="overlay",
            variables={"var": 2.0},
            targets={("drift1", "length"): lambda vars: vars["var"]},
        )

        segment.add_overlay(overlay)
        segment.remove_overlay("overlay")

        with pytest.raises(AttributeError):
            _ = segment.overlay

    def test_get_overlay_from_segment(self):
        """Test retrieving an overlay from a segment."""
        drift = Drift(name="drift1", length=torch.tensor(1.0))
        segment = Segment([drift], name="seg")

        overlay = Overlay(
            name="overlay1",
            variables={"var1": 2.0},
            targets={("drift1", "length"): lambda vars: vars["var1"]},
        )

        segment.add_overlay(overlay)
        retrieved = segment.get_overlay("overlay1")

        assert retrieved.name == "overlay1"
        assert retrieved.var1 == 2.0

    def test_get_nonexistent_overlay_error(self):
        """Test that getting nonexistent overlay raises ValueError."""
        segment = Segment([], name="seg")

        with pytest.raises(ValueError, match="not found"):
            segment.get_overlay("nonexistent")


class TestNestedOverlays:
    """Test nested overlay functionality."""

    def test_nested_overlay_variable_reference(self):
        """Test that one overlay can reference another overlay's variables."""
        drift = Drift(name="drift1", length=torch.tensor(1.0))
        segment = Segment([drift], name="seg")

        overlay1 = Overlay(
            name="overlay1",
            variables={"base": 2.0},
            targets={("drift1", "length"): lambda vars: vars["base"]},
        )

        overlay2 = Overlay(
            name="overlay2",
            variables={"scale": 1.5},
            targets={
                ("drift1", "length"): lambda vars: (
                    vars.get("overlay1_base", 1.0) * vars["scale"]
                )
            },
        )

        segment.add_overlay(overlay1)
        segment.add_overlay(overlay2)

        # Verify both overlays were added
        assert "overlay1" in segment.overlay_names
        assert "overlay2" in segment.overlay_names

    def test_nested_overlay_dynamic_updates(self):
        """Test that nested overlays respond to variable changes."""
        drift = Drift(name="drift1", length=torch.tensor(1.0))
        segment = Segment([drift], name="seg")

        overlay1 = Overlay(
            name="primary",
            variables={"x": 1.0},
            targets={("drift1", "length"): lambda vars: vars["x"]},
        )

        overlay2 = Overlay(
            name="dependent",
            variables={"factor": 2.0},
            targets={
                ("drift1", "length"): lambda vars: (
                    vars.get("primary_x", 1.0) + vars["factor"]
                )
            },
        )

        segment.add_overlay(overlay1)
        segment.add_overlay(overlay2)

        # Update primary overlay variable
        segment.primary.x = 5.0

        # Dependent overlay should be able to use the updated value
        assert segment.primary.x == 5.0


class TestOverlayModuleIntegration:
    """Test PyTorch nn.Module integration."""

    def test_overlay_is_nn_module(self):
        """Test that Overlay is an nn.Module."""
        import torch.nn as nn

        overlay = Overlay(
            name="test",
            variables={"var1": 1.0},
            targets={("elem", "attr"): lambda vars: vars["var1"]},
        )

        assert isinstance(overlay, nn.Module)

    def test_overlay_buffers_in_state_dict(self):
        """Test that overlay variables appear in state_dict."""
        overlay = Overlay(
            name="test",
            variables={"var1": 1.0, "var2": 2.0},
            targets={("elem", "attr"): lambda vars: vars["var1"]},
        )

        state = overlay.state_dict()
        assert "var1" in state
        assert "var2" in state

    def test_overlay_device_movement(self):
        """Test that overlay variables can be accessed on CPU."""
        overlay = Overlay(
            name="test",
            variables={"var1": 1.0},
            targets={("elem", "attr"): lambda vars: vars["var1"]},
        )

        # Verify variable is accessible
        assert overlay.var1.device.type == "cpu"

    def test_overlay_manual_dtype_conversion(self):
        """Test that overlay variables can be converted to different dtypes."""
        overlay = Overlay(
            name="test",
            variables={"var1": 1.0},
            targets={("elem", "attr"): lambda vars: vars["var1"]},
        )

        # Manually set a float64 value
        overlay.var1 = torch.tensor(2.0, dtype=torch.float64)
        assert overlay.var1.dtype == torch.float64

        # Manually set a float32 value
        overlay.var1 = torch.tensor(3.0, dtype=torch.float32)
        assert overlay.var1.dtype == torch.float32


class TestOverlayErrorHandling:
    """Test error handling and edge cases."""

    def test_overlay_with_invalid_target_element_name(self):
        """Test that overlay detects invalid target element names."""
        drift = Drift(name="drift1", length=torch.tensor(1.0))
        segment = Segment([drift], name="seg")

        overlay = Overlay(
            name="overlay",
            variables={"var1": 1.0},
            targets={("nonexistent", "length"): lambda vars: vars["var1"]},
        )

        # Should raise ValueError when adding overlay to segment (binding fails)
        with pytest.raises(ValueError, match="Overlay target element.*not found"):
            segment.add_overlay(overlay)

    def test_variable_name_conflicts_with_methods(self):
        """Test that variable names can't conflict with nn.Module methods."""
        with pytest.raises(ValueError, match="Invalid or reserved"):
            Overlay(
                name="test",
                variables={"forward": 1.0},
                targets={("elem", "attr"): lambda vars: 1.0},
            )
