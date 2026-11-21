from collections.abc import Callable, Mapping
from keyword import iskeyword
from types import MappingProxyType

import torch
from torch import nn


OverlayRule = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]


class Overlay(nn.Module):
    """Fan out named variables to element attributes using callable rules.

    ``targets`` maps ``(element_name, attribute_name)`` pairs to pure functions
    of the variable mapping. All elements with a matching name are updated,
    including elements in nested segments. Targets must be tensor attributes,
    not registered parameters; the overlay variables control their values.

    Variables are registered as buffers on the overlay module for proper PyTorch
    integration. Write to variables using dot notation (e.g., ``overlay.var1 = 2.0``)
    or use ``set`` to update several together. Rules are evaluated before any target
    is written. Results use the target's dtype and device without detaching gradients.
    Reassign variables rather than mutating their tensors in place, which does not
    trigger rule evaluation.

    Overlays are nestable: one overlay can reference variables from other overlays
    by including them in the variable mapping during rule evaluation.

    Example::

        overlay = Overlay(
            name="length_control",
            variables={"base_length": 2.0},
            targets={
                ("drift1", "length"): lambda vars: vars["base_length"],
                ("drift2", "length"): lambda vars: vars["base_length"] * 1.5,
            }
        )
        segment = Segment([drift1, drift2], overlays=[overlay])
        segment.length_control.base_length = 3.0  # Update via dot notation
    """

    def __init__(
        self,
        name: str,
        *,
        variables: Mapping[str, int | float | torch.Tensor],
        targets: Mapping[tuple[str, str], OverlayRule],
    ) -> None:
        super().__init__()

        if not name:
            raise ValueError("An overlay must have a nonempty name.")
        if not variables or not targets:
            raise ValueError("An overlay requires variables and targets.")

        # Validate variables and register them as buffers
        for variable, value in variables.items():
            if (
                not variable.isidentifier()
                or iskeyword(variable)
                or variable.startswith("_")
                or hasattr(type(self), variable)
            ):
                raise ValueError(f"Invalid or reserved overlay variable: {variable!r}")
            validated_value = self._validate_value(variable, value)
            # Register as buffer (not parameter) so it's not trained
            tensor_value = (
                torch.as_tensor(validated_value)
                if not isinstance(validated_value, torch.Tensor)
                else validated_value
            )
            self.register_buffer(variable, tensor_value)

        # Validate targets
        for target, rule in targets.items():
            if (
                not isinstance(target, tuple)
                or len(target) != 2
                or not all(isinstance(part, str) and part for part in target)
            ):
                raise ValueError(
                    f"Expected (element, attribute) target, got {target!r}"
                )
            if not callable(rule):
                raise TypeError(f"Overlay rule for {target!r} must be callable.")

        self._name = name
        self._rules = dict(targets)
        self._targets: dict[tuple[str, str], list] = {}
        self._initial_variables: dict[str, torch.Tensor] | None = None
        self._segment: "Segment | None" = None  # noqa: F821

    @property
    def name(self) -> str:
        return self._name

    @property
    def variables(self) -> Mapping[str, torch.Tensor]:
        """Return a read-only view of all registered variables."""
        var_dict = {}
        for name, buffer in self.named_buffers():
            if not name.startswith("_"):
                var_dict[name] = buffer
        return MappingProxyType(var_dict)

    def __setattr__(self, name: str, value: object) -> None:
        """Support dot notation for setting variables (e.g., overlay.var1 = 2.0)."""
        # Let nn.Module handle its own attributes (starting with _)
        if name.startswith("_") or name in {"_parameters", "_buffers", "_modules"}:
            super().__setattr__(name, value)
        # Check if this is an existing buffer (variable)
        elif hasattr(self, "_buffers") and name in self._buffers:
            self.set(**{name: self._validate_value(name, value)})
        # Check if it's an existing nn.Module attribute or method
        elif hasattr(type(self), name):
            super().__setattr__(name, value)
        # Otherwise, try to set it as a variable
        else:
            # If buffers aren't initialized yet, use super
            if not hasattr(self, "_buffers"):
                super().__setattr__(name, value)
            else:
                # This is a new variable assignment - not allowed after init
                raise AttributeError(
                    f"Cannot add new variable '{name}' after overlay creation. "
                    "Define all variables in the overlay constructor."
                )

    @staticmethod
    def _validate_value(name: str, value: object) -> int | float | torch.Tensor:
        if not isinstance(value, (int, float, torch.Tensor)):
            raise TypeError(f"Overlay variable {name!r} must be a number or tensor.")
        return value

    def set(self, **values: int | float | torch.Tensor) -> None:
        """Update variables together and immediately recompute all targets."""
        for name, value in values.items():
            if name not in self._buffers:
                raise AttributeError(f"Overlay {self.name!r} has no variable {name!r}.")
            validated_value = self._validate_value(name, value)
            # Convert to tensor and register as buffer
            tensor_value = (
                torch.as_tensor(validated_value)
                if not isinstance(validated_value, torch.Tensor)
                else validated_value.clone()
            )
            self.register_buffer(name, tensor_value)

        self._update_targets(self.variables, self._targets)

    def _build_combined_variables(
        self, variables: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Build combined variables including nested overlay variables.

        This enables nestable overlays: the combined mapping includes variables from
        this overlay plus all other overlays in the segment, allowing rules to
        reference variables from other overlays.
        """
        combined = dict(variables)
        if self._segment is not None:
            for overlay in self._segment.overlays:
                if overlay is not self:
                    for var_name in overlay._buffers:
                        if not var_name.startswith("_"):
                            overlay_var_name = f"{overlay.name}_{var_name}"
                            combined[overlay_var_name] = getattr(overlay, var_name)
        return combined

    def _update_targets(
        self,
        variables: Mapping[str, torch.Tensor],
        targets: Mapping[tuple[str, str], list],
    ) -> None:
        # Build combined variables including other overlay variables
        combined_variables = self._build_combined_variables(variables)

        updates = []
        for target, elements in targets.items():
            value = self._rules[target](MappingProxyType(combined_variables))
            self._validate_value(str(target), value)
            attribute = target[1]
            for element in elements:
                current = getattr(element, attribute)
                result = torch.as_tensor(
                    value, dtype=current.dtype, device=current.device
                ).clone()
                updates.append((element, attribute, result))
        for element, attribute, result in updates:
            setattr(element, attribute, result)

    def _bind(self, segment: "Segment", *, reset: bool = False) -> None:  # noqa: F821
        if self._initial_variables is not None and not reset:
            raise ValueError(f"Overlay {self.name!r} is already bound to a simulator.")
        elements = segment.flattened().elements
        targets = {}
        for element_name, attribute in self._rules:
            matches = [element for element in elements if element.name == element_name]
            if not matches:
                raise ValueError(f"Overlay target element {element_name!r} not found.")
            for element in matches:
                if not isinstance(getattr(element, attribute, None), torch.Tensor):
                    raise ValueError(
                        f"Overlay target {element_name}[{attribute}] "
                        "must be an existing tensor attribute."
                    )
                if attribute in dict(element.named_parameters(recurse=False)):
                    raise ValueError(
                        f"Overlay target {element_name}[{attribute}] is a registered "
                        "parameter; use a tensor buffer controlled by the overlay."
                    )
            targets[element_name, attribute] = matches

        if reset:
            if self._initial_variables is None:
                raise RuntimeError("Cannot reset an unbound overlay.")
            variables = self._copy_values(self._initial_variables)
            for var_name, var_value in variables.items():
                self.register_buffer(var_name, var_value)
        else:
            variables = dict(self.variables)
        self._segment = segment
        self._update_targets(variables, targets)
        if not reset:
            self._initial_variables = self._copy_values(variables)
        self._targets = targets

    @staticmethod
    def _copy_values(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: (
                value.detach().clone().requires_grad_(value.requires_grad)
                if isinstance(value, torch.Tensor)
                else value
            )
            for name, value in values.items()
        }
