from __future__ import annotations

import warnings
from copy import deepcopy
from math import degrees, inf, isclose, isinf, isnan, radians
from pathlib import Path
from typing import Any

import torch

import cheetah
from cheetah.utils import UnknownElementWarning


def _import_pals():
    try:
        import pals
    except ImportError as exc:
        raise ImportError(
            "To use the PALS converter, install the PALS Python package "
            "`pals_schema`."
        ) from exc

    return pals


def _factory_kwargs(
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict:
    """Factory kwargs for converted tensors.

    :param device: The device to create converted tensors on. If `None`, uses
        the default PyTorch device.
    :param dtype: The dtype to create converted tensors with. If `None`, uses
        the default PyTorch dtype.
    :return: A dictionary of keyword arguments to pass to tensor constructors
        when converting element parameters.
    """
    default_device = torch.get_default_device()
    return {
        "device": torch.device(device) if device is not None else default_device,
        "dtype": dtype if dtype is not None else torch.get_default_dtype(),
    }


def _t2f(value: torch.Tensor) -> float:
    """Convert a scalar tensor to a float.
    :param value: A scalar tensor to convert.
    :return: The float value of the tensor.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(value).__name__}.")
    if value.numel() != 1:
        raise ValueError(
            "PALS conversion only supports scalar element parameters. "
            "Slice batched tensors before converting."
        )
    return float(value.detach().cpu().item())


def _f2t(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(value, device=device, dtype=dtype)


def _nonnegative_length(value: torch.Tensor, element_name: str) -> float:
    """Currently PALS requires thick elements with non-negative length."""
    length = _t2f(value)
    if length < 0:
        raise ValueError(
            f"Element {element_name!r} has negative length {length}; PALS lengths "
            "must be non-negative."
        )
    return length


def _is_zero(value: torch.Tensor) -> bool:
    return torch.allclose(value, torch.zeros_like(value))


def _require_default(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(message)


def _pals_payload(element) -> dict:
    dumped = element.model_dump()
    if hasattr(element, "name") and isinstance(dumped, dict) and element.name in dumped:
        return deepcopy(dumped[element.name])
    return deepcopy(dumped)


def _model_dump_or_empty(model) -> dict:
    return deepcopy(model.model_dump()) if model is not None else {}


def _attach_pals_extras(element: cheetah.Element, extras: dict) -> cheetah.Element:
    if extras:
        element.pals_extras = extras
    return element


def _split_pals_extras(element: cheetah.Element) -> tuple[dict, dict]:
    extras = deepcopy(getattr(element, "pals_extras", {}))
    if not isinstance(extras, dict):
        raise TypeError(f"{element.name}.pals_extras must be a dictionary.")
    return extras.pop("_cheetah", {}), extras


def _extract_extras(
    element,
    consumed_fields: set[str] | None = None,
    consumed_groups: dict[str, set[str] | None] | None = None,
    converter_metadata: dict | None = None,
) -> dict:
    """Extract extras from a PALS element during conversion to a Cheetah element."""
    consumed_fields = consumed_fields or set()
    consumed_groups = consumed_groups or {}
    payload = _pals_payload(element)

    payload.pop("kind", None)
    payload.pop("name", None)
    for field in consumed_fields:
        payload.pop(field, None)

    extras = {}
    for key, value in payload.items():
        if key not in consumed_groups:
            extras[key] = value
            continue

        consumed = consumed_groups[key]
        if consumed is None:
            continue

        remaining = deepcopy(value)
        for consumed_key in consumed:
            remaining.pop(consumed_key, None)
        if remaining:
            extras[key] = remaining

    if converter_metadata:
        extras["_cheetah"] = converter_metadata

    return extras


def _merge_group(pals, group_name: str, base: dict, extras: dict) -> Any:
    group_data = deepcopy(extras.pop(group_name, {}))
    group_data.update(base)
    if not group_data:
        return None

    group_class = getattr(pals, group_name.removesuffix("P") + "Parameters", None)
    if group_class is None:
        return group_data
    return group_class(**group_data)


def _common_kwargs_from_extras(pals_cls, extras: dict) -> dict:
    valid_fields = set(pals_cls.model_fields)
    kwargs = {}
    for key in list(extras):
        if key in valid_fields:
            kwargs[key] = extras.pop(key)
    if extras:
        raise NotImplementedError(
            f"Cannot re-emit unsupported PALS extras: {sorted(extras)}."
        )
    return kwargs


def _multipole_value(multipole: dict, order: int) -> tuple[float, str | None]:
    for key in (f"Kn{order}", f"Bn{order}"):
        if key in multipole:
            return multipole[key], key
    return 0.0, None


def _tilt_value(multipole: dict, order: int) -> float:
    return multipole.get(f"tilt{order}", 0.0)


def _aperture_limits_from_extent(
    value: torch.Tensor, element_name: str, plane: str
) -> list[float | None]:
    extent = _t2f(value)
    if isnan(extent) or extent < 0:
        raise ValueError(
            f"Aperture {element_name!r} has invalid {plane}_max={extent}; "
            "PALS conversion requires a non-negative aperture half width."
        )
    if isinf(extent):
        return [None, None]
    return [-extent, extent]


def _extent_from_aperture_limits(
    limits: list[float | None], element_name: str, plane: str
) -> float:
    if limits == [None, None]:
        return inf

    lower, upper = limits
    if lower is None or upper is None:
        raise NotImplementedError(
            f"PALS aperture {element_name!r} has one-sided {plane}_limits; "
            "Cheetah Aperture requires symmetric limits."
        )
    if not isclose(lower, -upper):
        raise NotImplementedError(
            f"PALS aperture {element_name!r} has asymmetric {plane}_limits; "
            "Cheetah Aperture requires symmetric limits."
        )
    return abs(upper)


def _aperture_shape_to_pals(shape: str, element_name: str) -> str:
    if shape == "rectangular":
        return "RECTANGULAR"
    if shape == "elliptical":
        return "ELLIPTICAL"
    raise NotImplementedError(
        f"Aperture {element_name!r} uses unsupported shape {shape!r}."
    )


def _aperture_shape_from_pals(shape: str, element_name: str) -> str:
    if shape == "RECTANGULAR":
        return "rectangular"
    if shape == "ELLIPTICAL":
        return "elliptical"
    raise NotImplementedError(
        f"PALS aperture {element_name!r} uses unsupported shape {shape!r}."
    )


# Start Element converters
def _drift_to_pals(element: cheetah.Drift, pals):
    _require_default(
        element.tracking_method == "linear",
        f"Drift {element.name!r} uses unsupported tracking_method "
        f"{element.tracking_method!r}.",
    )
    _, extras = _split_pals_extras(element)
    kwargs = _common_kwargs_from_extras(pals.Drift, extras)
    return pals.Drift(
        name=element.name,
        length=_nonnegative_length(element.length, element.name),
        **kwargs,
    )


def _drift_from_pals(element, factory_kwargs: dict) -> cheetah.Drift:
    converted = cheetah.Drift(
        length=_f2t(element.length, **factory_kwargs),
        name=element.name,
    )
    extras = _extract_extras(element, consumed_fields={"length"})
    return _attach_pals_extras(converted, extras)


def _marker_to_pals(element: cheetah.Marker, pals):
    _, extras = _split_pals_extras(element)
    kwargs = _common_kwargs_from_extras(pals.Marker, extras)
    return pals.Marker(name=element.name, **kwargs)


def _marker_from_pals(element, factory_kwargs: dict) -> cheetah.Element:
    if element.ApertureP is not None:
        return _aperture_from_pals(element, factory_kwargs)

    converted = cheetah.Marker(name=element.name, **factory_kwargs)
    extras = _extract_extras(element, consumed_fields={"length"})
    return _attach_pals_extras(converted, extras)


def _aperture_to_pals(element: cheetah.Aperture, pals):
    _, extras = _split_pals_extras(element)
    aperture_data = {
        "x_limits": _aperture_limits_from_extent(element.x_max, element.name, "x"),
        "y_limits": _aperture_limits_from_extent(element.y_max, element.name, "y"),
        "shape": _aperture_shape_to_pals(element.shape, element.name),
        "aperture_active": element.is_active,
    }
    aperture = _merge_group(pals, "ApertureP", aperture_data, extras)
    kwargs = _common_kwargs_from_extras(pals.Marker, extras)
    return pals.Marker(name=element.name, ApertureP=aperture, **kwargs)


def _aperture_from_pals(element, factory_kwargs: dict) -> cheetah.Aperture:
    aperture = _model_dump_or_empty(element.ApertureP)
    converted = cheetah.Aperture(
        x_max=_f2t(
            _extent_from_aperture_limits(
                aperture.get("x_limits", [None, None]), element.name, "x"
            ),
            **factory_kwargs,
        ),
        y_max=_f2t(
            _extent_from_aperture_limits(
                aperture.get("y_limits", [None, None]), element.name, "y"
            ),
            **factory_kwargs,
        ),
        shape=_aperture_shape_from_pals(
            aperture.get("shape", "RECTANGULAR"), element.name
        ),
        is_active=aperture.get("aperture_active", True),
        name=element.name,
        **factory_kwargs,
    )
    extras = _extract_extras(
        element,
        consumed_fields={"length"},
        consumed_groups={
            "ApertureP": {
                "x_limits",
                "y_limits",
                "shape",
                "aperture_active",
            }
        },
    )
    return _attach_pals_extras(converted, extras)


def _quad_to_pals(element: cheetah.Quadrupole, pals):
    _require_default(
        element.tracking_method == "linear",
        f"Quadrupole {element.name!r} uses unsupported tracking_method "
        f"{element.tracking_method!r}.",
    )
    _require_default(
        element.num_steps == 1,
        f"Quadrupole {element.name!r} has unsupported num_steps={element.num_steps}.",
    )
    _require_default(
        _is_zero(element.misalignment),
        f"Quadrupole {element.name!r} has unsupported misalignment.",
    )

    metadata, extras = _split_pals_extras(element)
    k1_key = metadata.get("k1_key", "Kn1")
    multipole_data = {
        k1_key: _t2f(element.k1),
    }
    if not _is_zero(element.tilt) or metadata.get("tilt1_present", False):
        multipole_data["tilt1"] = _t2f(element.tilt)
    multipole = _merge_group(pals, "MagneticMultipoleP", multipole_data, extras)
    kwargs = _common_kwargs_from_extras(pals.Quadrupole, extras)

    return pals.Quadrupole(
        name=element.name,
        length=_nonnegative_length(element.length, element.name),
        MagneticMultipoleP=multipole,
        **kwargs,
    )


def _quad_from_pals(element, factory_kwargs: dict) -> cheetah.Quadrupole:
    multipole = _model_dump_or_empty(element.MagneticMultipoleP)
    if element.MagneticMultipoleP is None and element.ElectricMultipoleP is not None:
        warnings.warn(
            f"PALS Quadrupole {element.name!r} has no magnetic multipole; "
            "using Marker.",
            category=UnknownElementWarning,
            stacklevel=2,
        )
        return _unknown_from_pals(element, factory_kwargs)

    k1, k1_key = _multipole_value(multipole, 1)
    tilt = _tilt_value(multipole, 1)
    converted = cheetah.Quadrupole(
        length=_f2t(element.length, **factory_kwargs),
        k1=_f2t(k1, **factory_kwargs),
        tilt=_f2t(tilt, **factory_kwargs),
        name=element.name,
    )

    consumed = set()
    if k1_key is not None:
        consumed.add(k1_key)
    if "tilt1" in multipole:
        consumed.add("tilt1")
    metadata = {}
    if k1_key is not None:
        metadata["k1_key"] = k1_key
    if "tilt1" in multipole:
        metadata["tilt1_present"] = True
    extras = _extract_extras(
        element,
        consumed_fields={"length"},
        consumed_groups={"MagneticMultipoleP": consumed},
        converter_metadata=metadata or None,
    )
    return _attach_pals_extras(converted, extras)


def _dipole_to_pals(element: cheetah.Dipole, pals):
    _require_default(
        element.tracking_method == "linear",
        f"Dipole {element.name!r} uses unsupported tracking_method "
        f"{element.tracking_method!r}.",
    )
    _require_default(
        _is_zero(element.gap) and _is_zero(element.gap_exit),
        f"Dipole {element.name!r} has unsupported gap settings.",
    )
    _require_default(
        element.fringe_at == "both",
        f"Dipole {element.name!r} has unsupported fringe_at={element.fringe_at!r}.",
    )
    _require_default(
        element.fringe_type == "linear_edge",
        f"Dipole {element.name!r} has unsupported fringe_type={element.fringe_type!r}.",
    )

    metadata, extras = _split_pals_extras(element)
    length = _nonnegative_length(element.length, element.name)
    angle = _t2f(element.angle)
    if length == 0.0 and angle != 0.0:
        raise ValueError(f"Dipole {element.name!r} has nonzero angle and zero length.")

    bend_data = {
        "e1": _t2f(element.dipole_e1),
        "e2": _t2f(element.dipole_e2),
        "edge_int1": _t2f(element.fringe_integral),
        "edge_int2": _t2f(element.fringe_integral_exit),
        "tilt_ref": _t2f(element.tilt),
    }
    if metadata.get("angle_key") == "rho_ref":
        bend_data["rho_ref"] = length / angle if angle != 0.0 else 0.0
    else:
        bend_data["g_ref"] = angle / length if length != 0.0 else 0.0

    bend = _merge_group(pals, "BendP", bend_data, extras)

    multipole = None
    multipole_extras = extras.get("MagneticMultipoleP", {})
    if not _is_zero(element.k1) or multipole_extras:
        k1_key = metadata.get("k1_key", "Kn1")
        multipole_data = {k1_key: _t2f(element.k1)}
        multipole = _merge_group(pals, "MagneticMultipoleP", multipole_data, extras)

    kwargs = _common_kwargs_from_extras(pals.SBend, extras)
    return pals.SBend(
        name=element.name,
        length=length,
        BendP=bend,
        MagneticMultipoleP=multipole,
        **kwargs,
    )


def _dipole_from_pals(element, factory_kwargs: dict) -> cheetah.Dipole:
    bend = _model_dump_or_empty(element.BendP)
    multipole = _model_dump_or_empty(element.MagneticMultipoleP)
    length = element.length

    angle_key = "g_ref"
    if bend.get("g_ref", 0.0) != 0.0:
        angle = bend["g_ref"] * length
    elif bend.get("rho_ref", 0.0) != 0.0:
        angle_key = "rho_ref"
        angle = length / bend["rho_ref"]
    else:
        angle = 0.0

    k1, k1_key = _multipole_value(multipole, 1)
    converted = cheetah.Dipole(
        length=_f2t(length, **factory_kwargs),
        angle=_f2t(angle, **factory_kwargs),
        k1=_f2t(k1, **factory_kwargs),
        dipole_e1=_f2t(bend.get("e1", 0.0), **factory_kwargs),
        dipole_e2=_f2t(bend.get("e2", 0.0), **factory_kwargs),
        tilt=_f2t(bend.get("tilt_ref", 0.0), **factory_kwargs),
        fringe_integral=_f2t(bend.get("edge_int1", 0.0), **factory_kwargs),
        fringe_integral_exit=_f2t(bend.get("edge_int2", 0.0), **factory_kwargs),
        name=element.name,
    )

    bend_consumed = {"e1", "e2", "edge_int1", "edge_int2", "tilt_ref", angle_key}
    multipole_consumed = set()
    if k1_key is not None:
        multipole_consumed.add(k1_key)
    metadata = {"angle_key": angle_key}
    if k1_key is not None:
        metadata["k1_key"] = k1_key
    extras = _extract_extras(
        element,
        consumed_fields={"length"},
        consumed_groups={
            "BendP": bend_consumed,
            "MagneticMultipoleP": multipole_consumed,
        },
        converter_metadata=metadata,
    )
    return _attach_pals_extras(converted, extras)


def _cavity_to_pals(element: cheetah.Cavity, pals):
    _require_default(
        element.cavity_type == "standing_wave",
        f"Cavity {element.name!r} has unsupported cavity_type="
        f"{element.cavity_type!r}.",
    )

    _, extras = _split_pals_extras(element)
    rf_data = {
        "frequency": _t2f(element.frequency),
        "voltage": _t2f(element.voltage),
        "phase": radians(_t2f(element.phase)),
        "cavity_type": "STANDING_WAVE",
    }
    rf = _merge_group(pals, "RFP", rf_data, extras)
    kwargs = _common_kwargs_from_extras(pals.RFCavity, extras)
    return pals.RFCavity(
        name=element.name,
        length=_nonnegative_length(element.length, element.name),
        RFP=rf,
        **kwargs,
    )


def _cavity_from_pals(element, factory_kwargs: dict) -> cheetah.Cavity:
    rf = _model_dump_or_empty(element.RFP)
    converted = cheetah.Cavity(
        length=_f2t(element.length, **factory_kwargs),
        voltage=_f2t(rf.get("voltage", 0.0), **factory_kwargs),
        phase=_f2t(degrees(rf.get("phase", 0.0)), **factory_kwargs),
        frequency=_f2t(rf.get("frequency", 0.0), **factory_kwargs),
        cavity_type="standing_wave",
        name=element.name,
    )
    extras = _extract_extras(
        element,
        consumed_fields={"length"},
        consumed_groups={"RFP": {"frequency", "voltage", "phase", "cavity_type"}},
    )
    return _attach_pals_extras(converted, extras)


def _corrector_multipole_to_pals(element: cheetah.Element, pals, data: dict):
    _, extras = _split_pals_extras(element)
    multipole = _merge_group(pals, "MagneticMultipoleP", data, extras)
    kwargs = _common_kwargs_from_extras(pals.Kicker, extras)
    return pals.Kicker(
        name=element.name,
        length=_nonnegative_length(element.length, element.name),
        MagneticMultipoleP=multipole,
        **kwargs,
    )


def _horizontal_corrector_to_pals(element: cheetah.HorizontalCorrector, pals):
    return _corrector_multipole_to_pals(
        element,
        pals,
        {"Kn0": _t2f(element.angle)},
    )


def _vertical_corrector_to_pals(element: cheetah.VerticalCorrector, pals):
    return _corrector_multipole_to_pals(
        element,
        pals,
        {"Ks0": _t2f(element.angle)},
    )


def _combined_corrector_to_pals(element: cheetah.CombinedCorrector, pals):
    return _corrector_multipole_to_pals(
        element,
        pals,
        {
            "Kn0": _t2f(element.horizontal_angle),
            "Ks0": _t2f(element.vertical_angle),
        },
    )


def _kicker_from_pals(element, factory_kwargs: dict) -> cheetah.Element:
    multipole = _model_dump_or_empty(element.MagneticMultipoleP)
    has_horizontal_angle = "Kn0" in multipole
    has_vertical_angle = "Ks0" in multipole
    horizontal_angle = multipole.get("Kn0", 0.0)
    vertical_angle = multipole.get("Ks0", 0.0)
    length = _f2t(element.length, **factory_kwargs)

    if has_horizontal_angle and has_vertical_angle:
        converted = cheetah.CombinedCorrector(
            length=length,
            horizontal_angle=_f2t(horizontal_angle, **factory_kwargs),
            vertical_angle=_f2t(vertical_angle, **factory_kwargs),
            name=element.name,
        )
    elif has_vertical_angle:
        converted = cheetah.VerticalCorrector(
            length=length,
            angle=_f2t(vertical_angle, **factory_kwargs),
            name=element.name,
        )
    else:
        converted = cheetah.HorizontalCorrector(
            length=length,
            angle=_f2t(horizontal_angle, **factory_kwargs),
            name=element.name,
        )

    consumed = set()
    if "Kn0" in multipole:
        consumed.add("Kn0")
    if "Ks0" in multipole:
        consumed.add("Ks0")
    extras = _extract_extras(
        element,
        consumed_fields={"length"},
        consumed_groups={"MagneticMultipoleP": consumed},
    )
    return _attach_pals_extras(converted, extras)


def _solenoid_to_pals(element: cheetah.Solenoid, pals):
    _require_default(
        _is_zero(element.misalignment),
        f"Solenoid {element.name!r} has unsupported misalignment.",
    )

    _, extras = _split_pals_extras(element)
    solenoid = _merge_group(
        pals,
        "SolenoidP",
        {"Ksol": _t2f(element.k)},
        extras,
    )
    kwargs = _common_kwargs_from_extras(pals.Solenoid, extras)
    return pals.Solenoid(
        name=element.name,
        length=_nonnegative_length(element.length, element.name),
        SolenoidP=solenoid,
        **kwargs,
    )


def _solenoid_from_pals(element, factory_kwargs: dict) -> cheetah.Solenoid:
    solenoid = _model_dump_or_empty(element.SolenoidP)
    converted = cheetah.Solenoid(
        length=_f2t(element.length, **factory_kwargs),
        k=_f2t(solenoid.get("Ksol", 0.0), **factory_kwargs),
        name=element.name,
    )
    extras = _extract_extras(
        element,
        consumed_fields={"length"},
        consumed_groups={"SolenoidP": {"Ksol"}},
    )
    return _attach_pals_extras(converted, extras)


def _unknown_to_pals(element: cheetah.Element, pals):
    raise NotImplementedError(
        f"Cheetah element {element.name!r} of type {element.__class__.__name__} "
        "is not supported by the PALS converter."
    )


def _unknown_from_pals(element, factory_kwargs: dict) -> cheetah.Marker:
    warnings.warn(
        f"PALS element {element.name!r} of kind {element.kind!r} is not supported by "
        "the PALS converter. Using Marker.",
        category=UnknownElementWarning,
        stacklevel=2,
    )
    converted = cheetah.Marker(name=element.name)
    converted.pals_extras = _pals_payload(element)
    return converted


_TO_PALS = {
    cheetah.Aperture: _aperture_to_pals,
    cheetah.Cavity: _cavity_to_pals,
    cheetah.CombinedCorrector: _combined_corrector_to_pals,
    cheetah.Drift: _drift_to_pals,
    cheetah.HorizontalCorrector: _horizontal_corrector_to_pals,
    cheetah.Marker: _marker_to_pals,
    cheetah.Quadrupole: _quad_to_pals,
    cheetah.Dipole: _dipole_to_pals,
    cheetah.Solenoid: _solenoid_to_pals,
    cheetah.VerticalCorrector: _vertical_corrector_to_pals,
}

_FROM_PALS = {
    "Drift": _drift_from_pals,
    "Kicker": _kicker_from_pals,
    "Marker": _marker_from_pals,
    "Quadrupole": _quad_from_pals,
    "RFCavity": _cavity_from_pals,
    "SBend": _dipole_from_pals,
    "Solenoid": _solenoid_from_pals,
}


def _element_to_pals(element: cheetah.Element, pals):
    if isinstance(element, cheetah.Segment):
        return _segment_to_beamline(element, pals, element.name)

    converter = _TO_PALS.get(type(element), _unknown_to_pals)
    return converter(element, pals)


def _segment_to_beamline(segment: cheetah.Segment, pals, name: str | None):
    return pals.BeamLine(
        name=name or segment.name,
        line=[_element_to_pals(element, pals) for element in segment.elements],
    )


def _element_from_pals(element, factory_kwargs: dict) -> cheetah.Element:
    pals = _import_pals()
    if isinstance(element, pals.PlaceholderName):
        if element.element is None:
            raise NotImplementedError(
                f"Unresolved PALS element reference {element.name!r} cannot be "
                "converted."
            )
        element = element.element

    if isinstance(element, pals.BeamLine):
        return _beamline_from_pals(element, factory_kwargs)
    if isinstance(element, pals.Lattice):
        return _lattice_from_pals(element, factory_kwargs)

    # Look for existing converters based on the "kind" field
    converter = _FROM_PALS.get(getattr(element, "kind", None), _unknown_from_pals)
    return converter(element, factory_kwargs)


def _beamline_from_pals(beamline, factory_kwargs: dict) -> cheetah.Segment:
    return cheetah.Segment(
        elements=[
            _element_from_pals(element, factory_kwargs) for element in beamline.line
        ],
        name=beamline.name,
    )


def _lattice_from_pals(lattice, factory_kwargs: dict) -> cheetah.Segment:
    if len(lattice.branches) != 1:
        raise NotImplementedError(
            "PALS conversion only supports Lattice objects with one branch."
        )
    return _element_from_pals(lattice.branches[0], factory_kwargs)


def convert_lattice_to_pals(
    segment: cheetah.Segment,
    name: str = "lattice",
):
    """
    Convert a Cheetah segment to a PALS Lattice.

    :param segment: Cheetah `Segment` to convert.
    :param name: Name to give the generated PALS `Lattice`.
    :return: PALS `Lattice` representing the Cheetah segment.
    """
    pals = _import_pals()
    if not isinstance(segment, cheetah.Segment):
        raise TypeError("convert_lattice_to_pals expects a cheetah.Segment.")

    branch_name = segment.name or f"{name}_line"
    return pals.Lattice(
        name=name,
        branches=[_segment_to_beamline(segment, pals, branch_name)],
    )


def convert_lattice_from_pals(
    obj,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> cheetah.Segment:
    """
    Convert a PALS Lattice, BeamLine, or PALSroot to a Cheetah segment.

    :param obj: PALS object to convert. May be a `Lattice`, `BeamLine`,
        `PALSroot`, or individual PALS element.
    :param device: Device to use for converted tensors. If `None`, uses the
        current default PyTorch device.
    :param dtype: Data type to use for converted tensors. If `None`, uses the
        current default PyTorch dtype.
    :return: Cheetah `Segment` representing the PALS object.
    """
    pals = _import_pals()
    factory_kwargs = _factory_kwargs(device=device, dtype=dtype)

    if isinstance(obj, pals.PALSroot):
        if len(obj.facility) != 1:
            raise NotImplementedError(
                "PALS conversion only supports PALSroot objects with one "
                "facility entry."
            )
        obj = obj.facility[0]

    converted = _element_from_pals(obj, factory_kwargs)
    if isinstance(converted, cheetah.Segment):
        return converted
    return cheetah.Segment(elements=[converted], name=getattr(obj, "name", "pals"))


def save_lattice_to_pals(
    segment: cheetah.Segment,
    filename: str | Path,
    name: str = "lattice",
) -> None:
    """
    Save a Cheetah segment as a PALS file.

    :param segment: Cheetah `Segment` to save.
    :param filename: Path to the PALS file to write.
    :param name: Name to give the generated PALS `Lattice`.
    :return: None.
    """
    pals = _import_pals()
    pals.store(
        str(filename),
        convert_lattice_to_pals(segment, name=name),
    )


def load_lattice_from_pals(
    filename: str | Path,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> cheetah.Segment:
    """
    Load a Cheetah segment from a PALS file.

    :param filename: Path to the PALS file to read.
    :param device: Device to use for converted tensors. If `None`, uses the
        current default PyTorch device.
    :param dtype: Data type to use for converted tensors. If `None`, uses the
        current default PyTorch dtype.
    :return: Cheetah `Segment` loaded from the PALS file.
    """
    pals = _import_pals()
    return convert_lattice_from_pals(
        pals.load(str(filename)), device=device, dtype=dtype
    )
