import os
import warnings
from pathlib import Path

import torch

import cheetah
from cheetah.converters.utils.fortran_namelist import (
    merge_delimiter_continued_lines,
    parse_lines,
    read_clean_lines,
    validate_understood_properties,
)
from cheetah.utils import UnknownElementWarning


def convert_element(
    name: str,
    context: dict,
    sanitize_name: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> "cheetah.Element":
    """Convert a parsed Bmad element dict to a cheetah Element.

    :param name: Name of the (top-level) element to convert.
    :param context: Context dictionary parsed from Bmad lattice file(s).
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name.
    :param device: Device to put the element on. If `None`, the current default device
        of PyTorch is used.
    :param dtype: Data type to use for the element. If `None`, the current default dtype
        of PyTorch is used.
    :return: Converted cheetah Element. If you are calling this function yourself
        as a user of Cheetah, this is most likely a `Segment`.
    """
    factory_kwargs = {
        "device": device or torch.get_default_device(),
        "dtype": dtype or torch.get_default_dtype(),
    }
    bmad_parsed = context[name]

    shared_properties = ["element_type", "alias", "type"]

    if isinstance(bmad_parsed, list):
        return cheetah.Segment(
            elements=[
                convert_element(element_name, context, sanitize_name, device, dtype)
                for element_name in bmad_parsed
            ],
            name=name,
            sanitize_name=sanitize_name,
        )
    elif isinstance(bmad_parsed, dict) and "element_type" in bmad_parsed:
        if bmad_parsed["element_type"] == "marker":
            validate_understood_properties(shared_properties, bmad_parsed)
            return cheetah.Marker(name=name, sanitize_name=sanitize_name)
        elif bmad_parsed["element_type"] == "monitor":
            validate_understood_properties(shared_properties + ["l"], bmad_parsed)
            if "l" in bmad_parsed:
                return cheetah.Drift(
                    length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                    name=name,
                    sanitize_name=sanitize_name,
                )
            else:
                return cheetah.Marker(name=name, sanitize_name=sanitize_name)
        elif bmad_parsed["element_type"] == "instrument":
            validate_understood_properties(shared_properties + ["l"], bmad_parsed)
            if "l" in bmad_parsed:
                return cheetah.Drift(
                    length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                    name=name,
                    sanitize_name=sanitize_name,
                )
            else:
                return cheetah.Marker(name=name, sanitize_name=sanitize_name)
        elif bmad_parsed["element_type"] == "pipe":
            validate_understood_properties(
                shared_properties + ["l", "descrip"], bmad_parsed
            )
            return cheetah.Drift(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif bmad_parsed["element_type"] == "drift":
            validate_understood_properties(
                shared_properties + ["l", "descrip"], bmad_parsed
            )
            return cheetah.Drift(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif bmad_parsed["element_type"] == "hkicker":
            validate_understood_properties(shared_properties + ["kick"], bmad_parsed)
            return cheetah.HorizontalCorrector(
                length=torch.tensor(bmad_parsed.get("l", 0.0), **factory_kwargs),
                angle=torch.tensor(bmad_parsed.get("kick", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif bmad_parsed["element_type"] == "vkicker":
            validate_understood_properties(shared_properties + ["kick"], bmad_parsed)
            return cheetah.VerticalCorrector(
                length=torch.tensor(bmad_parsed.get("l", 0.0), **factory_kwargs),
                angle=torch.tensor(bmad_parsed.get("kick", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif bmad_parsed["element_type"] == "sbend":
            validate_understood_properties(
                shared_properties
                + ["hgap", "l", "angle", "e1", "e2", "fint", "fintx", "ref_tilt"],
                bmad_parsed,
            )
            return cheetah.Dipole(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                gap=torch.tensor(2 * bmad_parsed.get("hgap", 0.0), **factory_kwargs),
                angle=torch.tensor(bmad_parsed.get("angle", 0.0), **factory_kwargs),
                dipole_e1=torch.tensor(bmad_parsed["e1"], **factory_kwargs),
                dipole_e2=torch.tensor(bmad_parsed.get("e2", 0.0), **factory_kwargs),
                tilt=torch.tensor(bmad_parsed.get("ref_tilt", 0.0), **factory_kwargs),
                fringe_integral=torch.tensor(
                    bmad_parsed.get("fint", 0.0), **factory_kwargs
                ),
                fringe_integral_exit=(
                    torch.tensor(bmad_parsed["fintx"], **factory_kwargs)
                    if "fintx" in bmad_parsed
                    else None
                ),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif bmad_parsed["element_type"] == "quadrupole":
            validate_understood_properties(
                shared_properties + ["l", "k1", "tilt"], bmad_parsed
            )
            return cheetah.Quadrupole(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                k1=torch.tensor(bmad_parsed["k1"], **factory_kwargs),
                tilt=torch.tensor(bmad_parsed.get("tilt", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif bmad_parsed["element_type"] == "sextupole":
            validate_understood_properties(
                shared_properties + ["l", "k2", "tilt"], bmad_parsed
            )
            return cheetah.Sextupole(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                k2=torch.tensor(bmad_parsed["k2"], **factory_kwargs),
                tilt=torch.tensor(bmad_parsed.get("tilt", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif bmad_parsed["element_type"] == "solenoid":
            validate_understood_properties(shared_properties + ["l", "ks"], bmad_parsed)
            return cheetah.Solenoid(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                k=torch.tensor(bmad_parsed["ks"], **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif bmad_parsed["element_type"] == "lcavity":
            validate_understood_properties(
                shared_properties + ["l", "rf_frequency", "voltage", "phi0"],
                bmad_parsed,
            )
            return cheetah.Cavity(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                voltage=torch.tensor(bmad_parsed.get("voltage", 0.0), **factory_kwargs),
                phase=-torch.rad2deg(
                    torch.tensor(bmad_parsed.get("phi0", 0.0), **factory_kwargs)
                    * 2
                    * torch.pi
                ),
                frequency=torch.tensor(bmad_parsed["rf_frequency"], **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif bmad_parsed["element_type"] == "rcollimator":
            validate_understood_properties(
                shared_properties + ["l", "x_limit", "y_limit"],
                bmad_parsed,
            )
            return cheetah.Segment(
                elements=[
                    cheetah.Drift(
                        length=torch.tensor(
                            bmad_parsed.get("l", 0.0), **factory_kwargs
                        ),
                        name=name + "_drift",
                        sanitize_name=sanitize_name,
                    ),
                    cheetah.Aperture(
                        x_max=torch.tensor(
                            bmad_parsed.get("x_limit", torch.inf), **factory_kwargs
                        ),
                        y_max=torch.tensor(
                            bmad_parsed.get("y_limit", torch.inf), **factory_kwargs
                        ),
                        shape="rectangular",
                        name=name + "_aperture",
                        sanitize_name=sanitize_name,
                    ),
                ],
                name=name,
                sanitize_name=sanitize_name,
            )
        elif bmad_parsed["element_type"] == "ecollimator":
            validate_understood_properties(
                shared_properties + ["l", "x_limit", "y_limit"],
                bmad_parsed,
            )
            return cheetah.Segment(
                elements=[
                    cheetah.Drift(
                        length=torch.tensor(
                            bmad_parsed.get("l", 0.0), **factory_kwargs
                        ),
                        name=name + "_drift",
                        sanitize_name=sanitize_name,
                    ),
                    cheetah.Aperture(
                        x_max=torch.tensor(
                            bmad_parsed.get("x_limit", torch.inf), **factory_kwargs
                        ),
                        y_max=torch.tensor(
                            bmad_parsed.get("y_limit", torch.inf), **factory_kwargs
                        ),
                        shape="elliptical",
                        name=name + "_aperture",
                        sanitize_name=sanitize_name,
                    ),
                ],
                name=name,
                sanitize_name=sanitize_name,
            )
        elif bmad_parsed["element_type"] == "wiggler":
            validate_understood_properties(shared_properties + ["l"], bmad_parsed)
            return cheetah.Undulator(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif bmad_parsed["element_type"] == "patch":
            # TODO: Does this need to be implemented in Cheetah in a more proper way?
            validate_understood_properties(shared_properties + ["l"], bmad_parsed)
            return cheetah.Drift(
                length=torch.tensor(bmad_parsed.get("l", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        else:
            warnings.warn(
                f"Element {name} of type {bmad_parsed['element_type']} cannot be"
                " converted correctly. Using drift section instead.",
                category=UnknownElementWarning,
                stacklevel=2,
            )
            return cheetah.Drift(
                length=torch.tensor(bmad_parsed.get("l", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
    else:
        raise ValueError(f"Unknown Bmad element type for {name = }")  # noqa: E202, E251


def convert_lattice_to_cheetah(
    bmad_lattice_file_path: Path,
    environment_variables: dict | None = None,
    sanitize_names: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> "cheetah.Element":
    """
    Convert a Bmad lattice file to a Cheetah `Segment`.

    NOTE: This function was designed at the example of the LCLS lattice. While this
        lattice is extensive, this function might not properly convert all features of
        a Bmad lattice. If you find that this function does not work for your lattice,
        please open an issue on GitHub.

    :param bmad_lattice_file_path: Path to the Bmad lattice file.
    :param environment_variables: Dictionary of environment variables to use when
        parsing the lattice file.
    :param sanitize_names: Whether to sanitise the names of the elements to be valid
        Python variable names. This is needed if you want to use the
        `segment.element_name` syntax to access the element in a segment.
    :param device: Device to use for the lattice. If `None`, the current default device
        of PyTorch is used.
    :param dtype: Data type to use for the lattice. If `None`, the current default dtype
        of PyTorch is used.
    :return: Cheetah `Segment` representing the Bmad lattice.
    """
    # If provided, set environment variables
    if environment_variables is not None:
        for key, value in environment_variables.items():
            os.environ[key] = value

    # Replace environment variables in the lattice file path
    resolved_lattice_file_path = Path(
        *[
            os.environ[part[1:]] if part.startswith("$") else part
            for part in bmad_lattice_file_path.parts
        ]
    )

    # Read and clean the lattice file(s)
    lines = read_clean_lines(resolved_lattice_file_path)

    # Merge multi-line statements
    merged_lines = merge_delimiter_continued_lines(
        lines, delimiter="&", remove_delimiter=True
    )
    merged_lines = merge_delimiter_continued_lines(
        merged_lines, delimiter=",", remove_delimiter=False
    )
    merged_lines = merge_delimiter_continued_lines(
        merged_lines, delimiter="{", remove_delimiter=False
    )
    assert len(merged_lines) <= len(
        lines
    ), "Merging lines should never produce more lines than there were before."

    # Parse the lattice file(s), i.e. basically execute them
    context = parse_lines(merged_lines)

    # Convert the parsed lattice info to Cheetah elements
    return convert_element(
        name=context["__use__"],
        context=context,
        sanitize_name=sanitize_names,
        device=device,
        dtype=dtype,
    )
