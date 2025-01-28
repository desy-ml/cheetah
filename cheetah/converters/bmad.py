import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

import cheetah
from cheetah.converters.utils.fortran_namelist import (
    merge_delimiter_continued_lines,
    parse_lines,
    read_clean_lines,
    validate_understood_properties,
)


def convert_element(
    name: str,
    context: dict,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> "cheetah.Element":
    """Convert a parsed Bmad element dict to a cheetah Element.

    :param name: Name of the (top-level) element to convert.
    :param context: Context dictionary parsed from Bmad lattice file(s).
    :param device: Device to put the element on. If `None`, the device is set to
        `torch.device("cpu")`.
    :param dtype: Data type to use for the element. Default is `torch.float32`.
    :return: Converted cheetah Element. If you are calling this function yourself
        as a user of Cheetah, this is most likely a `Segment`.
    """
    bmad_parsed = context[name]

    if isinstance(bmad_parsed, list):
        return cheetah.Segment(
            elements=[
                convert_element(element_name, context, device, dtype)
                for element_name in bmad_parsed
            ],
            name=name,
        )
    elif isinstance(bmad_parsed, dict) and "element_type" in bmad_parsed:
        if bmad_parsed["element_type"] == "marker":
            validate_understood_properties(
                [
                    "element_type",
                    "alias",
                    "type",
                    "sr_wake",
                    r"sr_wake%scale_with_length",
                    r"sr_wake%amp_scale",
                ],
                bmad_parsed,
            )
            return cheetah.Marker(name=name)
        elif bmad_parsed["element_type"] == "monitor":
            validate_understood_properties(
                ["element_type", "alias", "type", "l"], bmad_parsed
            )
            if "l" in bmad_parsed:
                return cheetah.Drift(
                    length=torch.tensor(bmad_parsed["l"]),
                    name=name,
                    device=device,
                    dtype=dtype,
                )
            else:
                return cheetah.Marker(name=name)
        elif bmad_parsed["element_type"] == "instrument":
            validate_understood_properties(
                ["element_type", "alias", "type", "l"], bmad_parsed
            )
            if "l" in bmad_parsed:
                return cheetah.Drift(
                    length=torch.tensor(bmad_parsed["l"]),
                    name=name,
                    device=device,
                    dtype=dtype,
                )
            else:
                return cheetah.Marker(name=name)
        elif bmad_parsed["element_type"] == "pipe":
            validate_understood_properties(
                ["element_type", "alias", "type", "l", "descrip"], bmad_parsed
            )
            return cheetah.Drift(
                length=torch.tensor(bmad_parsed["l"]),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif bmad_parsed["element_type"] == "drift":
            validate_understood_properties(
                ["element_type", "l", "type", "descrip"], bmad_parsed
            )
            return cheetah.Drift(
                length=torch.tensor(bmad_parsed["l"]),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif bmad_parsed["element_type"] == "hkicker":
            validate_understood_properties(
                ["element_type", "type", "alias"], bmad_parsed
            )
            return cheetah.HorizontalCorrector(
                length=torch.tensor(bmad_parsed.get("l", 0.0)),
                angle=torch.tensor(bmad_parsed.get("kick", 0.0)),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif bmad_parsed["element_type"] == "vkicker":
            validate_understood_properties(
                ["element_type", "type", "alias"], bmad_parsed
            )
            return cheetah.VerticalCorrector(
                length=torch.tensor(bmad_parsed.get("l", 0.0)),
                angle=torch.tensor(bmad_parsed.get("kick", 0.0)),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif bmad_parsed["element_type"] == "sbend":
            validate_understood_properties(
                [
                    "element_type",
                    "alias",
                    "type",
                    "hgap",
                    "l",
                    "angle",
                    "e1",
                    "e2",
                    "fint",
                    "fintx",
                    "fringe_type",
                    "ref_tilt",
                    "g",
                    "dg",
                ],
                bmad_parsed,
            )
            return cheetah.Dipole(
                length=torch.tensor(bmad_parsed["l"]),
                gap=torch.tensor(2 * bmad_parsed.get("hgap", 0.0)),
                angle=torch.tensor(bmad_parsed.get("angle", 0.0)),
                dipole_e1=torch.tensor(bmad_parsed["e1"]),
                dipole_e2=torch.tensor(bmad_parsed.get("e2", 0.0)),
                tilt=torch.tensor(bmad_parsed.get("ref_tilt", 0.0)),
                fringe_integral=torch.tensor(bmad_parsed.get("fint", 0.0)),
                fringe_integral_exit=(
                    torch.tensor(bmad_parsed["fintx"])
                    if "fintx" in bmad_parsed
                    else None
                ),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif bmad_parsed["element_type"] == "quadrupole":
            # TODO: Aperture for quadrupoles?
            validate_understood_properties(
                ["element_type", "l", "k1", "type", "aperture", "alias", "tilt"],
                bmad_parsed,
            )
            return cheetah.Quadrupole(
                length=torch.tensor(bmad_parsed["l"]),
                k1=torch.tensor(bmad_parsed["k1"]),
                tilt=torch.tensor(bmad_parsed.get("tilt", 0.0)),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif bmad_parsed["element_type"] == "solenoid":
            validate_understood_properties(
                ["element_type", "l", "ks", "alias"], bmad_parsed
            )
            return cheetah.Solenoid(
                length=torch.tensor(bmad_parsed["l"]),
                k=torch.tensor(bmad_parsed["ks"]),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif bmad_parsed["element_type"] == "lcavity":
            validate_understood_properties(
                [
                    "element_type",
                    "l",
                    "type",
                    "rf_frequency",
                    "voltage",
                    "phi0",
                    "sr_wake",
                    "cavity_type",
                    "alias",
                ],
                bmad_parsed,
            )
            return cheetah.Cavity(
                length=torch.tensor(bmad_parsed["l"]),
                voltage=torch.tensor(bmad_parsed.get("voltage", 0.0)),
                phase=torch.tensor(
                    -np.degrees(bmad_parsed.get("phi0", 0.0) * 2 * np.pi)
                ),
                frequency=torch.tensor(bmad_parsed["rf_frequency"]),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif bmad_parsed["element_type"] == "rcollimator":
            validate_understood_properties(
                ["element_type", "l", "alias", "type", "x_limit", "y_limit"],
                bmad_parsed,
            )
            return cheetah.Segment(
                elements=[
                    cheetah.Drift(
                        length=torch.tensor(bmad_parsed.get("l", 0.0)),
                        name=name + "_drift",
                        device=device,
                        dtype=dtype,
                    ),
                    cheetah.Aperture(
                        x_max=torch.tensor(bmad_parsed.get("x_limit", np.inf)),
                        y_max=torch.tensor(bmad_parsed.get("y_limit", np.inf)),
                        shape="rectangular",
                        name=name + "_aperture",
                        device=device,
                        dtype=dtype,
                    ),
                ],
                name=name,
            )
        elif bmad_parsed["element_type"] == "ecollimator":
            validate_understood_properties(
                ["element_type", "l", "alias", "type", "x_limit", "y_limit"],
                bmad_parsed,
            )
            return cheetah.Segment(
                elements=[
                    cheetah.Drift(
                        length=torch.tensor(bmad_parsed.get("l", 0.0)),
                        name=name + "_drift",
                        device=device,
                        dtype=dtype,
                    ),
                    cheetah.Aperture(
                        x_max=torch.tensor(bmad_parsed.get("x_limit", np.inf)),
                        y_max=torch.tensor(bmad_parsed.get("y_limit", np.inf)),
                        shape="elliptical",
                        name=name + "_aperture",
                        device=device,
                        dtype=dtype,
                    ),
                ],
            )
        elif bmad_parsed["element_type"] == "wiggler":
            validate_understood_properties(
                [
                    "element_type",
                    "type",
                    "l_period",
                    "n_period",
                    "b_max",
                    "l",
                    "alias",
                    "tilt",
                    "ds_step",
                ],
                bmad_parsed,
            )
            return cheetah.Undulator(
                length=torch.tensor(bmad_parsed["l"]),
                name=name,
                device=device,
                dtype=dtype,
            )
        elif bmad_parsed["element_type"] == "patch":
            # TODO: Does this need to be implemented in Cheetah in a more proper way?
            validate_understood_properties(["element_type", "tilt"], bmad_parsed)
            return cheetah.Drift(
                length=torch.tensor(bmad_parsed.get("l", 0.0)),
                name=name,
                device=device,
                dtype=dtype,
            )
        else:
            print(
                f"WARNING: Element {name} of type {bmad_parsed['element_type']} cannot"
                " be converted correctly. Using drift section instead."
            )
            # TODO: Remove the length if by adding markers to Cheeath
            return cheetah.Drift(
                name=name,
                length=torch.tensor(bmad_parsed.get("l", 0.0)),
                device=device,
                dtype=dtype,
            )
    else:
        raise ValueError(f"Unknown Bmad element type for {name = }")  # noqa: E202, E251


def convert_lattice_to_cheetah(
    bmad_lattice_file_path: Path,
    environment_variables: Optional[dict] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
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
    :param device: Device to use for the lattice. If `None`, the device is set to
        `torch.device("cpu")`.
    :param dtype: Data type to use for the lattice. Default is `torch.float32`.
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
    return convert_element(context["__use__"], context, device, dtype)
