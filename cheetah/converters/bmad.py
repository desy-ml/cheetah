import os
import warnings
from pathlib import Path

import torch

import cheetah
from cheetah.converters.utils import fortran_namelist
from cheetah.utils import PhysicsWarning, UnknownElementWarning


def convert_element(
    name: str,
    context: dict,
    superimpositions: dict[str, list[str]],
    sanitize_name: bool | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> "cheetah.Element":
    """
    Convert a parsed Bmad element dict to a Cheetah `Element`.

    :param name: Name of the (top-level) element to convert.
    :param context: Context dictionary parsed from Bmad lattice file(s).
    :param superimpositions: Mapping of base element names to lists of superimposed
        element names.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. If `None` (default), a warning is raised for invalid names. Set to `True`
        to sanitise, or `False` to silence the warning.
    :param device: Device to put the element on. If `None`, the current default device
        of PyTorch is used.
    :param dtype: Data type to use for the element. If `None`, the current default dtype
        of PyTorch is used.
    :return: Converted Cheetah `Element`. If you are calling this function yourself
        as a user of Cheetah, this is most likely a `Segment`.
    """
    factory_kwargs = {
        "device": device or torch.get_default_device(),
        "dtype": dtype or torch.get_default_dtype(),
    }

    bmad_parsed = context[name]
    metadata = (
        {k: bmad_parsed[k] for k in ["alias", "type"] if k in bmad_parsed}
        if isinstance(bmad_parsed, dict)
        else {}
    )

    shared_properties = ["element_type", "alias", "type", "ref", "superimpose"]

    if isinstance(bmad_parsed, list):
        element = cheetah.Segment(
            elements=[
                convert_element(
                    element_name,
                    context,
                    sanitize_name=sanitize_name,
                    device=device,
                    dtype=dtype,
                    superimpositions=superimpositions,
                )
                for element_name in bmad_parsed
            ],
            name=name,
            sanitize_name=sanitize_name,
        )
    elif isinstance(bmad_parsed, dict) and "element_type" in bmad_parsed:
        if bmad_parsed["element_type"] == "marker":
            fortran_namelist.validate_understood_properties(
                shared_properties, bmad_parsed
            )
            element = cheetah.Marker(
                name=name, sanitize_name=sanitize_name, metadata=metadata
            )
        elif bmad_parsed["element_type"] == "monitor":
            fortran_namelist.validate_understood_properties(
                shared_properties + ["l"], bmad_parsed
            )
            if "l" in bmad_parsed:
                element = cheetah.Drift(
                    length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                    name=name,
                    sanitize_name=sanitize_name,
                    metadata=metadata,
                )
            else:
                element = cheetah.Marker(
                    name=name, sanitize_name=sanitize_name, metadata=metadata
                )
        elif bmad_parsed["element_type"] == "instrument":
            fortran_namelist.validate_understood_properties(
                shared_properties + ["l"], bmad_parsed
            )
            if "l" in bmad_parsed:
                element = cheetah.Drift(
                    length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                    name=name,
                    sanitize_name=sanitize_name,
                    metadata=metadata,
                )
            else:
                element = cheetah.Marker(
                    name=name, sanitize_name=sanitize_name, metadata=metadata
                )
        elif bmad_parsed["element_type"] == "pipe":
            fortran_namelist.validate_understood_properties(
                shared_properties + ["l", "descrip"], bmad_parsed
            )
            element = cheetah.Drift(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
                metadata=metadata,
            )
        elif bmad_parsed["element_type"] == "drift":
            fortran_namelist.validate_understood_properties(
                shared_properties + ["l", "descrip"], bmad_parsed
            )
            element = cheetah.Drift(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
                metadata=metadata,
            )
        elif bmad_parsed["element_type"] == "hkicker":
            fortran_namelist.validate_understood_properties(
                shared_properties + ["l", "kick"], bmad_parsed
            )
            element = cheetah.HorizontalCorrector(
                length=torch.tensor(bmad_parsed.get("l", 0.0), **factory_kwargs),
                angle=torch.tensor(bmad_parsed.get("kick", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
                metadata=metadata,
            )
        elif bmad_parsed["element_type"] == "vkicker":
            fortran_namelist.validate_understood_properties(
                shared_properties + ["l", "kick"], bmad_parsed
            )
            element = cheetah.VerticalCorrector(
                length=torch.tensor(bmad_parsed.get("l", 0.0), **factory_kwargs),
                angle=torch.tensor(bmad_parsed.get("kick", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
                metadata=metadata,
            )
        elif bmad_parsed["element_type"] == "sbend":
            fortran_namelist.validate_understood_properties(
                shared_properties
                + ["hgap", "l", "angle", "e1", "e2", "fint", "fintx", "ref_tilt"],
                bmad_parsed,
            )
            element = cheetah.Dipole(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                gap=torch.tensor(2 * bmad_parsed.get("hgap", 0.0), **factory_kwargs),
                angle=torch.tensor(bmad_parsed.get("angle", 0.0), **factory_kwargs),
                dipole_e1=torch.tensor(bmad_parsed.get("e1", 0.0), **factory_kwargs),
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
                metadata=metadata,
            )
        elif bmad_parsed["element_type"] == "quadrupole":
            fortran_namelist.validate_understood_properties(
                shared_properties + ["l", "k1", "tilt"], bmad_parsed
            )
            element = cheetah.Quadrupole(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                k1=torch.tensor(bmad_parsed["k1"], **factory_kwargs),
                tilt=torch.tensor(bmad_parsed.get("tilt", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
                metadata=metadata,
            )
        elif bmad_parsed["element_type"] == "sextupole":
            fortran_namelist.validate_understood_properties(
                shared_properties + ["l", "k2", "tilt"], bmad_parsed
            )
            element = cheetah.Sextupole(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                k2=torch.tensor(bmad_parsed["k2"], **factory_kwargs),
                tilt=torch.tensor(bmad_parsed.get("tilt", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
                metadata=metadata,
            )
        elif bmad_parsed["element_type"] == "solenoid":
            fortran_namelist.validate_understood_properties(
                shared_properties + ["l", "ks"], bmad_parsed
            )
            element = cheetah.Solenoid(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                k=torch.tensor(bmad_parsed["ks"], **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
                metadata=metadata,
            )
        elif bmad_parsed["element_type"] == "lcavity":
            fortran_namelist.validate_understood_properties(
                shared_properties + ["l", "rf_frequency", "voltage", "phi0"],
                bmad_parsed,
            )
            element = cheetah.Cavity(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                voltage=torch.tensor(bmad_parsed.get("voltage", 0.0), **factory_kwargs),
                phase=-(
                    torch.tensor(bmad_parsed.get("phi0", 0.0), **factory_kwargs)
                    * 2
                    * torch.pi
                ).rad2deg(),
                frequency=torch.tensor(bmad_parsed["rf_frequency"], **factory_kwargs),
                cavity_type=bmad_parsed["cavity_type"],
                name=name,
                sanitize_name=sanitize_name,
                metadata=metadata,
            )
        elif bmad_parsed["element_type"] == "crab_cavity":
            fortran_namelist.validate_understood_properties(
                shared_properties + ["l", "rf_frequency", "voltage", "phi0"],
                bmad_parsed,
            )
            element = cheetah.TransverseDeflectingCavity(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                voltage=torch.tensor(bmad_parsed.get("voltage", 0.0), **factory_kwargs),
                phase=-(torch.tensor(bmad_parsed.get("phi0", 0.0), **factory_kwargs)),
                frequency=torch.tensor(bmad_parsed["rf_frequency"], **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
                metadata=metadata,
            )
        elif bmad_parsed["element_type"] == "rcollimator":
            fortran_namelist.validate_understood_properties(
                shared_properties + ["l", "x_limit", "y_limit"],
                bmad_parsed,
            )
            element = cheetah.Segment(
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
                metadata=metadata,
            )
        elif bmad_parsed["element_type"] == "ecollimator":
            fortran_namelist.validate_understood_properties(
                shared_properties + ["l", "x_limit", "y_limit"],
                bmad_parsed,
            )
            element = cheetah.Segment(
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
                metadata=metadata,
            )
        elif bmad_parsed["element_type"] == "wiggler":
            fortran_namelist.validate_understood_properties(
                shared_properties + ["l", "l_period"], bmad_parsed
            )

            # TODO: Map the magnetic strength `b_max of Bmad to the undulator
            # coefficient `kx`.
            element = cheetah.Undulator(
                length=torch.tensor(bmad_parsed["l"], **factory_kwargs),
                period=torch.tensor(bmad_parsed["l_period"], **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
                metadata=metadata,
            )
        elif bmad_parsed["element_type"] == "patch":
            # TODO: Does this need to be implemented in Cheetah in a more proper way?
            fortran_namelist.validate_understood_properties(
                shared_properties + ["l"], bmad_parsed
            )
            element = cheetah.Drift(
                length=torch.tensor(bmad_parsed.get("l", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
                metadata=metadata,
            )
        else:
            warnings.warn(
                f"Element {name} of type {bmad_parsed['element_type']} cannot be"
                " converted correctly. Using drift section instead.",
                category=UnknownElementWarning,
                stacklevel=2,
            )
            element = cheetah.Drift(
                length=torch.tensor(bmad_parsed.get("l", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
                metadata=metadata,
            )
    else:
        raise ValueError(f"Unknown Bmad element type for {name = }")  # noqa: E202, E251

    if name in superimpositions:
        superimposed_elements = [
            convert_element(
                superimposed_name,
                context,
                superimpositions,
                sanitize_name=sanitize_name,
                device=device,
                dtype=dtype,
            )
            for superimposed_name in superimpositions[name]
        ]

        try:
            return cheetah.Superimposed(
                base_element=element,
                superimposed_element=superimposed_elements,
                name=name,
                sanitize_name=sanitize_name,
                metadata=metadata,
            )
        except AssertionError as error:
            warnings.warn(
                f"Could not superimpose {superimpositions[name]} on {name}. "
                f"Keeping only the base element. Reason: {error}",
                category=PhysicsWarning,
                stacklevel=2,
            )

    return element


def collect_superimpositions(context: dict) -> dict[str, list[str]]:
    """
    Map base element names to lists of their superimposed element names from context.

    :param context: Context dictionary parsed from Bmad lattice file(s).
    :return: Mapping of base element names to lists of superimposed element names.
    """
    superimpositions: dict[str, list[str]] = {}
    for elem_name, elem_def in context.items():
        if isinstance(elem_def, dict) and "ref" in elem_def:
            superimpose_flag = elem_def.get("superimpose", True)
            if isinstance(superimpose_flag, str):
                superimpose_flag = superimpose_flag.lower() in {"t", "true", "1"}
            if bool(superimpose_flag):
                ref_name = elem_def["ref"]
                if ref_name in context and isinstance(context[ref_name], dict):
                    superimpositions.setdefault(ref_name, []).append(elem_name)

    return superimpositions


def convert_lattice(
    bmad_lattice_file_path: Path,
    environment_variables: dict | None = None,
    sanitize_names: bool | None = None,
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
        `segment.element_name` syntax to access the element in a segment. If `None`
        (default), a warning is raised for invalid names. Set to `True` to sanitise,
        or `False` to silence the warning.
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
    lines = fortran_namelist.read_clean_lines(resolved_lattice_file_path)

    # Merge multi-line statements
    merged_lines = fortran_namelist.merge_delimiter_continued_lines(
        lines, delimiter="&", remove_delimiter=True
    )
    merged_lines = fortran_namelist.merge_delimiter_continued_lines(
        merged_lines, delimiter=",", remove_delimiter=False
    )
    merged_lines = fortran_namelist.merge_delimiter_continued_lines(
        merged_lines, delimiter="{", remove_delimiter=False
    )
    assert len(merged_lines) <= len(
        lines
    ), "Merging lines should never produce more lines than there were before."

    # Parse the lattice file(s), i.e. basically execute them
    context = fortran_namelist.parse_lines(merged_lines)

    superimpositions = collect_superimpositions(context)

    # Convert the parsed lattice info to Cheetah elements
    return convert_element(
        name=context["__use__"],
        context=context,
        sanitize_name=sanitize_names,
        superimpositions=superimpositions,
        device=device,
        dtype=dtype,
    )
