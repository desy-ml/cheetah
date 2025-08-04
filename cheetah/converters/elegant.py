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
from cheetah.utils import NoBeamPropertiesInLatticeWarning, UnknownElementWarning


def convert_element(
    name: str,
    context: dict,
    sanitize_name: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> "cheetah.Element":
    """Convert a parsed elegant element dict to a cheetah Element.

    :param name: Name of the (top-level) element to convert.
    :param context: Context dictionary parsed from elegant lattice file(s).
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment.
    :param device: Device to use for the lattice. If `None`, the current default device
        of PyTorch is used.
    :param dtype: Data type to use for the lattice. If `None`, the current default dtype
        of PyTorch is used.
    :return: Converted cheetah Element. If you are calling this function yourself
        as a user of Cheetah, this is most likely a `Segment`.
    """
    factory_kwargs = {
        "device": device or torch.get_default_device(),
        "dtype": dtype or torch.get_default_dtype(),
    }

    is_reversed_line = name.startswith("-")
    name = name.removeprefix("-")

    parsed = context[name]

    shared_properties = ["element_type", "group"]

    if isinstance(parsed, list):
        segment = cheetah.Segment(
            elements=[
                convert_element(element_name, context, sanitize_name, device, dtype)
                for element_name in parsed
            ],
            name=name,
            sanitize_name=sanitize_name,
        )
        return segment if not is_reversed_line else segment.reversed()
    elif isinstance(parsed, dict) and "element_type" in parsed:
        if parsed["element_type"] == "sole":
            # The group property does not have an analoge in Cheetah, so it is neglected
            validate_understood_properties(shared_properties + ["l"], parsed)
            return cheetah.Solenoid(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] in ["hkick", "hkic"]:
            validate_understood_properties(shared_properties + ["l", "kick"], parsed)
            return cheetah.HorizontalCorrector(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                angle=torch.tensor(parsed.get("kick", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] in ["vkick", "vkic"]:
            validate_understood_properties(shared_properties + ["l", "kick"], parsed)
            return cheetah.VerticalCorrector(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                angle=torch.tensor(parsed.get("kick", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] in ["mark", "marker"]:
            validate_understood_properties(shared_properties, parsed)
            return cheetah.Marker(
                name=name, sanitize_name=sanitize_name, **factory_kwargs
            )
        elif parsed["element_type"] == "kick":
            validate_understood_properties(shared_properties + ["l"], parsed)

            # TODO Find proper element class
            return cheetah.Drift(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] in ["drift", "drif"]:
            validate_understood_properties(shared_properties + ["l"], parsed)
            return cheetah.Drift(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] in ["csrdrift", "csrdrif"]:
            # Drift that includes effects from coherent synchrotron radiation
            validate_understood_properties(shared_properties + ["l"], parsed)
            return cheetah.Drift(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] in ["lscdrift", "lscdrif"]:
            # Drift that includes space charge effects
            validate_understood_properties(shared_properties + ["l"], parsed)
            return cheetah.Drift(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] == "ecol":
            validate_understood_properties(
                shared_properties + ["l", "x_max", "y_max"], parsed
            )
            return cheetah.Segment(
                elements=[
                    cheetah.Drift(
                        length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                        name=name + "_drift",
                        sanitize_name=sanitize_name,
                    ),
                    cheetah.Aperture(
                        x_max=torch.tensor(
                            parsed.get("x_max", torch.inf), **factory_kwargs
                        ),
                        y_max=torch.tensor(
                            parsed.get("y_max", torch.inf), **factory_kwargs
                        ),
                        shape="elliptical",
                        name=name + "_aperture",
                        sanitize_name=sanitize_name,
                    ),
                ],
                name=name + "_segment",
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] == "rcol":
            validate_understood_properties(
                shared_properties + ["l", "x_max", "y_max"], parsed
            )
            return cheetah.Segment(
                elements=[
                    cheetah.Drift(
                        length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                        name=name + "_drift",
                        sanitize_name=sanitize_name,
                    ),
                    cheetah.Aperture(
                        x_max=torch.tensor(
                            parsed.get("x_max", torch.inf), **factory_kwargs
                        ),
                        y_max=torch.tensor(
                            parsed.get("y_max", torch.inf), **factory_kwargs
                        ),
                        shape="rectangular",
                        name=name + "_aperture",
                        sanitize_name=sanitize_name,
                    ),
                ],
                name=name + "_segment",
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] in ["quad", "quadrupole", "kquad"]:
            validate_understood_properties(
                shared_properties + ["l", "k1", "tilt"],
                parsed,
            )
            return cheetah.Quadrupole(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                k1=torch.tensor(parsed.get("k1", 0.0), **factory_kwargs),
                tilt=torch.tensor(parsed.get("tilt", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] in ["sext", "sextupole"]:
            validate_understood_properties(
                shared_properties + ["l", "k2", "tilt"],
                parsed,
            )
            return cheetah.Sextupole(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                k2=torch.tensor(parsed.get("k2", 0.0), **factory_kwargs),
                tilt=torch.tensor(parsed.get("tilt", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] == "moni":
            validate_understood_properties(shared_properties + ["l"], parsed)
            if "l" in parsed:
                return cheetah.Segment(
                    elements=[
                        cheetah.Drift(
                            length=torch.tensor(
                                parsed.get("l", 0.0) / 2, **factory_kwargs
                            ),
                            name=name + "_predrift",
                            sanitize_name=sanitize_name,
                        ),
                        cheetah.BPM(name=name, sanitize_name=sanitize_name),
                        cheetah.Drift(
                            length=torch.tensor(
                                parsed.get("l", 0.0) / 2, **factory_kwargs
                            ),
                            name=name + "_postdrift",
                            sanitize_name=sanitize_name,
                        ),
                    ],
                    name=name + "_segment",
                    sanitize_name=sanitize_name,
                )
            else:
                return cheetah.BPM(name=name, sanitize_name=sanitize_name)
        elif parsed["element_type"] == "ematrix":
            validate_understood_properties(
                shared_properties + ["l", "order", "c[1-6]", "r[1-6][1-6]"],
                parsed,
            )

            if parsed.get("order", 1) != 1:
                raise ValueError("Only first order modelling is supported")

            # Initially zero in elegant by convention
            R = torch.zeros((7, 7), **factory_kwargs)
            # Add linear component
            R[:6, :6] = torch.tensor(
                [
                    [parsed.get(f"r{i + 1}{j + 1}", 0.0) for j in range(6)]
                    for i in range(6)
                ],
                **factory_kwargs,
            )
            # Add affine component (constant offset)
            R[:6, 6] = torch.tensor(
                [parsed.get(f"c{i + 1}", 0.0) for i in range(6)], **factory_kwargs
            )
            # Ensure the affine component is passed along
            R[6, 6] = 1.0

            return cheetah.CustomTransferMap(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                predefined_transfer_map=R,
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] == "rfca":
            validate_understood_properties(
                shared_properties + ["l", "phase", "volt", "freq"], parsed
            )
            return cheetah.Cavity(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                # Elegant defines 90° as the phase of maximum acceleration,
                # while Cheetah uses 0°. We therefore add a phase offset to compensate.
                phase=torch.tensor(parsed.get("phase", 0.0) - 90, **factory_kwargs),
                voltage=torch.tensor(parsed.get("volt", 0.0), **factory_kwargs),
                frequency=torch.tensor(parsed.get("freq", 500e6), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] == "rfcw":
            validate_understood_properties(
                shared_properties + ["l", "phase", "volt", "freq"], parsed
            )
            return cheetah.Cavity(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                # Elegant defines 90° as the phase of maximum acceleration,
                # while Cheetah uses 0°. We therefore add a phase offset to compensate.
                phase=torch.tensor(parsed.get("phase", 0.0) - 90, **factory_kwargs),
                voltage=torch.tensor(parsed.get("volt", 0.0), **factory_kwargs),
                frequency=torch.tensor(parsed.get("freq", 500e6), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] == "rfdf":
            validate_understood_properties(
                shared_properties + ["l", "phase", "voltage", "freq"], parsed
            )
            return cheetah.TransverseDeflectingCavity(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                # Elegant defines 90° as the phase of maximum acceleration,
                # while Cheetah uses 0°. We therefore add a phase offset to compensate.
                phase=torch.tensor(parsed.get("phase", 0.0) - 90, **factory_kwargs),
                voltage=torch.tensor(parsed.get("voltage", 0.0), **factory_kwargs),
                frequency=torch.tensor(parsed.get("freq", 2.856e9), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] in ["sben", "csbend"]:
            validate_understood_properties(
                shared_properties + ["l", "angle", "k1", "e1", "e2", "tilt"],
                parsed,
            )
            return cheetah.Dipole(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                angle=torch.tensor(parsed.get("angle", 0.0), **factory_kwargs),
                k1=torch.tensor(parsed.get("k1", 0.0), **factory_kwargs),
                dipole_e1=torch.tensor(parsed.get("e1", 0.0), **factory_kwargs),
                dipole_e2=torch.tensor(parsed.get("e2", 0.0), **factory_kwargs),
                tilt=torch.tensor(parsed.get("tilt", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] == "rben":
            validate_understood_properties(
                shared_properties + ["l", "angle", "e1", "e2", "tilt"],
                parsed,
            )
            return cheetah.RBend(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                angle=torch.tensor(parsed.get("angle", 0.0), **factory_kwargs),
                rbend_e1=torch.tensor(parsed.get("e1", 0.0), **factory_kwargs),
                rbend_e2=torch.tensor(parsed.get("e2", 0.0), **factory_kwargs),
                tilt=torch.tensor(parsed.get("tilt", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] in ["csrcsben", "csrcsbend"]:
            validate_understood_properties(
                shared_properties + ["l", "angle", "k1", "e1", "e2", "tilt"],
                parsed,
            )
            return cheetah.Dipole(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                angle=torch.tensor(parsed.get("angle", 0.0), **factory_kwargs),
                k1=torch.tensor(parsed.get("k1", 0.0), **factory_kwargs),
                dipole_e1=torch.tensor(parsed.get("e1", 0.0), **factory_kwargs),
                dipole_e2=torch.tensor(parsed.get("e2", 0.0), **factory_kwargs),
                tilt=torch.tensor(parsed.get("tilt", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] == "watch":
            validate_understood_properties(shared_properties + ["filename"], parsed)
            return cheetah.Marker(
                name=name, sanitize_name=sanitize_name, **factory_kwargs
            )
        elif parsed["element_type"] in ["charge", "wake"]:
            warnings.warn(
                f"Information provided in element {name} of type"
                f" {parsed['element_type']} cannot be imported automatically. Consider"
                " manually providing the correct information.",
                category=NoBeamPropertiesInLatticeWarning,
                stacklevel=2,
            )
            return cheetah.Marker(
                name=name, sanitize_name=sanitize_name, **factory_kwargs
            )
        else:
            warnings.warn(
                f"Element {name} of type {parsed['element_type']} cannot be converted "
                "correctly. Using drift section instead.",
                category=UnknownElementWarning,
                stacklevel=2,
            )
            return cheetah.Drift(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
    else:
        raise ValueError(
            f"Unknown elegant element type for {name = }"  # noqa: E202, E251
        )


def convert_lattice_to_cheetah(
    elegant_lattice_file_path: Path,
    name: str,
    sanitize_names: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> "cheetah.Element":
    """
    Convert a elegant lattice file to a Cheetah `Segment`.

    :param elegant_lattice_file_path: Path to the elegant lattice file.
    :param name: Name of the root element.
    :param sanitize_names: Whether to sanitise the names of the elements as well as the
        name of the segment to be valid Python variable names. This is needed if you
        want to use the `segment.element_name` syntax to access the element in a
        segment.
    :param device: Device to use for the lattice. If `None`, the current default device
        of PyTorch is used.
    :param dtype: Data type to use for the lattice. If `None`, the current default dtype
        of PyTorch is used.
    :return: Cheetah `Segment` representing the elegant lattice.
    """

    # Read and clean the lattice file(s)
    lines = read_clean_lines(elegant_lattice_file_path)

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
    return convert_element(name, context, sanitize_names, device, dtype)
