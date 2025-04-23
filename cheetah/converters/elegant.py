from pathlib import Path

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
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> "cheetah.Element":
    """Convert a parsed elegant element dict to a cheetah Element.

    :param name: Name of the (top-level) element to convert.
    :param context: Context dictionary parsed from elegant lattice file(s).
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
    parsed = context[name]

    if isinstance(parsed, list):
        return cheetah.Segment(
            elements=[
                convert_element(element_name, context, device, dtype)
                for element_name in parsed
            ],
            name=name,
        )
    elif isinstance(parsed, dict) and "element_type" in parsed:
        if parsed["element_type"] == "sole":
            # The group property does not have an analoge in Cheetah, so it is neglected
            validate_understood_properties(["element_type", "l", "group"], parsed)
            return cheetah.Solenoid(
                length=torch.tensor(parsed["l"], **factory_kwargs), name=name
            )
        elif parsed["element_type"] in ["hkick", "hkic"]:
            validate_understood_properties(
                ["element_type", "l", "kick", "group"], parsed
            )
            return cheetah.HorizontalCorrector(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                angle=torch.tensor(parsed.get("kick", 0.0), **factory_kwargs),
                name=name,
            )
        elif parsed["element_type"] in ["vkick", "vkic"]:
            validate_understood_properties(
                ["element_type", "l", "kick", "group"], parsed
            )
            return cheetah.VerticalCorrector(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                angle=torch.tensor(parsed.get("kick", 0.0), **factory_kwargs),
                name=name,
            )
        elif parsed["element_type"] in ["mark", "marker"]:
            validate_understood_properties(["element_type", "group"], parsed)
            return cheetah.Marker(name=name)
        elif parsed["element_type"] == "kick":
            validate_understood_properties(["element_type", "l", "group"], parsed)

            # TODO Find proper element class
            return cheetah.Drift(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs), name=name
            )
        elif parsed["element_type"] in ["drift", "drif"]:
            validate_understood_properties(["element_type", "l", "group"], parsed)
            return cheetah.Drift(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs), name=name
            )
        elif parsed["element_type"] in ["csrdrift", "csrdrif"]:
            # Drift that includes effects from coherent synchrotron radiation
            validate_understood_properties(
                ["element_type", "l", "group", "use_stupakov", "n_kicks", "csr"], parsed
            )
            return cheetah.Drift(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs), name=name
            )
        elif parsed["element_type"] in ["lscdrift", "lscdrif"]:
            # Drift that includes space charge effects
            validate_understood_properties(
                [
                    "element_type",
                    "l",
                    "group",
                    "interpolate",
                    "smoothing",
                    "bins",
                    "high_frequency_cutoff0",
                    "high_frequency_cutoff1",
                    "lsc",
                ],
                parsed,
            )
            return cheetah.Drift(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs), name=name
            )
        elif parsed["element_type"] == "ecol":
            validate_understood_properties(
                ["element_type", "l", "x_max", "y_max"],
                parsed,
            )
            return cheetah.Segment(
                elements=[
                    cheetah.Drift(
                        length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                        name=name + "_drift",
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
                    ),
                ],
                name=name + "_segment",
            )
        elif parsed["element_type"] == "rcol":
            validate_understood_properties(
                ["element_type", "l", "x_max", "y_max"],
                parsed,
            )
            return cheetah.Segment(
                elements=[
                    cheetah.Drift(
                        length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                        name=name + "_drift",
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
                    ),
                ],
                name=name + "_segment",
            )
        elif parsed["element_type"] in ["quad", "quadrupole"]:
            validate_understood_properties(
                ["element_type", "l", "k1", "tilt", "group"],
                parsed,
            )
            return cheetah.Quadrupole(
                length=torch.tensor(parsed["l"], **factory_kwargs),
                k1=torch.tensor(parsed["k1"], **factory_kwargs),
                tilt=torch.tensor(parsed.get("tilt", 0.0), **factory_kwargs),
                name=name,
            )
        elif parsed["element_type"] == "moni":
            validate_understood_properties(["element_type", "group", "l"], parsed)
            if "l" in parsed:
                return cheetah.Segment(
                    elements=[
                        cheetah.Drift(
                            length=torch.tensor(parsed["l"] / 2, **factory_kwargs),
                            name=name + "_predrift",
                        ),
                        cheetah.BPM(name=name),
                        cheetah.Drift(
                            length=torch.tensor(parsed["l"] / 2, **factory_kwargs),
                            name=name + "_postdrift",
                        ),
                    ],
                    name=name + "_segment",
                )
            else:
                return cheetah.BPM(name=name)
        elif parsed["element_type"] == "ematrix":
            validate_understood_properties(
                ["element_type", "l", "order", "c[1-6]", "r[1-6][1-6]", "group"],
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
                [parsed.get(f"c{i + 1}", 0.0) for i in range(6)],
                **factory_kwargs,
            )

            return cheetah.CustomTransferMap(
                length=torch.tensor(parsed["l"], **factory_kwargs),
                predefined_transfer_map=R,
                name=name,
            )
        elif parsed["element_type"] == "rfca":
            validate_understood_properties(
                [
                    "element_type",
                    "l",
                    "phase",
                    "volt",
                    "freq",
                    "change_p0",
                    "end1_focus",
                    "end2_focus",
                    "body_focus_model",
                    "group",
                ],
                parsed,
            )

            # TODO Properly handle all parameters
            return cheetah.Cavity(
                length=torch.tensor(parsed["l"], **factory_kwargs),
                # Elegant defines 90° as the phase of maximum acceleration,
                # while Cheetah uses 0°. We therefore add a phase offset to compensate.
                phase=torch.tensor(parsed["phase"] - 90, **factory_kwargs),
                voltage=torch.tensor(parsed["volt"], **factory_kwargs),
                frequency=torch.tensor(parsed["freq"], **factory_kwargs),
                name=name,
            )
        elif parsed["element_type"] == "rfcw":
            validate_understood_properties(
                [
                    "element_type",
                    "l",
                    "phase",
                    "volt",
                    "freq",
                    "change_p0",
                    "end1_focus",
                    "end2_focus",
                    "cell_length",
                    "zwakefile",
                    "trwakefile",
                    "tcolumn",
                    "wxcolumn",
                    "wycolumn",
                    "wzcolumn",
                    "interpolate",
                    "n_kicks",
                    "smoothing",
                    "zwake",
                    "trwake",
                    "lsc",
                    "lsc_bins",
                    "lsc_high_frequency_cutoff0",
                    "lsc_high_frequency_cutoff1",
                    "group",
                ],
                parsed,
            )

            # TODO Properly handle all parameters
            return cheetah.Cavity(
                length=torch.tensor(parsed["l"], **factory_kwargs),
                # Elegant defines 90° as the phase of maximum acceleration,
                # while Cheetah uses 0°. We therefore add a phase offset to compensate.
                phase=torch.tensor(parsed["phase"] - 90, **factory_kwargs),
                voltage=torch.tensor(parsed["volt"], **factory_kwargs),
                frequency=torch.tensor(parsed["freq"], **factory_kwargs),
                name=name,
            )
        elif parsed["element_type"] == "rfdf":
            validate_understood_properties(
                [
                    "element_type",
                    "l",
                    "phase",
                    "voltage",
                    "frequency",
                    "group",
                ],
                parsed,
            )

            # TODO Properly handle all parameters
            return cheetah.TransverseDeflectingCavity(
                length=torch.tensor(parsed["l"], **factory_kwargs),
                # Elegant defines 90° as the phase of maximum acceleration,
                # while Cheetah uses 0°. We therefore add a phase offset to compensate.
                phase=torch.tensor(parsed["phase"] - 90, **factory_kwargs),
                voltage=torch.tensor(parsed["voltage"], **factory_kwargs),
                frequency=torch.tensor(parsed["frequency"], **factory_kwargs),
                name=name,
            )
        elif parsed["element_type"] in ["sben", "csbend"]:
            validate_understood_properties(
                ["element_type", "l", "angle", "k1", "e1", "e2", "tilt", "group"],
                parsed,
            )
            return cheetah.Dipole(
                length=torch.tensor(parsed["l"], **factory_kwargs),
                angle=torch.tensor(parsed.get("angle", 0.0), **factory_kwargs),
                k1=torch.tensor(parsed.get("k1", 0.0), **factory_kwargs),
                dipole_e1=torch.tensor(parsed.get("e1", 0.0), **factory_kwargs),
                dipole_e2=torch.tensor(parsed.get("e2", 0.0), **factory_kwargs),
                tilt=torch.tensor(parsed.get("tilt", 0.0), **factory_kwargs),
                name=name,
            )
        elif parsed["element_type"] == "rben":
            validate_understood_properties(
                ["element_type", "l", "angle", "e1", "e2", "tilt", "group"],
                parsed,
            )
            return cheetah.RBend(
                length=torch.tensor(parsed["l"], **factory_kwargs),
                angle=torch.tensor(parsed.get("angle", 0.0), **factory_kwargs),
                rbend_e1=torch.tensor(parsed.get("e1", 0.0), **factory_kwargs),
                rbend_e2=torch.tensor(parsed.get("e2", 0.0), **factory_kwargs),
                tilt=torch.tensor(parsed.get("tilt", 0.0), **factory_kwargs),
                name=name,
            )
        elif parsed["element_type"] == "csrcsben":
            validate_understood_properties(
                [
                    "element_type",
                    "l",
                    "angle",
                    "e1",
                    "e2",
                    "edge1_effects",
                    "edge2_effects",
                    "tilt",
                    "hgap",
                    "fint",
                    "sg_halfwidth",
                    "sg_order",
                    "steady_state",
                    "bins",
                    "n_kicks",
                    "integration_order",
                    "isr",
                    "csr",
                    "group",
                ],
                parsed,
            )
            return cheetah.Dipole(
                length=torch.tensor(parsed["l"], **factory_kwargs),
                angle=torch.tensor(parsed.get("angle", 0.0), **factory_kwargs),
                k1=torch.tensor(parsed.get("k1", 0.0), **factory_kwargs),
                dipole_e1=torch.tensor(parsed.get("e1", 0.0), **factory_kwargs),
                dipole_e2=torch.tensor(parsed.get("e2", 0.0), **factory_kwargs),
                tilt=torch.tensor(parsed.get("tilt", 0.0), **factory_kwargs),
                name=name,
            )
        elif parsed["element_type"] == "watch":
            validate_understood_properties(
                ["element_type", "group", "filename"], parsed
            )
            return cheetah.Marker(name=name)
        elif parsed["element_type"] in ["charge", "wake"]:
            print(
                f"WARNING: Information provided in element {name} of type"
                f" {parsed['element_type']} cannot be imported automatically. Consider"
                " manually providing the correct information."
            )
            return cheetah.Marker(name=name)
        else:
            print(
                f"WARNING: Element {name} of type {parsed['element_type']} cannot"
                " be converted correctly. Using drift section instead."
            )
            # TODO: Remove the length if by adding markers to Cheetah
            return cheetah.Drift(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs), name=name
            )
    else:
        raise ValueError(
            f"Unknown elegant element type for {name = }"  # noqa: E202, E251
        )


def convert_lattice_to_cheetah(
    elegant_lattice_file_path: Path,
    name: str,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> "cheetah.Element":
    """
    Convert a elegant lattice file to a Cheetah `Segment`.

    :param elegant_lattice_file_path: Path to the elegant lattice file.
    :param name: Name of the root element.
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

    # Strip EOL char (;) from final merged lines
    stripped_lines = [line.strip(";") for line in merged_lines]

    # Parse the lattice file(s), i.e. basically execute them
    context = parse_lines(stripped_lines)

    # Convert the parsed lattice info to Cheetah elements
    return convert_element(name, context, device, dtype)
