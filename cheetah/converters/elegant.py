import warnings
from pathlib import Path

import torch
from scipy.constants import physical_constants, speed_of_light

import cheetah
from cheetah.converters.utils.fortran_namelist import (
    merge_delimiter_continued_lines,
    parse_lines,
    read_clean_lines,
    validate_understood_properties,
)
from cheetah.utils import NoBeamPropertiesInLatticeWarning, UnknownElementWarning

electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


def convert_element(
    name: str,
    context: dict,
    sanitize_name: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> "cheetah.Element":
    """
    Convert a parsed Elegant element dict to a cheetah Element.

    :param name: Name of the (top-level) element to convert.
    :param context: Context dictionary parsed from Elegant lattice file(s).
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
        elif parsed["element_type"] in ["kick", "kicker"]:
            validate_understood_properties(
                shared_properties + ["l", "hkick", "vkick"], parsed
            )
            return cheetah.CombinedCorrector(
                length=torch.tensor(parsed.get("l", 0.0), **factory_kwargs),
                horizontal_angle=torch.tensor(
                    parsed.get("hkick", 0.0), **factory_kwargs
                ),
                vertical_angle=torch.tensor(parsed.get("vkick", 0.0), **factory_kwargs),
                name=name,
                sanitize_name=sanitize_name,
            )
        elif parsed["element_type"] in ["mark", "marker"]:
            validate_understood_properties(shared_properties, parsed)
            return cheetah.Marker(
                name=name, sanitize_name=sanitize_name, **factory_kwargs
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

            # Initially zero in Elegant by convention
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
            f"Unknown Elegant element type for {name = }"  # noqa: E202, E251
        )


def convert_lattice(
    elegant_lattice_file_path: Path,
    name: str,
    sanitize_names: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> "cheetah.Element":
    """
    Convert a Elegant lattice file to a Cheetah `Segment`.

    :param elegant_lattice_file_path: Path to the Elegant lattice file.
    :param name: Name of the root element.
    :param sanitize_names: Whether to sanitise the names of the elements as well as the
        name of the segment to be valid Python variable names. This is needed if you
        want to use the `segment.element_name` syntax to access the element in a
        segment.
    :param device: Device to use for the lattice. If `None`, the current default device
        of PyTorch is used.
    :param dtype: Data type to use for the lattice. If `None`, the current default dtype
        of PyTorch is used.
    :return: Cheetah `Segment` representing the Elegant lattice.
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


def convert_beam(
    file_path: Path,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Read the beam distribution from an Elegant SDDS file.

    :param file_path: Path to the SDDS file containing the Elegant beam distribution.
    :param device: Device to use for the beam distribution. If `None`, the current
        default device of PyTorch is used.
    :param dtype: Data type to use for the beam distribution. If `None`, the current
        default dtype of PyTorch is used.
    :return: A tuple containing the particles tensor, the reference energy in eV, and
        the tensor of particle charges.
    """
    try:
        import sdds
    except ImportError:
        raise ImportError(
            "The soliday.sdds package is required to convert Elegant beam. Please "
            "install it via pip (`pip install soliday.sdds`) and try again."
        )

    factory_kwargs = {
        "device": device or torch.get_default_device(),
        "dtype": dtype or torch.get_default_dtype(),
    }

    sdds_data = sdds.load(file_path.as_posix())

    is_elegant = sdds_data.columnName[:6] == ["x", "xp", "y", "yp", "t", "p"]
    is_spiffe = sdds_data.columnName[:6] == ["r", "pz", "pr", "pphi", "t", "q"]
    if is_spiffe:
        raise ValueError(
            "The beam distribution is stored in the spiffe format, which is not "
            "currently supported. Use spiffe2elegant to conver the beam first."
        )
    elif not is_elegant:
        raise ValueError(
            "The first six columns of the SDDS file do not match the expected Elegant "
            "beam convention. Please ensure the SDDS file is in the correct format."
        )

    # (6, num_pages, num_particles) -> (num_pages, num_particles, 6)
    elegant_coordinates = torch.tensor(
        sdds_data.columnData[:6], **factory_kwargs
    ).permute(1, 2, 0)

    # Check if reference momentum is provided in the SDDS file. If not, use the momentum
    # of the first particle as the reference momentum, which is the default reference
    # particle.
    p_central = (
        torch.tensor(sdds_data.getParameterValueList("pCentral"), **factory_kwargs)
        if "pCentral" in sdds_data.parameterName
        else elegant_coordinates[..., 0, 5]
    )
    reference_momentum_eV = p_central * electron_mass_eV  # Convert to eV

    cheetah_coordinates = elegant_to_cheetah_coordinates(elegant_coordinates, p_central)
    reference_energy_eV = (reference_momentum_eV**2 + electron_mass_eV**2).sqrt()

    # Add seventh column for Cheetah coordinates
    particles = cheetah_coordinates.new_zeros((*cheetah_coordinates.shape[:-1], 7))
    particles[..., :6] = cheetah_coordinates  # copy the first 6 columns
    particles[..., 6] = 1.0

    # Check whether charge is present in the SDDS file, otherwise default to 1
    particle_charges = (
        torch.tensor(sdds_data.getColumnValueLists("q"), **factory_kwargs)
        if "q" in sdds_data.columnName
        else torch.ones(particles.shape[:-1], **factory_kwargs)
    )

    return particles, reference_energy_eV, particle_charges


def elegant_to_cheetah_coordinates(
    elegant_coordinates: torch.Tensor, p_central: torch.Tensor
) -> torch.Tensor:
    r"""
    Convert Elegant coordinates to Cheetah coordinates.

    :param elegant_coordinates: Elegant coordinates of shape (..., num_particles, 6)
        with columns: [x, x', y, y', t, p].
    :param p_central: The reference momentum in :math:`\beta * \gamma` units.
    :return: Cheetah coordinates of shape (..., num_particles, 7).
    """
    reference_momentum_eV = p_central * electron_mass_eV
    reference_energy_eV = (reference_momentum_eV**2 + electron_mass_eV**2).sqrt()

    momentum_eV = elegant_coordinates[..., 5] * electron_mass_eV
    energy_eV = (momentum_eV**2 + electron_mass_eV**2).sqrt()
    delta_p = (
        elegant_coordinates[..., 5] - p_central.unsqueeze(-1)
    ) / p_central.unsqueeze(
        -1
    )  # (p - p0) / p0

    cheetah_coordinates = elegant_coordinates.new_ones(
        (*elegant_coordinates.shape[:-1], 7)
    )
    cheetah_coordinates[..., 0] = elegant_coordinates[..., 0]  # x
    cheetah_coordinates[..., 2] = elegant_coordinates[..., 2]  # y

    x_prime = elegant_coordinates[..., 1]
    y_prime = elegant_coordinates[..., 3]
    cheetah_coordinates[..., 1] = (
        x_prime * (1.0 + delta_p) / (1.0 + x_prime.square() + y_prime.square()).sqrt()
    )  # px = P_x / p_0
    cheetah_coordinates[..., 3] = (
        y_prime * (1.0 + delta_p) / (1.0 + x_prime.square() + y_prime.square().sqrt())
    )

    cheetah_coordinates[..., 4] = (
        elegant_coordinates[..., 4] * speed_of_light
    )  # \tau = c * \Delta t
    cheetah_coordinates[..., 5] = (
        energy_eV - reference_energy_eV
    ) / reference_momentum_eV  # pz = P_z / p_0

    return cheetah_coordinates
