import csv
from pathlib import Path

import torch

import cheetah


def translate_element(row: list[str], header: list[str]) -> dict | None:
    """
    Translate a row of an NX Tables file to a Cheetah `Element`.

    :param row: A row of an NX Tables file as a list of column elements.
    :param header: The header row of the NX Tables file as a list of column names.
    :return: Dictionary of Cheetah `Element` object best representing the row and its
        center s position if the element is relevant for the Cheetah model, `None`
        otherwise.
    """
    class_name = row[header.index("CLASS")]
    name = row[header.index("NAME")]
    s_position = float(row[header.index("Z_beam")])

    IGNORE_CLASSES = [
        "RSBG",
        "MSOB",
        "MSOH",
        "MSOG",
        "VVAG",
        "BSCL",
        "MIRA",
        "BAML",
        "SCRL",
        "TEMG",
        "FCNG",
        "SOLE",
        "EOLE",
        "MSOL",
        "BELS",
        "VVAF",
        "MIRM",
        "SCRY",
        "FPSA",
        "VPUL",
        "SOLC",
        "SCRE",
        "SOLX",
        "ICTB",
        "BSCS",
    ]
    if class_name in IGNORE_CLASSES:
        return None
    elif class_name == "MCXG":  # TODO: Check length with Willi
        assert name[6] == "X"
        horizontal_coil = cheetah.HorizontalCorrector(
            name=name[:6] + "H" + name[6 + 1 :], length=torch.tensor(5e-05)
        )
        vertical_coil = cheetah.VerticalCorrector(
            name=name[:6] + "V" + name[6 + 1 :], length=torch.tensor(5e-05)
        )
        element = cheetah.Segment(elements=[horizontal_coil, vertical_coil], name=name)
    elif class_name == "BSCX":
        element = cheetah.Screen(
            name=name,
            resolution=(2464, 2056),
            pixel_size=torch.tensor((0.00343e-3, 0.00247e-3)),
            binning=1,
        )
    elif class_name == "BSCR":
        element = cheetah.Screen(
            name=name,
            resolution=(2448, 2040),
            pixel_size=torch.tensor([3.5488e-6, 2.5003e-6]),
            binning=1,
        )
    elif class_name == "BSCM":
        element = cheetah.Screen(  # TODO: Ask for actual parameters
            name=name,
            resolution=(2448, 2040),
            pixel_size=torch.tensor([3.5488e-6, 2.5003e-6]),
            binning=1,
        )
    elif class_name == "BSCO":
        element = cheetah.Screen(  # TODO: Ask for actual parameters
            name=name,
            resolution=(2448, 2040),
            pixel_size=torch.tensor([3.5488e-6, 2.5003e-6]),
            binning=1,
        )
    elif class_name == "BSCA":
        element = cheetah.Screen(  # TODO: Ask for actual parameters
            name=name,
            resolution=(2448, 2040),
            pixel_size=torch.tensor([3.5488e-6, 2.5003e-6]),
            binning=1,
        )
    elif class_name == "BSCE":
        element = cheetah.Screen(  # TODO: Ask for actual parameters
            name=name,
            resolution=(2464, 2056),
            pixel_size=torch.tensor((0.00998e-3, 0.00715e-3)),
            binning=1,
        )
    elif class_name == "SCRD":
        element = cheetah.Screen(  # TODO: Ask for actual parameters
            name=name,
            resolution=(2464, 2056),
            pixel_size=torch.tensor((0.00998e-3, 0.00715e-3)),
            binning=1,
        )
    elif class_name == "BPMG":
        element = cheetah.BPM(name=name)
    elif class_name == "BPML":
        element = cheetah.BPM(name=name)
    elif class_name == "SLHG":
        element = cheetah.Aperture(  # TODO: Ask for actual size and shape
            name=name,
            x_max=torch.tensor(float("inf")),
            y_max=torch.tensor(float("inf")),
            shape="elliptical",
        )
    elif class_name == "SLHB":
        element = cheetah.Aperture(  # TODO: Ask for actual size and shape
            name=name,
            x_max=torch.tensor(float("inf")),
            y_max=torch.tensor(float("inf")),
            shape="rectangular",
        )
    elif class_name == "SLHS":
        element = cheetah.Aperture(  # TODO: Ask for actual size and shape
            name=name,
            x_max=torch.tensor(float("inf")),
            y_max=torch.tensor(float("inf")),
            shape="rectangular",
        )
    elif class_name == "MCHM":
        element = cheetah.HorizontalCorrector(name=name, length=torch.tensor(0.02))
    elif class_name == "MCVM":
        element = cheetah.VerticalCorrector(name=name, length=torch.tensor(0.02))
    elif class_name == "MBHL":
        element = cheetah.Dipole(name=name, length=torch.tensor(0.322))
    elif class_name == "MBHB":
        element = cheetah.Dipole(name=name, length=torch.tensor(0.22))
    elif class_name == "MBHO":
        element = cheetah.Dipole(
            name=name,
            length=torch.tensor(0.43852543421396856),
            angle=torch.tensor(0.8203047484373349),
            dipole_e2=torch.tensor(-0.7504915783575616),
        )
    elif class_name == "MQZM":
        element = cheetah.Quadrupole(name=name, length=torch.tensor(0.122))
    elif class_name == "RSBL":
        element = cheetah.Cavity(
            name=name,
            length=torch.tensor(4.139),
            frequency=torch.tensor(2.998e9),
            voltage=torch.tensor(76e6),
        )
    elif class_name == "RXBD":
        element = cheetah.Cavity(  # TODO: TD? and tilt?
            name=name,
            length=torch.tensor(1.0),
            frequency=torch.tensor(11.9952e9),
            voltage=torch.tensor(0.0),
        )
    elif class_name == "UNDA":  # TODO: Figure out actual length
        element = cheetah.Undulator(name=name, length=torch.tensor(0.25))
    elif class_name in [
        "SOLG",
        "BCMG",
        "EOLG",
        "SOLS",
        "EOLS",
        "SOLA",
        "EOLA",
        "SOLT",
        "BSTB",
        "TORF",
        "EOLT",
        "SOLO",
        "EOLO",
        "SOLB",
        "EOLB",
        "ECHA",
        "MKBB",
        "MKBE",
        "MKPM",
        "EOLC",
        "SOLM",
        "EOLM",
        "SOLH",
        "BSCD",
        "STDE",  # STRIDENAS detector
        "ECHS",  # STRIDENAS chamber
        "EOLH",
        "WINA",
        "LINA",
        "EOLX",
    ]:
        element = cheetah.Marker(name=name)
    else:
        raise ValueError(f"Encountered unknown class {class_name} for element {name}")

    return {"element": element, "s_position": s_position}


def convert_lattice_to_cheetah(filepath: Path) -> "cheetah.Element":
    """
    Read an NX Tables CSV-like file generated for the ARES lattice into a Cheetah
    `Segment`.

    :param filepath: Path to the NX Tables file.
    :return: Converted Cheetah `Segment`.
    """
    with open(filepath, "r") as csvfile:
        nx_tables_rows = csv.reader(csvfile, delimiter=",")
        nx_tables_rows = list(nx_tables_rows)

    header = nx_tables_rows[0]
    nx_tables_rows = nx_tables_rows[1:]

    translated = [translate_element(row, header) for row in nx_tables_rows]
    filtered = [element for element in translated if element is not None]

    # Sort by s position
    sorted_filtered = sorted(filtered, key=lambda x: x["s_position"])

    # Insert drift sections
    filled_with_drifts = [sorted_filtered[0]["element"]]
    for previous, current in zip(sorted_filtered[:-1], sorted_filtered[1:]):
        previous_length = (
            previous["element"].length
            if hasattr(previous["element"], "length")
            else 0.0
        )
        current_length = (
            current["element"].length if hasattr(current["element"], "length") else 0.0
        )

        center_to_center_distance = current["s_position"] - previous["s_position"]
        drift_length = (
            center_to_center_distance - previous_length / 2 - current_length / 2
        )

        assert drift_length >= 0.0, (
            f"Elements {previous['element'].name} and {current['element'].name} overlap"
            f" by {drift_length}."
        )

        if drift_length > 0.0:
            filled_with_drifts.append(
                cheetah.Drift(
                    name=f"DRIFT_{previous['element'].name}_{current['element'].name}",
                    length=torch.as_tensor([drift_length]),
                )
            )

        filled_with_drifts.append(current["element"])

    segment = cheetah.Segment(elements=filled_with_drifts, name=filepath.stem)

    # Return flattened because conversion prduces nested segments
    return segment.flattened()
