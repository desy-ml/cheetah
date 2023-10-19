import csv
from pathlib import Path

import torch

import cheetah


def read_nx_tables(filepath: Path) -> "cheetah.Element":
    """
    Read an NX Tables CSV-like file generated for the ARES lattice into a Cheetah
    `Segment`.

    :param filepath: Path to the NX Tables file.
    :return: Converted Cheetah `Segment`.
    """
    with open(filepath, "r") as csvfile:
        nx_tables_rows = csv.reader(csvfile, delimiter=",")

    return nx_tables_rows
