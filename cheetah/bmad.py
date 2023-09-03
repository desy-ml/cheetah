import os
import re
from pathlib import Path


def read_clean_lines(lattice_file_path: Path) -> list[str]:
    """
    Recursevely read lines from Bmad lattice files, removing comments and empty lines,
    and replacing lines calling external files with the lines of the external file.

    :param lattice_file_path: Path to the root Bmad lattice file.
    :return: List of lines from the root Bmad lattice file and all external files.
    """
    with open(lattice_file_path) as f:
        lines = f.readlines()

    # Remove comments and empty lines
    lines = [line.strip() for line in lines]
    # Remove comments (i.e. all characters after a '!')
    lines = [re.sub(r"!.*", "", line) for line in lines]
    # Remove empty lines
    lines = [line for line in lines if line]

    # Replace lines calling external files with the lines of the external file
    replaced_lines = []
    for i, line in enumerate(lines):
        if line.startswith("call, file ="):
            external_file_path = Path(line.split("=")[1].strip())
            resolved_external_file_path = Path(
                *[
                    os.environ[part[1:]] if part.startswith("$") else part
                    for part in external_file_path.parts
                ]
            )
            if not resolved_external_file_path.is_absolute():
                resolved_external_file_path = (
                    lattice_file_path.parent / resolved_external_file_path
                )
            external_file_lines = read_clean_lines(resolved_external_file_path)
            replaced_lines += external_file_lines
        else:
            replaced_lines.append(line)

    # Make lines all lower case (done late because environment variables are case
    # sensitive)
    replaced_lines = [line.lower() for line in replaced_lines]

    return replaced_lines
