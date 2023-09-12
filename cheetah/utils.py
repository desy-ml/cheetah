import json
from typing import Optional

import numpy as np
import torch
from scipy.constants import physical_constants

import cheetah

# Electron mass in eV
electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


def from_astrabeam(path: str) -> tuple[np.ndarray, float]:
    """
    Read from a ASTRA beam distribution, and prepare for conversion to a Cheetah
    ParticleBeam or ParameterBeam.

    Adapted from the implementation in ocelot:
    https://github.com/ocelot-collab/ocelot/blob/master/ocelot/adaptors/astra2ocelot.py

    :param path: Path to the ASTRA beam distribution file.
    :return: Particle 6D phase space information and mean energy of the particle beam.
    """
    P0 = np.loadtxt(path)

    # remove lost particles
    inds = np.argwhere(P0[:, 9] > 0)
    inds = inds.reshape(inds.shape[0])

    P0 = P0[inds, :]
    n_particles = P0.shape[0]

    # s_ref = P0[0, 2]
    Pref = P0[0, 5]

    xp = P0[:, :6]
    xp[0, 2] = 0.0
    xp[0, 5] = 0.0

    gamref = np.sqrt((Pref / electron_mass_eV) ** 2 + 1)
    # energy in eV: E = gamma * m_e
    energy = gamref * electron_mass_eV

    n_particles = xp.shape[0]
    particles = np.zeros((n_particles, 6))

    u = np.c_[xp[:, 3], xp[:, 4], xp[:, 5] + Pref]
    gamma = np.sqrt(1 + np.sum(u * u, 1) / electron_mass_eV**2)
    beta = np.sqrt(1 - gamma**-2)
    betaref = np.sqrt(1 - gamref**-2)

    p0 = np.linalg.norm(u, 2, 1).reshape((n_particles, 1))

    u = u / p0
    cdt = -xp[:, 2] / (beta * u[:, 2])
    particles[:, 0] = xp[:, 0] + beta * u[:, 0] * cdt
    particles[:, 2] = xp[:, 1] + beta * u[:, 1] * cdt
    particles[:, 4] = cdt
    particles[:, 1] = xp[:, 3] / Pref
    particles[:, 3] = xp[:, 4] / Pref
    particles[:, 5] = (gamma / gamref - 1) / betaref

    return particles, energy


def ocelot2cheetah(element, warnings: bool = True) -> "cheetah.Element":
    """
    Translate an Ocelot element to a Cheetah element.

    NOTE Object not supported by Cheetah are translated to drift sections. Screen
    objects are created only from `ocelot.Monitor` objects when the string "BSC" is
    contained in their `id` attribute. Their screen properties are always set to default
    values and most likely need adjusting afterwards. BPM objects are only created from
    `ocelot.Monitor` objects when their id has a substring "BPM".

    :param element: Ocelot element object representing an element of particle
        accelerator.
    :param warnings: Whether to print warnings when elements might not be converted as
        expected.
    :return: Cheetah element object representing an element of particle accelerator.
    """
    try:
        import ocelot
    except ImportError:
        raise ImportError(
            """To use the ocelot2cheetah lattice converter, Ocelot must be first
        installed, see https://github.com/ocelot-collab/ocelot """
        )

    if isinstance(element, ocelot.Drift):
        return cheetah.Drift(torch.tensor(element.l), name=element.id)
    elif isinstance(element, ocelot.Quadrupole):
        return cheetah.Quadrupole(
            torch.tensor(element.l), torch.tensor(element.k1), name=element.id
        )
    elif isinstance(element, ocelot.Hcor):
        return cheetah.HorizontalCorrector(
            torch.tensor(element.l), torch.tensor(element.angle), name=element.id
        )
    elif isinstance(element, ocelot.Vcor):
        return cheetah.VerticalCorrector(
            torch.tensor(element.l), torch.tensor(element.angle), name=element.id
        )
    elif isinstance(element, ocelot.Cavity):
        return cheetah.Cavity(torch.tensor(element.l), name=element.id)
    elif isinstance(element, ocelot.Monitor) and ("BSC" in element.id):
        # NOTE This pattern is very specific to ARES and will need a more complex
        # solution for other accelerators
        if warnings:
            print(
                "WARNING: Diagnostic screen was converted with default screen"
                " properties."
            )
        return cheetah.Screen(
            torch.tensor([2448, 2040]),
            torch.tensor([3.5488e-6, 2.5003e-6]),
            name=element.id,
        )
    elif isinstance(element, ocelot.Monitor) and "BPM" in element.id:
        return cheetah.BPM(name=element.id)
    elif isinstance(element, ocelot.Undulator):
        return cheetah.Undulator(torch.tensor(element.l), name=element.id)
    else:
        if warnings:
            print(
                f"WARNING: Unknown element {element.id} of type {type(element)},"
                " replacing with drift section."
            )
        return cheetah.Drift(torch.tensor(element.l), name=element.id)


def subcell_of_ocelot(cell: list, start: str, end: str) -> list:
    """Extract a subcell `[start, end]` from an Ocelot cell."""
    subcell = []
    is_in_subcell = False
    for el in cell:
        if el.id == start:
            is_in_subcell = True
        if is_in_subcell:
            subcell.append(el)
        if el.id == end:
            break

    return subcell


# Saving Cheetah to JSON
def parse_cheetah_element(element: cheetah.Element):
    """Get information from cheetah element for saving

    :param element: Cheetah element
    :return: Tuple of element name, element class, and element parameters
    """
    element_name = element.name
    if isinstance(element, cheetah.Drift):
        element_class = "Drift"
        params = {"length": element.length}
    elif isinstance(element, cheetah.Dipole):
        element_class = "Dipole"
        params = {
            "length": element.length,
            "angle": element.angle,
            "e1": element.e1,
            "e2": element.e2,
            "gap": element.gap,
            "tilt": element.tilt,
            "fint": element.fringe_integral,
            "fintx": element.fringe_integral,
        }
    elif isinstance(element, cheetah.Quadrupole):
        element_class = "Quadrupole"
        params = {
            "length": element.length,
            "k1": element.k1,
            "misalignment": element.misalignment,
            "tilt": element.tilt,
        }
    elif isinstance(element, cheetah.HorizontalCorrector):
        element_class = "HorizontalCorrector"
        params = {"length": element.length, "angle": element.angle}
    elif isinstance(element, cheetah.VerticalCorrector):
        element_class = "VerticalCorrector"
        params = {"length": element.length, "angle": element.angle}
    elif isinstance(element, cheetah.Cavity):
        element_class = "Cavity"
        params = {
            "length": element.length,
            "voltage": element.voltage,
            "phase": element.phase,
        }
    elif isinstance(element, cheetah.BPM):
        element_class = "BPM"
        params = {}
    elif isinstance(element, cheetah.Marker):
        element_class = "Marker"
        params = {}
    elif isinstance(element, cheetah.Screen):
        element_class = "Screen"
        params = {
            "resolution": element.resolution,
            "pixel_size": element.pixel_size,
            "binning": element.binning,
            "misalignment": element.misalignment,
        }
    elif isinstance(element, cheetah.Aperture):
        element_class = "Aperture"
        params = {"x_max": element.x_max, "y_max": element.y_max, "type": element.shape}
    elif isinstance(element, cheetah.Solenoid):
        element_class = "Solenoid"
        params = {
            "length": element.length,
            "k": element.k,
            "misalignment": element.misalignment,
        }
    elif isinstance(element, cheetah.Undulator):
        element_class = "Undulator"
        params = {"length": element.length}
    else:
        print(element)
        raise ValueError("Element type not supported")

    return element_name, element_class, params


def save_cheetah_model(
    segment: cheetah.Segment, fname: str, metadata: Optional[dict] = None
):
    """Save a cheetah model to json file accoding to the lattice-json convention
    c.f. https://github.com/nobeam/latticejson

    :param segment: Cheetah segment to save
    :param fname: Filename to save to
    :param metadata: Metadata for the saved lattice, by default {}
    """
    if metadata is None:
        metadata = {
            "version": "1.0",
            "title": "Test Lattice",
            "info": "This is a placeholder lattice description",
            "root": "cell",
        }
    lattice_dict = metadata.copy()

    # Get elements
    cell = []
    elements = {}
    for element in segment.elements:
        element_name, element_class, params = parse_cheetah_element(element)
        elements[element_name] = [element_class, params]
        cell.append(element_name)
    lattice_dict["elements"] = elements
    lattice_dict["lattices"] = {
        "cell": cell,
    }
    with open(fname, "w") as f:
        s = json.dumps(lattice_dict, cls=CompactJSONEncoder, indent=4)
        f.write(s)
        # json.dump(lattice_dict, f, indent=4, separators=(",", ': '))


# taken from https://github.com/nobeam/latticejson/blob/main/latticejson/format.py
class CompactJSONEncoder(json.JSONEncoder):
    """A JSON Encoder which only indents the first two levels."""

    def encode(self, obj, level=0):
        if isinstance(obj, dict) and level < 2:
            items_indent = (level + 1) * self.indent * " "
            items_string = ",\n".join(
                f"{items_indent}{json.dumps(key)}: {self.encode(value, level=level+1)}"
                for key, value in obj.items()
            )
            dict_indent = level * self.indent * " "
            newline = "\n" if level == 0 else ""
            return f"{{\n{items_string}\n{dict_indent}}}{newline}"
        else:
            return json.dumps(obj)


# Loading Cheetah from JSON
def load_cheetah_model(fname: str, name: Optional[str] = None) -> cheetah.Segment:
    """Load a cheetah model from json file"""
    with open(fname, "r") as f:
        lattice_dict = json.load(f)
    cell = []
    for element_name in lattice_dict["lattices"]["cell"]:
        # Construct new element
        cell.append(
            str_to_class(lattice_dict["elements"][element_name][0])(
                name=element_name, **lattice_dict["elements"][element_name][1]
            )
        )

    return cheetah.Segment(cell=cell, name=name)


def str_to_class(classname: str):
    # get class from string
    return getattr(cheetah, classname)
