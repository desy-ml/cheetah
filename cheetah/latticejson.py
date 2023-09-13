import json
from typing import Optional

import cheetah


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

    return cheetah.Segment(elements=cell, name=name)


def str_to_class(classname: str):
    # get class from string
    return getattr(cheetah, classname)
