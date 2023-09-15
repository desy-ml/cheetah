import json
from typing import Any, Optional, Tuple

import torch

import cheetah


def feature2nontorch(value: Any) -> Any:
    """
    if necesary, convert an the value of a feature of a `cheetah.Element` to a non-torch
    type that can be saved to LatticeJSON.

    :param value: Value of the feature that might be in some kind of PyTorch format,
        such as `torch.Tensor` or `torch.nn.Parameter`.
    :return: Value of the feature if it is not in a PyTorch format, otherwise the
        value converted to a non-PyTorch format.
    """
    return (
        value.tolist()
        if isinstance(value, (torch.Tensor, torch.nn.Parameter))
        else value
    )


def convert_element(element: "cheetah.Element"):
    """
    Deconstruct an element into its name, class and parameters for saving to JSON.

    :param element: Cheetah element
    :return: Tuple of element name, element class, and element parameters
    """
    params = {
        feauture: feature2nontorch(getattr(element, feauture))
        for feauture in element.defining_features
    }

    return element.name, element.__class__.__name__, params


def convert_segment(segment: "cheetah.Segment") -> Tuple[dict, dict]:
    """
    Deconstruct a segment into its name, a list of its elements and a dictionary of
    its element parameters for saving to JSON.

    :param segment: Cheetah segment.
    :return: Tuple of elments and lattices dictionaries found in segment, including
        the segment itself.
    """
    elements = {}
    lattices = {}

    cell = []

    for element in segment.elements:
        if isinstance(element, cheetah.Segment):
            segment_elements, segment_lattices = convert_segment(element)

            elements.update(segment_elements)
            lattices.update(segment_lattices)
        else:
            element_name, element_class, element_params = convert_element(element)

            elements[element_name] = [element_class, element_params]

        cell.append(element_name)

    lattices[segment.name] = cell

    return elements, lattices


def save_cheetah_model(
    segment: "cheetah.Segment",
    filename: str,
    title: Optional[str] = None,
    info: str = "This is a placeholder lattice description",
) -> None:
    """
    Save a cheetah model to json file accoding to the lattice-json convention
    c.f. https://github.com/nobeam/latticejson

    :param segment: Cheetah `Segment` to save.
    :param filename: Name/path of the file to save the lattice to.
    :param title: Title of the lattice. If not provided, defaults to the name of the
        `Segment` object. If that also does not have a name, defaults to "Unnamed
        Lattice".
    :param info: Information about the lattice. Defaults to "This is a placeholder
        lattice description".
    """
    if title is None:
        title = segment.name if segment.name is not None else "Unnamed Lattice"

    metadata = {
        "version": "cheetah-0.6",
        "title": title,
        "info": info,
        "root": segment.name if segment.name is not None else "cell",
    }

    lattice_dict = metadata.copy()

    elements, lattices = convert_segment(segment)
    lattice_dict["elements"] = elements
    lattice_dict["lattices"] = lattices

    with open(filename, "w") as f:
        s = json.dumps(lattice_dict, cls=CompactJSONEncoder, indent=4)
        f.write(s)


class CompactJSONEncoder(json.JSONEncoder):
    """
    A JSON Encoder which only indents the first two levels.

    Taken from https://github.com/nobeam/latticejson/blob/main/latticejson/format.py
    """

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


def nontorch2feature(value: Any) -> Any:
    """
    Convert a value like a float, int, etc. to a torch tensor if necessary. Values of
    type `str` and `bool` are not converted, because the all currently existing
    `cheetah.Element` subclasses expect these values to not be `torch.Tensor`s.

    :param value: Value to convert to a `torch.Tensor` if necessary.
    :return: Value converted to a `torch.Tensor` if necessary.
    """
    return value if isinstance(value, (str, bool)) else torch.tensor(value)


def parse_element(name: str, lattice_dict: dict) -> "cheetah.Element":
    """
    Parse an `Element` named `name` from a `lattice_dict`.

    :param name: Name of the `Element` to parse.
    :param lattice_dict: Dictionary containing the lattice information.
    """
    element_class = getattr(cheetah, lattice_dict["elements"][name][0])
    params = lattice_dict["elements"][name][1]

    converted_params = {key: nontorch2feature(value) for key, value in params.items()}

    return element_class(name=name, **converted_params)


def parse_segment(name: str, lattice_dict: dict) -> "cheetah.Segment":
    """
    Parse a `Segment` named `name` from a `lattice_dict`.

    :param name: Name of the `Segment` to parse.
    :param lattice_dict: Dictionary containing the lattice information.
    """
    elements = []
    for element_name in lattice_dict["lattices"][name]:
        # Construct new element
        if element_name in lattice_dict["lattices"]:
            new_element = parse_segment(element_name, lattice_dict)
        else:
            new_element = parse_element(element_name, lattice_dict)

        # Append the element to the list of elements
        elements.append(new_element)

    return cheetah.Segment(elements=elements, name=name)


def load_cheetah_model(filename: str) -> "cheetah.Segment":
    """
    Load a Cheetah model from a JSON file.

    :param filename: Name/path of the file to load the lattice from.
    :return: Loaded Cheetah `Segment`.
    """
    with open(filename, "r") as f:
        lattice_dict = json.load(f)

    root_name = lattice_dict["root"]

    return parse_segment(root_name, lattice_dict)
