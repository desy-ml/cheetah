import math
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import scipy
from scipy.constants import physical_constants

from cheetah.converters.utils import rpn


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
    for line in lines:
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

    # Finally remove spaces again, because some may now have appeared
    replaced_lines = [line.strip() for line in replaced_lines]

    return replaced_lines


def merge_delimiter_continued_lines(
    lines: list[str], delimiter: str, remove_delimiter: bool = False
) -> list[str]:
    """
    Merge lines ending with some character as a delimitter with the following line.

    :param lines: List of lines to merge.
    :param delimitter: Character to use as a delimitter.
    :param remove_delimitter: Whether to remove the delimitter from the merged line.
    :return: List of lines with ampersand-continued lines merged.
    """
    merged_lines = deepcopy(lines)
    for i in range(len(merged_lines) - 1):
        if merged_lines[i] is not None and merged_lines[i].endswith(delimiter):
            num_added_lines = 1
            while merged_lines[i].endswith(delimiter):
                if remove_delimiter:
                    merged_lines[i] = (
                        merged_lines[i][:-1] + merged_lines[i + num_added_lines]
                    )
                else:
                    merged_lines[i] = (
                        merged_lines[i] + merged_lines[i + num_added_lines]
                    )
                merged_lines[i + num_added_lines] = None
                num_added_lines += 1

    # Prune None lines
    merged_lines = [line for line in merged_lines if line is not None]

    # Remove spaces again, because some may now have appeared
    merged_lines = [line.strip() for line in merged_lines]

    return merged_lines


def evaluate_expression(expression: str, context: dict) -> Any:
    """
    Evaluate an expression in the context of a dictionary of variables.

    :param expression: Expression to evaluate.
    :param context: Dictionary of variables to evaluate the expression in the context
        of.
    :return: Result of evaluating the expression.
    """

    # Try reading the expression as an integer
    try:
        return int(expression)
    except ValueError:
        pass

    # Try reading the expression as a float
    try:
        return float(expression)
    except ValueError:
        pass

    # Check against allowed keywords
    if expression in ["open", "electron", "t", "f", "traveling_wave", "full"]:
        return expression

    # Check against previously defined variables
    if expression in context:
        return context[expression]

    # Evaluate as a mathematical expression
    try:
        # Surround expressions in brackets with quotes
        expression = re.sub(r"\[([a-z0-9_%]+)\]", r"['\1']", expression)
        # Replace power operator with python equivalent
        expression = re.sub(r"\^", r"**", expression)
        # Replace abs with abs_func when it is followed by a (
        # NOTE: This is a hacky fix to deal with abs being overwritten in the LCLS
        # lattice file. I'm not sure this replacement will lead to the intended
        # behaviour.
        expression = re.sub(r"abs\(", r"abs_func(", expression)

        return (
            eval(expression, context)
            if not rpn.is_valid_expression(expression)
            else rpn.eval_expression(expression, context)
        )
    except SyntaxError:
        if not (
            len(expression.split(":")) == 3 or len(expression.split(":")) == 4
        ):  # It's probably an alias
            print(
                f"DEBUG: Evaluating expression {expression}. Assuming it is a string."
            )
        return expression
    except Exception as e:
        print(expression)
        raise e


def resolve_object_name_wildcard(wildcard_pattern: str, context: dict) -> list:
    """
    Return a list of object names that match the given wildcard pattern.

    :param wildcard_pattern: Wildcard pattern to match.
    :param context: Dictionary of variables among which to search for matching object.
    :return: List of object names that match the given wildcard pattern, both in terms
        of name and element type.
    """
    object_type, object_name = wildcard_pattern.split("::")

    pattern = object_name.replace("*", ".*").replace("%", ".")
    name_matching_keys = [key for key in context.keys() if re.fullmatch(pattern, key)]
    type_matching_keys = [
        key
        for key in name_matching_keys
        if isinstance(context[key], dict)
        and "element_type" in context[key]
        and context[key]["element_type"] == object_type
    ]

    return type_matching_keys


def assign_property(line: str, context: dict) -> dict:
    """
    Assign a property of an element to the context.

    :param line: Line of a property assignment to be parsed.
    :param context: Dictionary of variables to assign the property to and from which to
        read variables.
    :return: Updated context.
    """
    pattern = r"([a-z0-9_\*:]+)\[([a-z0-9_%]+)\]\s*=(.*)"
    match = re.fullmatch(pattern, line)

    object_name = match.group(1).strip()
    property_name = match.group(2).strip()
    property_expression = match.group(3).strip()  # TODO: Evaluate expression first

    if "*" in object_name or "%" in object_name:
        object_names = resolve_object_name_wildcard(object_name, context)
    else:
        object_names = [object_name]

    expression_result = evaluate_expression(property_expression, context)

    for name in object_names:
        if name not in context:
            context[name] = {}
        context[name][property_name] = expression_result

    return context


def assign_variable(line: str, context: dict) -> dict:
    """
    Assign a variable to the context.

    :param line: Line of a variable assignment to be parsed.
    :param context: Dictionary of variables to assign the variable to and from which to
        read variables.
    :return: Updated context.
    """
    pattern = r"([a-z0-9_]+)\s*=(.*)"
    match = re.fullmatch(pattern, line)

    variable_name = match.group(1).strip()
    variable_expression = match.group(2).strip()

    context[variable_name] = evaluate_expression(variable_expression, context)

    return context


def define_element(line: str, context: dict) -> dict:
    """
    Define an element in the context.

    :param line: Line of an element definition to be parsed.
    :param context: Dictionary of variables to define the element in and from which to
        read variables.
    :return: Updated context.
    """
    pattern = r"([a-z0-9_\.]+)\s*\:\s*([a-z0-9_]+)(\,(.*))?"
    match = re.fullmatch(pattern, line)

    element_name = match.group(1).strip()
    element_type = match.group(2).strip()

    if element_type in context:
        element_properties = deepcopy(context[element_type])
    else:
        element_properties = {"element_type": element_type}

    if match.group(3) is not None:
        element_properties_string = match.group(4).strip()

        property_pattern = (
            r"([a-z0-9_]+\s*\=\s*\"[^\"]+\"|[a-z0-9_]+\s*\=\s*[^\=\,\"]+)"
        )
        property_matches = re.findall(property_pattern, element_properties_string)

        for property_string in property_matches:
            property_string = property_string.strip()

            property_name, property_expression = property_string.split("=")
            property_name = property_name.strip()
            property_expression = property_expression.strip()

            element_properties[property_name] = evaluate_expression(
                property_expression, context
            )

    context[element_name] = element_properties

    return context


def define_line(line: str, context: dict) -> dict:
    """
    Define a beam line in the context.

    :param line: Line of a beam line definition to be parsed.
    :param context: Dictionary of variables to define the beam line in and from which
        to read variables.
    :return: Updated context.
    """
    pattern = r"([a-z0-9_]+)\s*\:\s*line\s*=\s*\((.*)\)"
    match = re.fullmatch(pattern, line)

    line_name = match.group(1).strip()
    line_elements_string = match.group(2).strip()

    line_elements = []
    for element_name in line_elements_string.split(","):
        element_name = element_name.strip()

        line_elements.append(element_name)

    context[line_name] = line_elements

    return context


def define_overlay(line: str, context: dict) -> dict:
    """
    Define an overlay in the context.

    :param line: Line of an overlay definition to be parsed.
    :param context: Dictionary of variables to define the overlay in and from which to
        read variables.
    :return: Updated context.
    """
    knot_based_pattern = r"([a-z0-9_]+)\s*\:\s*overlay\s*=\s*\{(.*)\}\s*\,\s*var\s*=\s*\{\s*([a-z0-9_]+)\s*\}\s*\,\s*x_knot\s*=\s*\{(.*)\}"  # noqa: E501
    expression_based_pattern = r"([a-z0-9_]+)\s*\:\s*overlay\s*=\s*\{(.*)\}\s*\,\s*var\s*=\s*\{(.*)\}\s*(\,.*)*"  # noqa: E501

    expression_match = re.fullmatch(expression_based_pattern, line)
    knot_match = re.fullmatch(knot_based_pattern, line)

    if knot_match:
        overlay_name = knot_match.group(1).strip()
        overlay_definition = knot_match.group(2).strip()
        overlay_variable = knot_match.group(3).strip()
        overlay_x_knot = knot_match.group(4).strip()

        context[overlay_name] = {
            "overlay_definition": overlay_definition,
            "overlay_variable": overlay_variable,
            "overlay_x_knot": overlay_x_knot,
        }
    elif expression_match:
        overlay_name = expression_match.group(1).strip()
        overlay_definition = expression_match.group(2).strip()
        overlay_variables = expression_match.group(3).strip()
        if expression_match.group(4) is not None:
            overlay_parameters = expression_match.group(4).strip()[1:].strip()
        else:
            overlay_parameters = None

        context[overlay_name] = {
            "overlay_definition": overlay_definition,
            "overlay_variables": overlay_variables,
            "overlay_parameters": overlay_parameters,
        }
    else:
        raise ValueError(f"Overlay definition {line} not understood.")

    return context


def parse_use_line(line: str, context: dict) -> dict:
    """
    Parse a use line.

    :param line: Line of a use statement to be parsed.
    :param context: Dictionary of variables to define the overlay in and from which to
        read variables.
    :return: Updated context.
    """
    pattern = r"use\s*\,\s*([a-z0-9_]+)"
    match = re.fullmatch(pattern, line)

    use_line_name = match.group(1).strip()
    context["__use__"] = use_line_name

    return context


def parse_lines(lines: str) -> dict:
    """
    Parse a list of lines from a Bmad lattice file. They should be cleaned and merged
    before being passed to this function.

    :param lines: List of lines to parse.
    :return: Dictionary of variables defined in the lattice file.
    """
    property_assignment_pattern = r"[a-z0-9_\*:]+\[[a-z0-9_%]+\]\s*=.*"
    variable_assignment_pattern = r"[a-z0-9_]+\s*=.*"
    element_definition_pattern = r"[a-z0-9_\.]+\s*\:\s*[a-z0-9_]+.*"
    line_definition_pattern = r"[a-z0-9_]+\s*\:\s*line\s*=\s*\(.*\)"
    overlay_definition_pattern = r"[a-z0-9_]+\s*\:\s*overlay\s*=\s*\{.*"
    use_line_pattern = r"use\s*\,\s*[a-z0-9_]+"

    context = {
        "pi": scipy.constants.pi,
        "twopi": 2 * scipy.constants.pi,
        "c_light": scipy.constants.c,
        "emass": physical_constants["electron mass energy equivalent in MeV"][0] * 1e-3,
        "m_electron": (
            physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
        ),
        "sqrt": math.sqrt,
        "asin": math.asin,
        "sin": math.sin,
        "cos": math.cos,
        "abs_func": abs,
        "raddeg": scipy.constants.degree,
    }

    for line in lines:
        if re.fullmatch(property_assignment_pattern, line):
            context = assign_property(line, context)
        elif re.fullmatch(variable_assignment_pattern, line):
            context = assign_variable(line, context)
        elif re.fullmatch(line_definition_pattern, line):
            context = define_line(line, context)
        elif re.fullmatch(overlay_definition_pattern, line):
            context = define_overlay(line, context)
        elif re.fullmatch(element_definition_pattern, line):
            context = define_element(line, context)
        elif re.fullmatch(use_line_pattern, line):
            context = parse_use_line(line, context)

    return context


def validate_understood_properties(understood: list[str], properties: dict) -> None:
    """
    Validate that all properties are understood. This function primarily ensures that
    properties not understood by Cheetah are not ignored silently.

    Raises an `AssertionError` if a property is found that is not understood.

    :param understood: List of properties understood (or purpusefully ignored) by
        Cheetah.
    :param properties: Dictionary of properties to validate.
    :return: None
    """
    for property in properties:
        assert any([re.fullmatch(pattern, property) for pattern in understood]), (
            f"Property {property} with value {properties[property]} for element type"
            f" {properties['element_type']} is currently not understood. Other values"
            f" in properties are {properties.keys()}."  # noqa: B038
        )
