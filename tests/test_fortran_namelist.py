from cheetah.converters.utils.fortran_namelist import (
    evaluate_expression,
    parse_lines,
)
import pytest
import scipy.constants
from scipy.constants import physical_constants
import math
from cheetah.utils import PhysicsWarning


def test_evaluate_expression():
    context = {
        "mc2": 0.511750
    }

    value = evaluate_expression('mc2+0.750e-3', context)
    assert value == pytest.approx(context["mc2"] + 0.750e-3)


def test_define_element_type_and_alias_stored_as_metadata():
    lines = [
        'q1: quadrupole, l = 0.2, alias = "q1_alias", type = control_label, k1 = 1.0'
    ]

    # alias and type are string-like control metadata and may trigger fallback parsing.
    with pytest.warns(PhysicsWarning):
        context = parse_lines(lines)

    q1 = context["q1"]

    assert "alias" not in q1
    assert "type" not in q1
    assert "element_metadata" in q1
    assert q1["element_metadata"]["alias"] == '"q1_alias"'
    assert q1["element_metadata"]["type"] == "control_label"
    assert q1["k1"] == pytest.approx(1.0)