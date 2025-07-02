import pytest

from cheetah.converters.utils import rpn


def test_valid_rpn_expression():
    """
    Test that a valid RPN expression without nesting is correctly recognised as a valid
    RPN expression, meaning it is evaluated to the correct number without throwing an
    exception.
    """
    expression = "2 3 +"

    assert rpn.evaluate_expression(expression) == 5


def test_valid_rpn_expression_in_single_quotes():
    """
    Test that when single quotes are placed around a valid RPN expression, it is
    correctly recognised as invalid and throws a `SyntaxError`.
    """
    expression = "'2 3 +'"

    with pytest.raises(SyntaxError):
        rpn.evaluate_expression(expression)


def test_falsely_validated_normal_expression():
    """
    Tests that the expression `"ldsp2h +dldsp17h +lblxsph/2-lbxsph/2"`, which was
    falsely recognised as a valid RPN expression in a previous version of Cheetah, is
    correctly recognised as invalid.
    """
    expression = "ldsp2h +dldsp17h +lblxsph/2-lbxsph/2"

    with pytest.raises(SyntaxError):
        rpn.evaluate_expression(expression)


def test_nested_rpn_expression():
    """
    Test that a valid RPN expression with nesting is correctly evaluated without
    throwing an exception.
    """
    expression = "10 2 * 4 2 ^ + sqrt"  # sqrt(20 + 16) = 6

    assert rpn.evaluate_expression(expression) == 6


def test_nested_rpn_expression_with_comment():
    """
    Test that a valid RPN expression with nesting and a comment is evaluated correctly
    without throwing an exception.
    """
    expression = "10 2 * 3 4 * + #should be valid"  # 20 + 12 = 32

    assert rpn.evaluate_expression(expression) == 32


def test_nested_rpn_expression_with_context():
    """
    Test that a valid RPN expression with nesting and a context is evaluated correctly
    without throwing an exception, meaning that the context is correctly applied to the
    expression.
    """
    context = {"pi": 3}  # Close enough :D
    expression = "10 2 * pi 4 * +"  # 20 + 12 = 32

    assert rpn.evaluate_expression(expression, context) == 32


def test_rpn_nested_var_lookup():
    """
    Test that an RPN expression with nested variable lookups is correctly evaluated.
    """
    expression = "a b[test] -"
    context = {"a": 10, "b": {"beep": 10, "boop": 100, "test": 5}, "test": 3}

    assert rpn.evaluate_expression(expression, context) == 5
