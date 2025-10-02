import pytest

from cheetah.converters.utils import infix


def test_valid_infix_expression():
    """
    Test that a valid infix expression without nesting is correctly recognised as a
    valid expression, meaning it is evaluated to the correct number without throwing an
    exception.
    """
    expression = "2 + 3"

    assert infix.evaluate_expression(expression) == 5


def test_invalid_infix_expression():
    """
    Test that an invalid infix expression is correctly recognised as invalid and throws
    a `SyntaxError`.
    """
    expression = "2 +"

    with pytest.raises(SyntaxError):
        infix.evaluate_expression(expression)


def test_nested_infix_expression():
    """
    Test that a valid infix expression with nesting is correctly evaluated without
    throwing an exception.
    """
    expression = "(10 * 2) + (4 ^ 2)"  # (20 + 16) = 36

    assert infix.evaluate_expression(expression) == 36


def test_infix_expression_with_context():
    """
    Test that a valid infix expression with context is correctly evaluated without
    throwing an exception.
    """
    expression = "a + b"
    context = {"a": 10, "b": 5}

    assert infix.evaluate_expression(expression, context) == 15


def test_infix_expression_with_invalid_context():
    """
    Test that an infix expression with an invalid context variable raises a
    `SyntaxError`.
    """
    expression = "a + b"
    context = {"a": 10}

    with pytest.raises(SyntaxError):
        infix.evaluate_expression(expression, context)


def test_infix_expression_with_invalid_token():
    """
    Test that an infix expression with an invalid token raises a `SyntaxError`.
    """
    expression = "2 + invalid_token"

    with pytest.raises(SyntaxError):
        infix.evaluate_expression(expression)


def test_infix_expression_with_mismatched_parentheses():
    """
    Test that an infix expression with mismatched parentheses raises a `SyntaxError`.
    """
    expression = "(2 + 3"

    with pytest.raises(SyntaxError):
        infix.evaluate_expression(expression)


def test_infix_single_bracket_expression():
    """
    Test that an infix expression with a single value in a bracket is correctly
    evaluated.
    """
    expression = "((((5))))"

    assert infix.evaluate_expression(expression) == 5


def test_infix_expression_with_function_call():
    """Test that an infix expression with a function call is correctly evaluated."""
    expression = "sqrt(16) + abs(-4)"

    assert infix.evaluate_expression(expression) == 8


def test_infix_expression_with_var_conflict():
    """
    Test that an infix expression with a variable name that conflicts with a function
    name is correctly evaluated.
    """
    expression = "abs(abs) + 3"
    context = {"abs": -10}

    assert infix.evaluate_expression(expression, context) == 13  # 10 + 3 = 13


def test_infix_expression_deeply_nested():
    """Test that a deeply nested infix expression is correctly evaluated."""
    expression = "((2 + 3) * (4 - 1)) ^ 2"  # ((5 * 3) ^ 2) = 225

    assert infix.evaluate_expression(expression) == 225


def test_infix_error_handling():
    """Test that appropriate errors are raised and caught"""
    expression = "sqrt(-1)"

    with pytest.raises(SyntaxError):
        infix.evaluate_expression(expression)


def test_infix_error_handling_2():
    """Test that appropriate errors are raised and caught"""
    expression = "sin(abc)"

    with pytest.raises(SyntaxError):
        infix.evaluate_expression(expression)


def test_infix_nested_var_lookup():
    """
    Test that an infix expression with nested variable lookups is correctly evaluated.
    """
    expression = "a - b[test]"
    context = {"a": 10, "b": {"beep": 10, "boop": 100, "test": 5}, "test": 3}

    assert infix.evaluate_expression(expression, context) == 5
