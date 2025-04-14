import pytest

from cheetah.converters.utils import rpn


def test_valid_rpn_expression():
    """
    Test that a valid RPN expression without nesting is correctly recognised as a valid
    RPN expression.
    """
    expression = "2 3 +"
    # try with empty context
    assert rpn.try_eval_expression(expression, []) == 5


def test_complex_rpn_expression():
    """
    Test that a valid RPN expression with nesting is correctly recognised as a valid
    RPN expression.
    """
    expression = "10 2 * 3 4 * +"  # 20 + 12 = 32
    # try with empty context
    assert rpn.try_eval_expression(expression, []) == 32


def test_complex_rpn_expression_with_context():
    """
    Test that a valid RPN expression with nesting and a variable is correctly
    recognised as a valid RPN expression.
    """
    context = {"pi": 3}  # close enough :D
    expression = "10 2 * pi 4 * +"  # 20 + 12 = 32
    assert rpn.try_eval_expression(expression, context) == 32


def test_valid_rpn_expression_with_single_quotes():
    """
    Test that a valid RPN expression that has single quotes around it and should
    therefore not be recognised as valid because these should have been stripped off
    before calling the function.
    """
    expression = "'2 3 +'"
    with pytest.raises(SyntaxError):
        rpn.try_eval_expression(expression, [])


def test_falsely_validated_normal_expression():
    """
    Tests that the expression `"ldsp2h +dldsp17h +lblxsph/2-lbxsph/2"`, which was
    falsely recognised as a valid RPN expression in a previous version of Cheetah, is
    correctly recognised as invalid.
    """
    expression = "ldsp2h +dldsp17h +lblxsph/2-lbxsph/2"

    with pytest.raises(SyntaxError):
        rpn.try_eval_expression(expression, [])
