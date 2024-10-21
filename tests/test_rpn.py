from cheetah.converters.utils import rpn


def test_valid_rpn_expression():
    """
    Test that a valid RPN expression without nesting is correctly recognised as a valid
    RPN expression.
    """
    expression = "2 3 +"

    assert rpn.is_valid_expression(expression)


def test_valid_rpn_expression_with_single_quotes():
    """
    Test that a valid RPN expression that has single quotes around it and should
    therefore not be recognised as valid because these should have been stripped off
    before calling the function.
    """
    expression = "'2 3 +'"

    assert not rpn.is_valid_expression(expression)


def test_falsely_validated_normal_expression():
    """
    Tests that the expression `"ldsp2h +dldsp17h +lblxsph/2-lbxsph/2"`, which was
    falsely recognised as a valid RPN expression in a previous version of Cheetah, is
    correctly recognised as invalid.
    """
    expression = "ldsp2h +dldsp17h +lblxsph/2-lbxsph/2"

    assert not rpn.is_valid_expression(expression)
