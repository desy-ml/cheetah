import math
from typing import Any

operators = {
    "+": {"precedence": 1, "inputs": 2, "func": lambda a, b: a + b},
    "-": {"precedence": 1, "inputs": 2, "func": lambda a, b: a - b},
    "*": {"precedence": 2, "inputs": 2, "func": lambda a, b: a * b},
    "/": {"precedence": 2, "inputs": 2, "func": lambda a, b: a / b},
    "^": {"precedence": 3, "inputs": 2, "func": lambda a, b: a**b},
    "sqrt": {"precedence": 4, "inputs": 1, "func": lambda a: math.sqrt(a)},
    "sin": {"precedence": 4, "inputs": 1, "func": lambda a: math.sin(a)},
    "cos": {"precedence": 4, "inputs": 1, "func": lambda a: math.cos(a)},
    "tan": {"precedence": 4, "inputs": 1, "func": lambda a: math.tan(a)},
    "abs": {"precedence": 4, "inputs": 1, "func": lambda a: abs(a)},
    "log": {"precedence": 4, "inputs": 1, "func": lambda a: math.log(a)},
}


def evaluate_expression(expression: str, context: dict | None = None) -> Any:
    """
    Evaluates an expression in Infix Notation.

    Throws a `SyntaxError` if the expression is invalid.
    """
    context = context or {}

    # tokenize the expression
    try:
        tokens = _tokenise_expression(expression, context)
    except Exception as e:
        raise SyntaxError(
            f"Invalid expression: {expression} Unable to tokenise- {str(e)}"
        )

    # parse tokens into AST
    try:
        ast = _parse_expression(tokens)
    except Exception as e:
        raise SyntaxError(f"Invalid expression: {expression} Unable to parse- {str(e)}")

    # evaluate the AST
    try:
        return _evaluate_ast(ast)
    except Exception as e:
        raise SyntaxError(
            f"Invalid expression: {expression} Unable to evaluate- {str(e)}"
        )


def _evaluate_ast(node: dict) -> Any:
    """
    Evaluates an Abstract Syntax Tree (AST) node.
    """
    # depth first traversal of the AST
    curnode = node

    if curnode["left"] is not None:
        curnode["left"] = _evaluate_ast(curnode["left"])
    if curnode["right"] is not None:
        curnode["right"] = _evaluate_ast(curnode["right"])

    if curnode["value"] in operators:
        if operators[curnode["value"]]["inputs"] == 1:
            return operators[curnode["value"]]["func"](curnode["left"])
        else:
            return operators[curnode["value"]]["func"](
                curnode["left"], curnode["right"]
            )
    else:
        return curnode["value"]


def _parse_expression(tokens: list[str]) -> dict:
    """
    Parses a list of tokens into an Abstract Syntax Tree (AST)
    """
    output = None
    stack = []
    operator_stack = []

    while len(tokens) > 0:
        token = tokens.pop(0)
        if token == "(":
            # extract nested expression and parse it
            nested_tokens = []
            nested_level = 1
            while nested_level > 0:
                token = tokens.pop(0)
                if token == "(":
                    nested_level += 1
                elif token == ")":
                    nested_level -= 1
                if nested_level > 0:
                    nested_tokens.append(token)
            if nested_level != 0:
                raise SyntaxError("Mismatched parentheses in expression")
            nested_ast = _parse_expression(nested_tokens)
            stack.append(nested_ast)
        elif token not in operators:
            stack.append({"value": float(token), "left": None, "right": None})
        else:
            while (
                operator_stack
                and operators[operator_stack[-1]]["precedence"]
                >= operators[token]["precedence"]
            ):
                if operators[operator_stack[-1]]["inputs"] == 1:
                    right = None
                else:
                    right = stack.pop()
                left = stack.pop()
                operator = operator_stack.pop()
                output = {"value": operator, "left": left, "right": right}
                stack.append(output)
            operator_stack.append(token)

    while operator_stack:
        if operators[operator_stack[-1]]["inputs"] == 1 or (
            operator_stack[-1] == "-" and len(stack) == 1
        ):
            # Handle unary functions or unary minus
            right = None
            if operator_stack[-1] == "-" and len(stack) == 1:
                # Unary minus, treat as 0 - x
                left = {"value": 0, "left": None, "right": None}
                right = stack.pop()
            else:
                left = stack.pop()
        else:
            right = stack.pop()
            left = stack.pop()
        operator = operator_stack.pop()
        output = {"value": operator, "left": left, "right": right}
        stack.append(output)
    if len(stack) != 1:
        raise SyntaxError("Invalid expression: too many values left in stack")
    output = stack.pop()

    return output


def _tokenise_expression(expression: str, context: dict) -> list[str]:
    """
    Tokenizes an infix expression into a list of tokens. Lookup in
    context for variable names.
    """
    tokens = []
    current_token = ""
    for char in expression:
        if char.isspace():
            if current_token:
                if current_token in context:
                    tokens.append(context[current_token])
                else:
                    tokens.append(current_token)
                current_token = ""
        elif char in "+-*/^()":
            if current_token:
                # workaround for conflicts between function and
                # var names (ie, abs and abs())
                if char != "(" and current_token in context:
                    tokens.append(context[current_token])
                else:
                    tokens.append(current_token)
                current_token = ""
            tokens.append(char)
        else:
            current_token += char
    if current_token:
        if current_token in context:
            tokens.append(context[current_token])
        else:
            tokens.append(current_token)
    return tokens
