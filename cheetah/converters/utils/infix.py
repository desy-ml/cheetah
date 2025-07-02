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
    Evaluates an expression in Infix notation.

    Throws a `SyntaxError` if the expression is invalid.
    """
    context = context or {}

    # Tokenize the expression
    # No try because this should never error
    tokens = _tokenize_expression(expression, context)

    # Parse tokens into AST
    # Syntax errors manually raised
    # IndexErrors from unexpectedly empty stacks
    # ValueError for things not being numbers
    try:
        ast = _parse_expression(tokens)
    except (SyntaxError, IndexError, ValueError) as e:
        raise SyntaxError(
            f"Invalid expression: {expression}. Unable to parse- {str(e)}."
        )

    # Evaluate the AST
    # TypeError for invalid function input (ie, abs("abc"))
    # ValueError for invalid function input (ie sqrt(-1))
    # IndexError for malformed AST node (in case something gets through the parser)
    try:
        return _evaluate_ast(ast)
    except (TypeError, ValueError, IndexError) as e:
        raise SyntaxError(
            f"Invalid expression: {expression}. Unable to evaluate- {str(e)}."
        )


def _evaluate_ast(node: dict) -> Any:
    """Evaluates an Abstract Syntax Tree (AST) node."""
    # Depth first traversal of the AST
    if node["left"] is not None:
        node["left"] = _evaluate_ast(node["left"])
    if node["right"] is not None:
        node["right"] = _evaluate_ast(node["right"])

    if node["value"] in operators:
        if operators[node["value"]]["inputs"] == 1:
            return operators[node["value"]]["func"](node["left"])
        else:
            return operators[node["value"]]["func"](node["left"], node["right"])
    else:
        return node["value"]


def _parse_expression(tokens: list[str]) -> dict:
    """Parses a list of tokens into an Abstract Syntax Tree (AST)."""
    output = None
    stack = []
    operator_stack = []

    while len(tokens) > 0:
        token = tokens.pop(0)
        if token == "(":
            # Extract nested expression and parse it
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


def _tokenize_expression(expression: str, context: dict) -> list[str]:
    """
    Tokenizes an infix expression into a list of tokens. Lookup in context for variable
    names.
    """
    tokens = []
    current_token = ""
    current_key = None

    for char in expression:
        if char.isspace() or char in "+-*/^()[]":
            if current_token:
                if char == "]" and current_key is not None:
                    # This will throw an index error if current key is invalid
                    tokens.append(context[current_token][current_key])
                    current_token = ""
                    current_key = None
                    continue
                if char == "[" and current_token in context:
                    # We're doing a var[key] lookup, start reading the key
                    current_key = ""
                    continue

                # Workaround for conflicts between function and var names (i.e. abs and
                # abs())
                if char != "(" and current_token in context:
                    tokens.append(context[current_token])
                else:
                    tokens.append(current_token)
                current_token = ""
            if not char.isspace():
                tokens.append(char)
        else:
            if current_key is not None:
                current_key += char
            else:
                current_token += char

    # Handle the last token if it exists
    if current_token:
        if current_token in context:
            tokens.append(context[current_token])
        else:
            tokens.append(current_token)
    return tokens
