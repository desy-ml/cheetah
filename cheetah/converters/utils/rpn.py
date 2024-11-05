from typing import Any


def is_valid_expression(expression: str) -> bool:
    """Checks if expression is a reverse Polish notation."""
    stripped = expression.strip()
    return stripped[-1] in "+-/*" and len(stripped.split(" ")) == 3


def eval_expression(expression: str, context: dict) -> Any:
    """
    Evaluates an expression in reverse Polish notation.

    NOTE: Does not support nested expressions.
    """
    splits = expression.strip().split(" ")
    return eval(" ".join([splits[0], splits[2], splits[1]]), context)
