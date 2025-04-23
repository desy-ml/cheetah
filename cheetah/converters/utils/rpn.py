import math
from typing import Any


def evaluate_expression(expression: str, context: dict | None = None) -> Any:
    """
    Evaluates an expression in Reverse Polish Notation.

    Throws a `SyntaxError` if the expression is invalid.
    """
    context = context or {}

    stack = []
    stripped = expression.strip()
    for token in stripped.split(" "):
        match token:
            case "+":
                try:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a + b)
                except IndexError:
                    raise SyntaxError(
                        f"Invalid expression: {expression} - Need two values before +"
                    )
            case "-":
                try:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a - b)
                except IndexError:
                    raise SyntaxError(
                        f"Invalid expression: {expression} - Need two values before -"
                    )
            case "*":
                try:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a * b)
                except IndexError:
                    raise SyntaxError(
                        f"Invalid expression: {expression} - Need two values before *"
                    )
            case "/":
                try:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a / b)
                except IndexError:
                    raise SyntaxError(
                        f"Invalid expression: {expression} - Need two values before /"
                    )
            case "^":
                try:
                    b = stack.pop()
                    a = stack.pop()
                    stack.append(a**b)
                except IndexError:
                    raise SyntaxError(
                        f"Invalid expression: {expression} - Need two values before ^"
                    )
            case "sqrt":
                try:
                    a = stack.pop()
                    stack.append(math.sqrt(a))
                except IndexError:
                    raise SyntaxError(
                        f"Invalid expression: {expression} - Need one value before sqrt"
                    )
            case "sin":
                try:
                    a = stack.pop()
                    stack.append(math.sin(a))
                except IndexError:
                    raise SyntaxError(
                        f"Invalid expression: {expression} - Need one value before sin"
                    )
            case "cos":
                try:
                    a = stack.pop()
                    stack.append(math.cos(a))
                except IndexError:
                    raise SyntaxError(
                        f"Invalid expression: {expression} - Need one value before cos"
                    )
            case "tan":
                try:
                    a = stack.pop()
                    stack.append(math.tan(a))
                except IndexError:
                    raise SyntaxError(
                        f"Invalid expression: {expression} - Need one value before tan"
                    )
            case "asin":
                try:
                    a = stack.pop()
                    stack.append(math.asin(a))
                except IndexError:
                    raise SyntaxError(
                        f"Invalid expression: {expression} - Need one value before asin"
                    )
            case _:  # all other tokens
                # commment, ignore this and all following tokens
                if token[0] == "#":
                    break
                try:
                    # read as float since it's all torch in the back anyway
                    number = float(token)
                except ValueError:
                    if token in context:
                        number = context[token]
                    else:
                        raise SyntaxError(
                            f"Invalid expression: {expression} - {token} is"
                            + " not a number or a variable"
                        )
                stack.append(number)
    if len(stack) != 1:
        raise SyntaxError(
            f"Invalid RPN expression: {expression} - Stack not empty after evaluation"
        )
    return stack[0]
