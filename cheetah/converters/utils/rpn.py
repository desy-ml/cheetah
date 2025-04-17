import math
from typing import Any


def try_eval_expression(expression: str, context: dict) -> Any:
    """
    Tries to evaluate an expression in reverse Polish notation.

    Throws Syntax Exception if the expression is not valid.
    """
    stack = []
    stripped = expression.strip()
    for token in stripped.split(" "):
        match token:  # match only the first character
            case "+":
                b = stack.pop()
                a = stack.pop()
                stack.append(a + b)
            case "-":
                b = stack.pop()
                a = stack.pop()
                stack.append(a - b)
            case "*":
                b = stack.pop()
                a = stack.pop()
                stack.append(a * b)
            case "/":
                b = stack.pop()
                a = stack.pop()
                stack.append(a / b)
            case "^":
                b = stack.pop()
                a = stack.pop()
                stack.append(a**b)
            case "sqrt":
                a = stack.pop()
                stack.append(math.sqrt(a))
            case "sin":
                a = stack.pop()
                stack.append(math.sin(a))
            case "cos":
                a = stack.pop()
                stack.append(math.cos(a))
            case "tan":
                a = stack.pop()
                stack.append(math.tan(a))
            case "asin":
                a = stack.pop()
                stack.append(math.asin(a))
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
            f"Invalid expression: {expression} - Stack not empty after evaluation"
        )
    return stack[0]
