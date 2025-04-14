from typing import Any


def try_eval_expression(expression: str, context: dict) -> Any:
    """
    Tries to evaluate an expression in reverse Polish notation.

    Throws Syntax Exception if the expression is not valid.
    """
    stack = []
    stripped = expression.strip()
    for token in stripped.split(" "):
        match token:
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
            case "#":  # commment, ignore this and all following tokens
                break
            case _:  # all other tokens
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
