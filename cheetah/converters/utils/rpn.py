from typing import Any


def try_eval_expression(expression: str, context: dict) -> Any:
    """
    Tries to evaluate an expression in reverse Polish notation.

    Throws Syntax Exception if the expression is not valid.
    """
    stack = []
    stripped = expression.strip()
    for token in stripped.split(" "):
        if token in "+-*/":
            b = stack.pop()
            a = stack.pop()

            # if else instead of match for compatibility reasons
            if token == "+":
                stack.append(a + b)
            elif token == "-":
                stack.append(a - b)
            elif token == "*":
                stack.append(a * b)
            elif token == "/":
                stack.append(a / b)
            else:
                raise SyntaxError(
                    f"Invalid expression: {expression} - Invalid operator: {token}"
                )
        else:
            if token.isnumeric():
                number = float(
                    token
                )  # putting everything as float since it's all torch in the back anyway
            elif token in context:
                number = context[token]
            else:
                raise SyntaxError(
                    f"Invalid expression: {expression} - {token} is not a number or a variable"
                )
            stack.append(number)
    if len(stack) != 1:
        raise SyntaxError(
            f"Invalid expression: {expression} - Stack not empty after evaluation"
        )
    return stack[0]
