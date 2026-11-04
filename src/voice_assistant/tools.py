"""Local tools the assistant can call. Each is a plain function plus a JSON schema for the LLM."""

from __future__ import annotations

import ast
import math
import operator
from datetime import datetime

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}  # fmt: skip
_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "abs": abs,
    "round": round,
}


def calculate(expression: str) -> str:
    """Evaluate arithmetic safely by walking the AST. eval() would run arbitrary code."""

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
        ):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        if isinstance(node, ast.Name) and node.id in ("pi", "e"):
            return getattr(math, node.id)
        raise ValueError("unsupported expression")

    try:
        expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
        value = ev(ast.parse(expr, mode="eval"))
        return f"{value:.6g}" if isinstance(value, float) else str(value)
    except Exception as e:
        return f"error: {e}"


def current_time() -> str:
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


TOOLS = {"calculate": calculate, "current_time": current_time}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression exactly, e.g. '23*17' or 'sqrt(2)'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def run_tool(name: str, arguments: dict) -> str:
    fn = TOOLS.get(name)
    if fn is None:
        return f"error: unknown tool {name}"
    try:
        return fn(**(arguments or {}))
    except TypeError as e:
        return f"error: bad arguments: {e}"
