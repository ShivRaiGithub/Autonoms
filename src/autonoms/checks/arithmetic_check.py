"""Checks the arithmetic written in Markdown notes, e.g. "12 / 4 = 3".

This file runs INSIDE the sandbox container as a standalone script, so it uses
only the standard library and imports nothing from Autonoms.

    python check.py <directory>

It prints one JSON object and exits 0 (all equations right) or 1 (some wrong).
Only equations where a number or fraction expression stands left of "=" are
checked; prose, formulas with letters ("a + b = c") and anything it cannot parse
are skipped, never guessed. So it can fail to notice a mistake, but a reported
mistake is a real one.
"""

import json
import re
import sys
from fractions import Fraction
from pathlib import Path

ALLOWED = set("0123456789.+-*/x×÷() ")
LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
MAX_FILE_BYTES = 200_000


class NotAnExpression(Exception):
    pass


class DivisionByZero(Exception):
    """A well-formed expression that divides by zero. Unlike NotAnExpression,
    this must NOT make us try a shorter piece of text: "1 / 0 = 5" would
    otherwise be read as "0 = 5"."""


def evaluate(text: str) -> Fraction:
    """Evaluate +, -, *, x, ×, /, ÷ and brackets with BODMAS precedence, exactly."""
    # Spaces are skipped by the regex, not deleted: deleting them would turn "2 3" into 23.
    tokens = re.findall(r"\d+(?:\.\d+)?|[()+\-*/x×÷]|\S", text)
    position = 0

    def peek():
        return tokens[position] if position < len(tokens) else None

    def take():
        nonlocal position
        position += 1
        return tokens[position - 1]

    # A recursive-descent parser: each level handles one BODMAS precedence level.
    def add_sub():
        value = mul_div()
        while peek() in ("+", "-"):
            operator = take()
            right = mul_div()
            value = value + right if operator == "+" else value - right
        return value

    def mul_div():
        value = unary()
        while peek() in ("*", "x", "×", "/", "÷"):
            operator = take()
            right = unary()
            if operator in ("/", "÷"):
                if right == 0:
                    raise DivisionByZero
                value = value / right
            else:
                value = value * right
        return value

    def unary():
        if peek() == "-":
            take()
            return -unary()
        return atom()

    def atom():
        token = peek()
        if token is None:
            raise NotAnExpression("unexpected end")
        if token == "(":
            take()
            value = add_sub()
            if peek() != ")":
                raise NotAnExpression("missing )")
            take()
            return value
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            take()
            return Fraction(token)
        raise NotAnExpression(f"unexpected {token}")

    value = add_sub()
    if position != len(tokens):
        raise NotAnExpression("trailing text")
    return value


def _longest_valid(candidates):
    for text in candidates:
        text = text.strip()
        if not text or not any(c.isdigit() for c in text):
            continue
        try:
            return text, evaluate(text)
        except (NotAnExpression, ValueError):
            continue
        except DivisionByZero:
            return None  # a real expression we cannot evaluate: skip the equation, don't shorten it
    return None


def left_expression(segment: str):
    """The arithmetic at the END of the text left of an '='."""
    end = len(segment)
    start = end
    while start > 0 and segment[start - 1] in ALLOWED:
        start -= 1
    return _longest_valid(segment[i:end] for i in range(start, end))


def right_expression(segment: str):
    """The arithmetic at the START of the text right of an '='."""
    end = 0
    while end < len(segment) and segment[end] in ALLOWED:
        end += 1
    # "= 20." ends a sentence: drop a trailing "." before parsing
    return _longest_valid(segment[:i].rstrip(".") for i in range(end, 0, -1))


def check_line(line: str):
    """Yield (text, claimed, actual) for every wrong equation in one line."""
    line = LIST_MARKER.sub("", line, count=1)
    parts = line.split("=")
    checked = 0
    wrong = []
    for left_text, right_text in zip(parts, parts[1:]):
        left = left_expression(left_text)
        right = right_expression(right_text)
        if left is None or right is None:
            continue
        checked += 1
        if left[1] != right[1]:
            wrong.append((f"{left[0]} = {right[0]}", right[1], left[1]))
    return checked, wrong


def show(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def check_directory(root: Path) -> dict:
    equations = 0
    files = 0
    problems = []
    for path in sorted(root.rglob("*.md")):
        if ".git" in path.relative_to(root).parts or path.is_symlink() or not path.is_file():
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        files += 1
        for number, line in enumerate(text.splitlines(), 1):
            checked, wrong = check_line(line)
            equations += checked
            for expression, claimed, actual in wrong:
                problems.append(
                    {
                        "file": str(path.relative_to(root)),
                        "line": number,
                        "text": expression,
                        "claimed": show(claimed),
                        "actual": show(actual),
                    }
                )
    return {"files": files, "equations": equations, "problems": problems}


if __name__ == "__main__":
    result = check_directory(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
    print(json.dumps(result))
    sys.exit(1 if result["problems"] else 0)
