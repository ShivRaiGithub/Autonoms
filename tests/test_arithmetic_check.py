from pathlib import Path

import pytest

from autonoms.checks.arithmetic_check import check_directory, check_line, evaluate


@pytest.mark.parametrize(
    "line",
    [
        "12 / 4 = 3",
        "2 + 3 x 4 = 14",  # multiplication first
        "(2 + 3) x 4 = 20",
        "10 - 4 - 3 = 3",  # left to right
        "3/6 + 2/6 = 5/6",
        "1/4 + 1/4 = 2/4 = 1/2",
        "4/8 = 1/2",
        "0.5 x 4 = 2",
        "- Addition: 5 + 3 = 8",  # list marker and label
        "1. 6 x 7 = 42",
        "9 + 16 = 25.",  # sentence-ending dot
        "so 3 x 4 = 12, then 2 + 12 = 14.",
        "3 x 4 = 12",
        "-3 + 5 = 2",
    ],
)
def test_correct_equations_pass(line):
    checked, wrong = check_line(line)
    assert checked >= 1 and wrong == []


@pytest.mark.parametrize(
    "line, text, claimed, actual",
    [
        ("- Division: 12 / 4 = 4", "12 / 4 = 4", "4", "3"),
        ("so 3 x 4 = 12, then 2 + 12 = 20.", "2 + 12 = 20", "20", "14"),
        ("Example: 1/4 + 1/4 = 2/8", "1/4 + 1/4 = 2/8", "1/4", "1/2"),
        ("2 + 3 x 4 = 20", "2 + 3 x 4 = 20", "20", "14"),
        ("10 - 4 - 3 = 9", "10 - 4 - 3 = 9", "9", "3"),
    ],
)
def test_wrong_equations_are_reported_exactly(line, text, claimed, actual):
    checked, wrong = check_line(line)
    assert wrong == [(text, __import__("fractions").Fraction(claimed), __import__("fractions").Fraction(actual))]


@pytest.mark.parametrize(
    "line",
    [
        "Rule: a + b = c",
        "area = pi x radius x 2",
        "x = 5",
        "take base",
        "1 / 0 = 5",  # division by zero is skipped, never guessed
    ],
)
def test_prose_and_formulas_are_skipped_not_guessed(line):
    assert check_line(line) == (0, [])


def test_evaluate_precedence_and_errors():
    from autonoms.checks.arithmetic_check import NotAnExpression

    assert evaluate("2 + 3 x 4") == 14
    assert evaluate("-(2 + 3) x 2") == -10
    for bad in ("", "2 +", "(2", "2 3", "a + 1"):
        with pytest.raises(NotAnExpression):
            evaluate(bad)


def test_check_directory(tmp_path: Path):
    (tmp_path / "a.md").write_text("12 / 4 = 4\nfine: 1 + 1 = 2\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("2 + 3 x 4 = 14\n")
    (tmp_path / "c.txt").write_text("1 + 1 = 3\n")  # not markdown: ignored
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.md").write_text("1 + 1 = 3\n")  # git internals: ignored
    result = check_directory(tmp_path)
    assert result["files"] == 2 and result["equations"] == 3
    assert [(p["file"], p["line"]) for p in result["problems"]] == [("a.md", 1)]


def test_the_seed_notes_have_exactly_the_three_visible_mistakes():
    seed = Path(__file__).parent.parent / "seed-repo"
    problems = [p for p in check_directory(seed)["problems"] if p["file"] != "ISSUES.md"]
    assert sorted((p["file"], p["text"]) for p in problems) == [
        ("arithmetic.md", "12 / 4 = 4"),
        ("bodmas.md", "2 + 12 = 20"),
        ("fractions.md", "1/4 + 1/4 = 2/8"),
    ]
