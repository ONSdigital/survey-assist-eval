"""Tests for question structure functions."""

import pytest

from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    has_question_mark,
    is_question,
)


def test_has_question_mark_returns_true_when_present():
    """Returns True when the text contains a question mark."""
    text = "Is this a question?"
    assert has_question_mark(text) is True


def test_has_question_mark_returns_false_when_absent():
    """Returns False when the text does not contain a question mark."""
    text = "This is a statement."
    assert has_question_mark(text) is False


def test_has_question_mark_multiple_question_marks():
    """Returns True when multiple question marks are present."""
    text = "What?? Really??"
    assert has_question_mark(text) is True


def test_has_question_mark_empty_string():
    """Returns False when input text is empty."""
    assert has_question_mark("") is False


def test_has_question_mark_none_input():
    """Returns False when input is None."""
    assert has_question_mark(None) is False


def test_has_question_mark_non_string_input():
    """Returns False when input is not a string."""
    assert has_question_mark(123) is False


def test_has_question_mark_whitespace_only():
    """Returns False when input contains only whitespace."""
    assert has_question_mark("   ") is False


def test_has_question_mark_question_mark_with_spaces():
    """Returns True when question mark is surrounded by whitespace."""
    assert has_question_mark(" ? ") is True


@pytest.mark.parametrize(
    "text, expected",
    [
        # Question mark
        ("Is this a question?", True),
        # WH words
        ("What is your name", True),
        ("How does this work", True),
        # Instruction-style prompts (depending on your implementation)
        ("Tell me how this works", True),
        ("Please explain this", True),
        # Interrogative start
        ("Do you like this", True),
        ("Can we proceed", True),
        # Multiple signals
        ("What is your name?", True),
        ("Can you explain this?", True),
        ("Please tell me what this means?", True),
        ("How can you fix this", True),
        # Clearly not questions
        ("This is a statement.", False),
        ("I like apples", False),
        ("Running quickly today", False),
    ],
)
def test_is_question_text_cases(text, expected):
    """Returns expected classification for a range of question
    and non-question text examples.
    """
    assert is_question(text) is expected


@pytest.mark.parametrize(
    "invalid_input",
    [None, 123, 12.5, [], {}, True],
)
def test_is_question_non_string(invalid_input):
    """Returns False for non-string inputs."""
    assert is_question(invalid_input) is False
