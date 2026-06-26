"""Tests for question structure functions."""
import pytest

from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    has_question_mark
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