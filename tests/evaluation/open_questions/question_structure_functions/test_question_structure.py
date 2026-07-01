"""Tests for question structure functions."""

import pytest

from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    has_question_mark,
    is_question,
)


def test_has_question_mark_returns_true_when_present():
    """Returns True when the text contains a question mark."""
    text = "Is this a question?"
    assert (
        has_question_mark(text) is True
    ), f"Expected True when '?' is present in text: {text!r}"


def test_has_question_mark_returns_false_when_absent():
    """Returns False when the text does not contain a question mark."""
    text = "This is a statement."
    assert (
        has_question_mark(text) is False
    ), f"Expected False when '?' is absent in text: {text!r}"


def test_has_question_mark_multiple_question_marks():
    """Returns True when multiple question marks are present."""
    text = "What?? Really??"
    assert (
        has_question_mark(text) is True
    ), f"Expected True when multiple '?' are present: {text!r}"


def test_has_question_mark_empty_string():
    """Returns False when input text is empty."""
    text = ""
    assert (
        has_question_mark(text) is False
    ), f"Expected False for empty string input: {text!r}"


def test_has_question_mark_none_input():
    """Returns False when input is None."""
    text = None
    assert (
        has_question_mark(text) is False
    ), "Expected False for None input, got unexpected result"


def test_has_question_mark_non_string_input():
    """Returns False when input is not a string."""
    text = 123
    assert (
        has_question_mark(text) is False
    ), f"Expected False for non-string input: {text!r} (type: {type(text).__name__})"


def test_has_question_mark_whitespace_only():
    """Returns False when input contains only whitespace."""
    text = "   "
    assert (
        has_question_mark(text) is False
    ), f"Expected False for whitespace-only input: {text!r}"


def test_has_question_mark_question_mark_with_spaces():
    """Returns True when question mark is surrounded by whitespace."""
    text = " ? "
    assert (
        has_question_mark(text) is True
    ), f"Expected True when '?' is surrounded by whitespace: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("Is this a question?", id="question_mark"),
        pytest.param("What is your name", id="wh_word_what"),
        pytest.param("How does this work", id="wh_word_how"),
        pytest.param("Tell me how this works", id="instruction_tell_me"),
        pytest.param("Please explain this", id="instruction_please_explain"),
        pytest.param("Do you like this", id="auxiliary_do"),
        pytest.param("Can we proceed", id="modal_can"),
    ],
)
def test_is_question_true_single_signal(text):
    """Returns True when exactly one question signal is present."""
    assert (
        is_question(text) is True
    ), f"Expected True for single question signal in: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("What is your name?", id="wh_word_plus_qmark"),
        pytest.param("Can you explain this?", id="modal_plus_qmark"),
        pytest.param(
            "Please tell me what this means?",
            id="instruction_wh_word_qmark",
        ),
        pytest.param("How can you fix this", id="wh_word_plus_modal"),
    ],
)
def test_is_question_true_multiple_signals(text):
    """Returns True when multiple question signals are present."""
    assert (
        is_question(text) is True
    ), f"Expected True for multiple question signals in: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("This is a statement.", id="statement_sentence"),
        pytest.param("I like apples", id="simple_statement"),
        pytest.param("Running quickly today", id="fragment_not_question"),
    ],
)
def test_is_question_false_no_signals(text):
    """Returns False when no question signals are present."""
    assert (
        is_question(text) is False
    ), f"Expected False when no question signals are present: {text!r}"


@pytest.mark.parametrize(
    "invalid_input",
    [
        pytest.param(None, id="none_input"),
        pytest.param(123, id="int_input"),
        pytest.param(12.5, id="float_input"),
        pytest.param([], id="list_input"),
        pytest.param({}, id="dict_input"),
        pytest.param(True, id="bool_input"),
    ],
)
def test_is_question_non_string(invalid_input):
    """Returns False for non-string inputs."""
    assert is_question(invalid_input) is False, (
        f"Expected False for invalid input: {invalid_input!r} "
        f"(type: {type(invalid_input).__name__})"
    )
