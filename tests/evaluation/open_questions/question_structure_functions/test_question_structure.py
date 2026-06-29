"""Tests for question structure functions."""

from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    count_question_signals,
    has_question_mark,
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


def test_count_question_signals_no_signals():
    """No question signals should return 0."""
    text = "This is a statement."
    assert count_question_signals(text) == 0


def test_count_question_signals_question_mark_only():
    """Only a question mark should return 1."""
    text = "This is a question?"
    assert count_question_signals(text) == 1


def test_count_question_signals_interrogative_start_only():
    """Interrogative at start only."""
    text = "What is your name"
    assert count_question_signals(text) == 1


def test_count_question_signals_interrogative_not_at_start_only():
    """Interrogative not at start."""
    text = "I wonder what this is"
    assert count_question_signals(text) == 1


def test_count_question_signals_instruction_prompt_start_only():
    """Instruction prompt at start."""
    text = "Tell me your name"
    assert count_question_signals(text) == 1


def test_count_question_signals_instruction_prompt_not_at_start_only():
    """Instruction prompt not at start."""
    text = "I want you to tell me your name"
    assert count_question_signals(text) == 1


def test_count_question_signals_all_signals_present():
    """Returns the number of distinct signal types present."""
    text = (
        "What do you think? I wonder what this is. "
        "Tell me something. I want you to tell me more."
    )
    assert count_question_signals(text) == 3


def test_count_question_signals_multiple_occurrences_same_signal():
    """Multiple occurrences of the same signal should still count as 1
    because signals are distinct, not cumulative.
    """
    text = "What is this? What is that? What is anything?"
    assert count_question_signals(text) == 2


def test_count_question_signals_empty_string():
    """Empty input should return 0."""
    assert count_question_signals("") == 0


def test_count_question_signals_whitespace_string():
    """Whitespace-only input should return 0."""
    assert count_question_signals("   ") == 0


def test_count_question_signals_mixed_case_handling():
    """Case should not affect detection."""
    text = "WHAT is this?"
    assert count_question_signals(text) >= 1


def test_count_question_signals_punctuation_without_question():
    """Other punctuation should not count."""
    text = "This is surprising!"
    assert count_question_signals(text) == 0
