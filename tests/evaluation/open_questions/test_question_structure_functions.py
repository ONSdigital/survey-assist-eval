"""Tests for question structure functions."""
import pytest

from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    has_question_mark,
    has_interrogative_not_at_start,
    has_interrogative_start,
    has_instruction_prompt_not_at_start
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


def test_has_interrogative_not_at_start_returns_false_when_at_start():
    """Returns False when WH-word is the first word."""
    assert has_interrogative_not_at_start("What is this") is False
    assert has_interrogative_not_at_start("Why does this happen") is False


def test_has_interrogative_not_at_start_returns_true_when_in_middle():
    """Returns True when WH-word appears later in the sentence."""
    assert has_interrogative_not_at_start("Tell me what this is") is True
    assert has_interrogative_not_at_start("I wonder why that happens") is True


def test_has_interrogative_not_at_start_case_insensitive():
    """Detects WH-words regardless of casing."""
    assert has_interrogative_not_at_start("Tell me WHAT this is") is True
    assert has_interrogative_not_at_start("WHY is that") is False


def test_has_interrogative_not_at_start_with_leading_whitespace():
    """Ignores leading whitespace when determining the first word."""
    assert has_interrogative_not_at_start("   What is this") is False
    assert has_interrogative_not_at_start("   Tell me what this is") is True


def test_has_interrogative_not_at_start_returns_false_when_absent():
    """Returns False when no WH-word is present."""
    assert has_interrogative_not_at_start("This is a statement") is False


def test_has_interrogative_not_at_start_word_boundaries():
    """Does not match partial words containing WH substrings."""
    assert has_interrogative_not_at_start("This is whatever you want") is False
    assert has_interrogative_not_at_start("The whole thing") is False


def test_has_interrogative_not_at_start_punctuation():
    """Handles punctuation correctly around WH-words."""
    assert has_interrogative_not_at_start("Tell me, what is this?") is True
    assert has_interrogative_not_at_start("What? Really.") is False


def test_has_interrogative_not_at_start_empty_string():
    """Returns False for empty string input."""
    assert has_interrogative_not_at_start("") is False


def test_has_interrogative_not_at_start_none_input():
    """Returns False when input is None."""
    assert has_interrogative_not_at_start(None) is False


def test_has_interrogative_not_at_start_non_string_input():
    """Returns False when input is not a string."""
    assert has_interrogative_not_at_start(123) is False


def test_has_interrogative_not_at_start_whitespace_only():
    """Returns False for whitespace-only input."""
    assert has_interrogative_not_at_start("   ") is False


def test_has_interrogative_start_returns_true_for_wh_words():
    """Returns True when text starts with a WH-word."""
    assert has_interrogative_start("What is this") is True
    assert has_interrogative_start("Why does this happen") is True
    assert has_interrogative_start("How does it work") is True


def test_has_interrogative_start_returns_true_for_auxiliary_verbs():
    """Returns True when text starts with an auxiliary verb."""
    assert has_interrogative_start("Is this correct") is True
    assert has_interrogative_start("Do you understand") is True
    assert has_interrogative_start("Can we proceed") is True


def test_has_interrogative_start_case_insensitive():
    """Detects interrogatives regardless of casing."""
    assert has_interrogative_start("WHAT is this") is True
    assert has_interrogative_start("is this correct") is True


def test_has_interrogative_start_with_leading_whitespace():
    """Ignores leading whitespace when checking the start of text."""
    assert has_interrogative_start("   What is this") is True
    assert has_interrogative_start("   Do you agree") is True


def test_has_interrogative_start_returns_false_when_not_at_start():
    """Returns False when interrogative appears later in the text."""
    assert has_interrogative_start("Tell me what this is") is False
    assert has_interrogative_start("I wonder why that happens") is False


def test_has_interrogative_start_returns_false_when_absent():
    """Returns False when no interrogative or auxiliary verb is present."""
    assert has_interrogative_start("This is a statement") is False


def test_has_interrogative_start_word_boundaries():
    """Does not match partial words containing interrogative substrings."""
    assert has_interrogative_start("Whatever happens") is False
    assert has_interrogative_start("The whole idea") is False


def test_has_interrogative_start_with_punctuation():
    """Handles punctuation correctly at the start of text."""
    assert has_interrogative_start("What? Really.") is True
    assert has_interrogative_start("Is this okay?") is True


def test_has_interrogative_start_empty_string():
    """Returns False for empty string input."""
    assert has_interrogative_start("") is False


def test_has_interrogative_start_none_input():
    """Returns False when input is None."""
    assert has_interrogative_start(None) is False


def test_has_interrogative_start_non_string_input():
    """Returns False when input is not a string."""
    assert has_interrogative_start(123) is False


def test_has_interrogative_start_whitespace_only():
    """Returns False for whitespace-only input."""
    assert has_interrogative_start("   ") is False


def test_has_instruction_prompt_not_at_start_returns_false_when_at_start():
    """Returns False when instruction prompt is the first phrase."""
    assert has_instruction_prompt_not_at_start("Tell me what happened") is False
    assert has_instruction_prompt_not_at_start("Explain this clearly") is False
    assert has_instruction_prompt_not_at_start("Please describe your role") is False


def test_has_instruction_prompt_not_at_start_returns_true_when_in_middle():
    """Returns True when instruction prompt appears later in the text."""
    assert has_instruction_prompt_not_at_start("Can you tell me what happened") is True
    assert has_instruction_prompt_not_at_start("I want you to explain this") is True
    assert has_instruction_prompt_not_at_start("Could you please describe your role") is True


def test_has_instruction_prompt_not_at_start_case_insensitive():
    """Detects instruction prompts regardless of casing."""
    assert has_instruction_prompt_not_at_start("Can you TELL ME more") is True
    assert has_instruction_prompt_not_at_start("PLEASE EXPLAIN this") is False


def test_has_instruction_prompt_not_at_start_with_leading_whitespace():
    """Ignores leading whitespace when determining the start."""
    assert has_instruction_prompt_not_at_start("   Tell me more") is False
    assert has_instruction_prompt_not_at_start("   Can you tell me more") is True


def test_has_instruction_prompt_not_at_start_returns_false_when_absent():
    """Returns False when no instruction prompt is present."""
    assert has_instruction_prompt_not_at_start("This is a normal sentence") is False


def test_has_instruction_prompt_not_at_start_word_boundaries():
    """Does not match partial words containing instruction substrings."""
    assert has_instruction_prompt_not_at_start("This is a shareholder report") is False
    assert has_instruction_prompt_not_at_start("Descriptive text only") is False


def test_has_instruction_prompt_not_at_start_with_punctuation():
    """Handles punctuation correctly around instruction prompts."""
    assert has_instruction_prompt_not_at_start("Can you, tell me what happened?") is True
    assert has_instruction_prompt_not_at_start("Tell me, what happened?") is False


def test_has_instruction_prompt_not_at_start_empty_string():
    """Returns False for empty string input."""
    assert has_instruction_prompt_not_at_start("") is False


def test_has_instruction_prompt_not_at_start_none_input():
    """Returns False when input is None."""
    assert has_instruction_prompt_not_at_start(None) is False


def test_has_instruction_prompt_not_at_start_non_string_input():
    """Returns False when input is not a string."""
    assert has_instruction_prompt_not_at_start(123) is False


def test_has_instruction_prompt_not_at_start_whitespace_only():
    """Returns False for whitespace-only input."""
    assert has_instruction_prompt_not_at_start("   ") is False
