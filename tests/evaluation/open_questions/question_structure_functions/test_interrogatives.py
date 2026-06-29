"""Tests for interrogative detection functions in question_structure_functions.py."""

import pytest

from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    has_interrogative_anywhere,
    has_interrogative_not_at_start,
    has_interrogative_start,
)

INTERROGATIVE_ANYWHERE_TRUE_CASES = [
    "What is this",
    "Tell me why this happens",
    "I want to know how it works",
    "Explain when this occurred",
    "Tell me where this is",
    "I wonder who is responsible",
    "Tell me whom this concerns",
    "I wonder whose idea this was",
    "Tell me which option is best",
]

INTERROGATIVE_ANYWHERE_FALSE_CASES = [
    "Is this correct",
    "This is a normal sentence",
    "Describe the process clearly",
    "Whatever you decide is fine",
    "The whole idea is simple",
    "",
    "   ",
]

INTERROGATIVE_START_TRUE_CASES = [
    # WH words
    "What is this",
    "Why does this happen",
    "How does it work",
    "When is this due",
    "Where is this located",
    "Who is responsible",
    "Whom does this affect",
    "Whose idea was this",
    "Which option is best",
    # auxiliary verbs
    "Is this correct",
    "Are we ready",
    "Do you understand",
    "Does this work",
    "Did you check",
    "Can we proceed",
    "Could you explain",
    "Would this help",
    "Should we continue",
    "Will this change",
    "Have you checked",
    "Has this been done",
    "Had this occurred",
]

INTERROGATIVE_NOT_AT_START_TRUE_CASES = [
    "Tell me what this is",
    "I wonder why that happens",
    "Explain how this works",
    "I need to know when this happened",
    "Tell me where this is located",
    "I want to know who is responsible",
    "Tell me whom this concerns",
    "I wonder whose idea this was",
    "Tell me which option is best",
]

ABSENT_INTERROGATIVE_CASES = [
    # Plain neutral sentences
    "This is a normal sentence",
    "The system works as expected",
    "We completed the analysis",
    # Instruction / statement (no WH, no auxiliary start)
    "Describe the process clearly",
    "Give an overview of the method",
    "Share your feedback",
    # Partial match traps (important!)
    "Whatever you decide is fine",
    "The whole idea is simple",
    "Someone handled this already",
    # Declarative sentences with verbs
    "It works well in practice",
    "He explained the results clearly",
]

EDGE_CASES = [
    None,
    123,
    12.5,
    True,
    [],
    {},
    (),
    "",
    "   ",
]


@pytest.mark.parametrize("text", INTERROGATIVE_ANYWHERE_TRUE_CASES)
def test_has_interrogative_anywhere_true(text):
    """Returns True when WH-word appears anywhere."""
    assert has_interrogative_anywhere(text) is True


@pytest.mark.parametrize("text", INTERROGATIVE_ANYWHERE_FALSE_CASES)
def test_has_interrogative_anywhere_false(text):
    """Returns False when no WH-word is present."""
    assert has_interrogative_anywhere(text) is False


@pytest.mark.parametrize("text", ABSENT_INTERROGATIVE_CASES)
def test_has_interrogative_anywhere_absent_case(text):
    """Returns False when no interrogative signal is present."""
    assert has_interrogative_anywhere(text) is False


@pytest.mark.parametrize("text", EDGE_CASES)
def test_has_interrogative_anywhere_edge_cases(text):
    """Returns False for all edge cases."""
    assert has_interrogative_anywhere(text) is False


@pytest.mark.parametrize("text", INTERROGATIVE_START_TRUE_CASES)
def test_has_interrogative_start_returns_true(text):
    """Returns True when text starts with interrogative or auxiliary."""
    assert has_interrogative_start(text) is True


def test_has_interrogative_start_case_insensitive():
    """Detects interrogatives regardless of casing."""
    assert has_interrogative_start("WHAT is this") is True
    assert has_interrogative_start("is this correct") is True


def test_has_interrogative_start_with_leading_whitespace():
    """Ignores leading whitespace."""
    assert has_interrogative_start("   What is this") is True


@pytest.mark.parametrize("text", INTERROGATIVE_NOT_AT_START_TRUE_CASES)
def test_has_interrogative_start_returns_false_for_middle(text):
    """Returns False when interrogative appears later."""
    assert has_interrogative_start(text) is False


def test_has_interrogative_start_punctuation():
    """Handles punctuation correctly."""
    assert has_interrogative_start("What? Really.") is True


@pytest.mark.parametrize("text", ABSENT_INTERROGATIVE_CASES)
def test_has_interrogative_start_false(text):
    """Returns False when no interrogative signal is present."""
    assert has_interrogative_start(text) is False


@pytest.mark.parametrize("text", EDGE_CASES)
def test_has_interrogative_start_edge_cases(text):
    """Returns False for all edge cases."""
    assert has_interrogative_start(text) is False


@pytest.mark.parametrize("text", INTERROGATIVE_NOT_AT_START_TRUE_CASES)
def test_has_interrogative_not_at_start_returns_true(text):
    """Returns True when WH-word appears later in the sentence."""
    assert has_interrogative_not_at_start(text) is True


@pytest.mark.parametrize("text", INTERROGATIVE_START_TRUE_CASES)
def test_has_interrogative_not_at_start_returns_false_for_start(text):
    """Returns False when interrogative is at the start."""
    assert has_interrogative_not_at_start(text) is False


def test_has_interrogative_not_at_start_case_insensitive():
    """Detects WH-words regardless of casing."""
    assert has_interrogative_not_at_start("Tell me WHAT this is") is True
    assert has_interrogative_not_at_start("WHY is that") is False


def test_has_interrogative_not_at_start_with_leading_whitespace():
    """Ignores leading whitespace."""
    assert has_interrogative_not_at_start("   Tell me what this is") is True
    assert has_interrogative_not_at_start("   What is this") is False


def test_has_interrogative_not_at_start_punctuation():
    """Handles punctuation correctly."""
    assert has_interrogative_not_at_start("Tell me, what is this?") is True
    assert has_interrogative_not_at_start("What? Really.") is False


@pytest.mark.parametrize("text", ABSENT_INTERROGATIVE_CASES)
def test_has_interrogative_not_at_start_false(text):
    """Returns False when no interrogative signal is present."""
    assert has_interrogative_not_at_start(text) is False


@pytest.mark.parametrize("text", EDGE_CASES)
def test_has_interrogative_not_at_start_edge_cases(text):
    """Returns False for all edge cases."""
    assert has_interrogative_not_at_start(text) is False
