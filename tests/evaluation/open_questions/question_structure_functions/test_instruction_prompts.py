"""Tests for instruction prompt detection functions in question_structure_functions.py."""

import pytest

from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    count_instruction_prompts,
    has_instruction_prompt_not_at_start,
    has_instruction_prompt_start,
)

INSTRUCTION_START_TRUE_CASES = [
    "Tell me what happened",
    "Describe your role",
    "Explain this clearly",
    "Please describe your role",
    "Please explain the process",
    "Please tell me more",
    "Please share your feedback",
    "Share your feedback",
    "Give details of the issue",
]

INSTRUCTION_NOT_AT_START_TRUE_CASES = [
    "Can you tell me what happened",
    "I want you to explain this",
    "Could you please describe your role",
    "Can you please explain this process",
    "I need you to please tell me more",
    "Could you please share the details",
    "We need you to give details of this",
]

ABSENT_INSTRUCTION_CASES = [
    # Neutral statements
    "This is a normal sentence",
    "The system works as expected",
    "We completed the analysis",
    # Interrogatives (but not instructions)
    "What is this",
    "Why does this happen",
    "How does it work",
    # Partial match traps
    "This is a shareholder report",
    "Descriptive text only",
    # Declarative sentences
    "He explained the results clearly",
    "It works well in practice",
]

EDGE_CASES = [
    "",
    "   ",
    123,
    12.5,
    True,
    None,
    [],
    {},
    (),
]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Please describe your role", 1),
        ("Explain the process", 1),
        ("Can you share your feedback", 1),
        ("I need you to give details of this", 1),
        ("Please tell me more", 1),
        ("Could you please explain this", 1),
        ("Tell me what happened", 1),
        ("Tell me and explain this", 2),
        ("Please describe and explain your role", 2),
        ("Tell me, describe, and explain", 3),
        ("Can you please describe the bird and describe yourself", 2),
    ],
)
def test_count_instruction_prompts_multiple(text, expected):
    """Counts multiple instruction prompts in a single string."""
    assert count_instruction_prompts(text) == expected


@pytest.mark.parametrize("text", EDGE_CASES)
def test_count_instruction_prompts_edge_cases(text):
    """Returns 0 for non-string and empty inputs."""
    assert count_instruction_prompts(text) == 0


@pytest.mark.parametrize("text", INSTRUCTION_START_TRUE_CASES)
def test_has_instruction_prompt_start_true(text):
    """Returns True when instruction prompt is at the start."""
    assert has_instruction_prompt_start(text) is True


@pytest.mark.parametrize("text", INSTRUCTION_NOT_AT_START_TRUE_CASES)
def test_has_instruction_prompt_start_false_for_middle(text):
    """Returns False when instruction appears later in text."""
    assert has_instruction_prompt_start(text) is False


@pytest.mark.parametrize("text", ABSENT_INSTRUCTION_CASES)
def test_has_instruction_prompt_start_false(text):
    """Returns False when no instruction prompt is present."""
    assert has_instruction_prompt_start(text) is False


def test_has_instruction_prompt_start_case_insensitive():
    """Detects instruction prompts regardless of casing."""
    assert has_instruction_prompt_start("PLEASE DESCRIBE your role") is True
    assert has_instruction_prompt_start("tell me more") is True


def test_has_instruction_prompt_start_with_leading_whitespace():
    """Ignores leading whitespace."""
    assert has_instruction_prompt_start("   Tell me what happened") is True


@pytest.mark.parametrize("text", EDGE_CASES)
def test_has_instruction_prompt_start_edge_cases(text):
    """Returns False for edge cases."""
    assert has_instruction_prompt_start(text) is False


@pytest.mark.parametrize("text", INSTRUCTION_NOT_AT_START_TRUE_CASES)
def test_has_instruction_prompt_not_at_start_true(text):
    """Returns True when instruction prompt appears later."""
    assert has_instruction_prompt_not_at_start(text) is True


@pytest.mark.parametrize("text", INSTRUCTION_START_TRUE_CASES)
def test_has_instruction_prompt_not_at_start_false_for_start(text):
    """Returns False when instruction is at the start."""
    assert has_instruction_prompt_not_at_start(text) is False


@pytest.mark.parametrize("text", ABSENT_INSTRUCTION_CASES)
def test_has_instruction_prompt_not_at_start_false(text):
    """Returns False when no instruction prompt is present."""
    assert has_instruction_prompt_not_at_start(text) is False


def test_has_instruction_prompt_not_at_start_case_insensitive():
    """Detects instruction prompts regardless of casing."""
    assert has_instruction_prompt_not_at_start("Can you TELL ME more") is True
    assert has_instruction_prompt_not_at_start("PLEASE EXPLAIN this") is False


def test_has_instruction_prompt_not_at_start_with_leading_whitespace():
    """Handles leading whitespace correctly."""
    assert has_instruction_prompt_not_at_start("   Tell me more") is False
    assert has_instruction_prompt_not_at_start("   Can you tell me more") is True


def test_has_instruction_prompt_not_at_start_punctuation():
    """Handles punctuation correctly."""
    assert (
        has_instruction_prompt_not_at_start("Can you, tell me what happened?") is True
    )
    assert has_instruction_prompt_not_at_start("Tell me, what happened?") is False


@pytest.mark.parametrize("text", EDGE_CASES)
def test_has_instruction_prompt_not_at_start_edge_cases(text):
    """Returns False for edge cases."""
    assert has_instruction_prompt_not_at_start(text) is False
