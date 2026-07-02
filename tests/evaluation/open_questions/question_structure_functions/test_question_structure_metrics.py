"""Tests for question structure metric functions."""

# pylint: disable=duplicate-code

import pytest

from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    contains_multiple_asks,
    get_question_structure_metrics,
    has_question_mark,
    is_question,
    is_single_question,
)

# ============================================================================
# Test Data - Shared between tests
# ============================================================================

NON_QUESTION_TEXT_INPUTS = [
    pytest.param("This is a statement.", id="statement"),
    pytest.param("I like apples", id="simple_statement"),
    pytest.param("Running quickly today", id="fragment"),
]

NON_STRING_INPUTS = [
    pytest.param(None, id="none_input"),
    pytest.param(123, id="integer_input"),
    pytest.param(12.5, id="float_input"),
    pytest.param([], id="list_input"),
    pytest.param({}, id="dict_input"),
    pytest.param(True, id="bool_input"),
]

EMPTY_TEXT_INPUTS = [
    pytest.param("", id="empty_string"),
    pytest.param("   ", id="whitespace_only"),
]

# ============================================================================
# Test has_question_mark function
# ============================================================================


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("Is this a question?", id="single_qmark"),
        pytest.param(" ? ", id="qmark_with_whitespace"),
        pytest.param("What?? Really??", id="multiple_qmarks"),
    ],
)
def test_has_question_mark_returns_true(text):
    """Returns True when a question mark is present."""
    assert (
        has_question_mark(text) is True
    ), f"Expected True when '?' is present in text: {text!r}"


@pytest.mark.parametrize("text", NON_QUESTION_TEXT_INPUTS)
def test_has_question_mark_returns_false_when_absent(text):
    """Returns False when a question mark is absent."""
    assert (
        has_question_mark(text) is False
    ), f"Expected False when '?' is absent in text: {text!r}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUTS)
def test_has_question_mark_empty_text(text):
    """Returns False for empty text inputs."""
    assert (
        has_question_mark(text) is False
    ), f"Expected False for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUTS)
def test_has_question_mark_non_string(text):
    """Returns False for non-string inputs."""
    assert has_question_mark(text) is False, (
        f"Expected False for non-string input: {text!r} "
        f"(type: {type(text).__name__})"
    )


# ============================================================================
# Test is_question function
# ============================================================================


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


@pytest.mark.parametrize("text", NON_QUESTION_TEXT_INPUTS)
def test_is_question_false_no_signals(text):
    """Returns False when no question signals are present."""
    assert (
        is_question(text) is False
    ), f"Expected False when no question signals are present: {text!r}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUTS)
def test_is_question_empty_text(text):
    """Returns False for empty text inputs."""
    assert is_question(text) is False, f"Expected False for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUTS)
def test_is_question_non_string(text):
    """Returns False for non-string inputs."""
    assert is_question(text) is False, (
        f"Expected False for non-string input: {text!r} "
        f"(type: {type(text).__name__})"
    )


# ============================================================================
# Test contains_multiple_asks function
# ============================================================================


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "What? Why?",
            id="two_question_groups",
        ),
        pytest.param(
            "Really?? Why??",
            id="multiple_question_groups",
        ),
        pytest.param(
            "What?? and why?",
            id="question_groups_with_conjunction",
        ),
    ],
)
def test_contains_multiple_asks_true_multiple_question_groups(text):
    """Returns True when multiple groups of question marks are present."""
    assert (
        contains_multiple_asks(text) is True
    ), f"Expected True for multiple question groups in: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("What??", id="single_question_group_double_qmark"),
        pytest.param("Really???", id="single_question_group_triple_qmark"),
    ],
)
def test_contains_multiple_asks_false_single_question_group(text):
    """Returns False when only one group of question marks is present."""
    assert (
        contains_multiple_asks(text) is False
    ), f"Expected False for a single question group in: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "What is your role, how long have you worked here",
            id="multiple_wh_no_conjunction",
        ),
        pytest.param(
            "What happened, when did it happen",
            id="multiple_wh_different_types",
        ),
    ],
)
def test_contains_multiple_asks_true_multiple_wh_interrogatives(text):
    """Returns True when multiple WH interrogatives are present."""
    assert (
        contains_multiple_asks(text) is True
    ), f"Expected True for multiple WH interrogatives in: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "Describe your role. Explain your responsibilities.",
            id="multiple_instructions_no_conjunction",
        ),
        pytest.param(
            "Tell me about your role. Describe your experience.",
            id="multiple_instruction_types",
        ),
    ],
)
def test_contains_multiple_asks_true_multiple_instruction_prompts(text):
    """Returns True when multiple instruction prompts are present."""
    assert (
        contains_multiple_asks(text) is True
    ), f"Expected True for multiple instruction prompts in: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "What is your name and where do you live",
            id="wh_with_and",
        ),
        pytest.param(
            "Can you explain this or give an example",
            id="modal_with_or",
        ),
        pytest.param(
            "Tell me how this works and also describe the output",
            id="instruction_with_and_also",
        ),
    ],
)
def test_contains_multiple_asks_true_conjunctions(text):
    """Returns True when clauses are joined by conjunctions."""
    assert (
        contains_multiple_asks(text) is True
    ), f"Expected True for conjunction-linked clauses in: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "What is this and how does it work?",
            id="wh_plus_conjunction_plus_qmark",
        ),
        pytest.param(
            "Explain this and describe the process?",
            id="instruction_plus_conjunction_plus_qmark",
        ),
        pytest.param(
            "Can you explain what happened and why it occurred",
            id="interrogative_start_and_multiple_wh",
        ),
    ],
)
def test_contains_multiple_asks_true_multiple_signals(text):
    """Returns True when multiple compound-question signals are present."""
    assert (
        contains_multiple_asks(text) is True
    ), f"Expected True for multiple compound-question signals in: {text!r}"


def test_contains_multiple_asks_case_insensitive():
    """Returns True regardless of casing."""
    text = "WHAT IS YOUR ROLE AND HOW LONG HAVE YOU WORKED HERE"

    assert (
        contains_multiple_asks(text) is True
    ), f"Expected case-insensitive detection for: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("What is your name?", id="single_question"),
        pytest.param("How does this work", id="single_wh"),
        pytest.param("Explain this", id="single_instruction"),
        pytest.param("Can you help me", id="single_interrogative_start"),
    ],
)
def test_contains_multiple_asks_false_single_question(text):
    """Returns False for non-compound questions."""
    assert (
        contains_multiple_asks(text) is False
    ), f"Expected False for single non-compound question: {text!r}"


@pytest.mark.parametrize("text", NON_QUESTION_TEXT_INPUTS)
def test_contains_multiple_asks_false_non_question(text):
    """Returns False for non-question text."""
    assert (
        contains_multiple_asks(text) is False
    ), f"Expected False for non-question text: {text!r}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUTS)
def test_contains_multiple_asks_empty_text(text):
    """Returns False for empty text inputs."""
    assert (
        contains_multiple_asks(text) is False
    ), f"Expected False for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUTS)
def test_contains_multiple_asks_non_string(text):
    """Returns False for non-string inputs."""
    assert contains_multiple_asks(text) is False, (
        f"Expected False for non-string input: {text!r} "
        f"(type: {type(text).__name__})"
    )


# ============================================================================
# Test is_single_question function
# ============================================================================


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("What is your name?", id="single_wh_question"),
        pytest.param("How does this work", id="single_wh_no_qmark"),
        pytest.param("Can you explain this?", id="single_interrogative_start"),
        pytest.param("Explain this", id="single_instruction_prompt"),
        pytest.param("What is this??", id="single_question_double_qmark"),
        pytest.param("Really???", id="single_question_triple_qmark"),
    ],
)
def test_is_single_question_true_valid_single_questions(text):
    """Returns True for valid non-compound questions."""
    assert (
        is_single_question(text) is True
    ), f"Expected True for valid single question: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("What? Why?", id="multiple_question_groups"),
        pytest.param("Really?? Why??", id="multiple_question_groups_emphasis"),
        pytest.param("What?? and why?", id="multiple_question_groups_with_and"),
    ],
)
def test_is_single_question_false_multiple_question_groups(text):
    """Returns False when multiple question groups are present."""
    assert (
        is_single_question(text) is False
    ), f"Expected False for multiple question groups in: {text!r}"


@pytest.mark.parametrize("text", NON_QUESTION_TEXT_INPUTS)
def test_is_single_question_false_non_questions(text):
    """Returns False when no question signals are present."""
    assert (
        is_single_question(text) is False
    ), f"Expected False for non-question text: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "What is your name and where do you live",
            id="compound_wh_and",
        ),
        pytest.param(
            "Can you explain this or give an example",
            id="compound_or",
        ),
        pytest.param(
            "Describe your role. Explain your responsibilities.",
            id="multiple_instruction_prompts",
        ),
        pytest.param(
            "What happened, when did it happen",
            id="multiple_wh_interrogatives",
        ),
    ],
)
def test_is_single_question_false_contains_multiple_askss(text):
    """Returns False for compound questions."""
    assert (
        is_single_question(text) is False
    ), f"Expected False for compound question: {text!r}"


def test_is_single_question_strips_whitespace():
    """Returns True when leading and trailing whitespace is present."""
    text = "   What is your name?   "

    assert (
        is_single_question(text) is True
    ), f"Expected True after stripping whitespace from: {text!r}"


def test_is_single_question_case_insensitive():
    """Returns True regardless of input casing."""
    text = "WHAT IS YOUR NAME?"

    assert (
        is_single_question(text) is True
    ), f"Expected case-insensitive detection for: {text!r}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUTS)
def test_is_single_question_empty_text(text):
    """Returns False for empty text inputs."""
    assert (
        is_single_question(text) is False
    ), f"Expected False for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUTS)
def test_is_single_question_non_string(text):
    """Returns False for non-string inputs."""
    assert is_single_question(text) is False, (
        f"Expected False for non-string input: {text!r} "
        f"(type: {type(text).__name__})"
    )


# ============================================================================
# Test get_question_structure_metrics function
# ============================================================================

EXPECTED_FALSE_QUESTION_STRUCTURE_METRICS = {
    "has_question_mark": False,
    "interrogative_start": False,
    "instruction_prompt_start": False,
    "interrogative_wh_count": 0,
    "instruction_prompt_count": 0,
    "is_question": False,
    "contains_multiple_asks": False,
    "is_single_question": False,
}


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param(
            "What is your name?",
            {
                "has_question_mark": True,
                "interrogative_start": True,
                "instruction_prompt_start": False,
                "interrogative_wh_count": 1,
                "instruction_prompt_count": 0,
                "is_question": True,
                "contains_multiple_asks": False,
                "is_single_question": True,
            },
            id="single_wh_question",
        ),
        pytest.param(
            "How does this work",
            {
                "has_question_mark": False,
                "interrogative_start": True,
                "instruction_prompt_start": False,
                "interrogative_wh_count": 1,
                "instruction_prompt_count": 0,
                "is_question": True,
                "contains_multiple_asks": False,
                "is_single_question": True,
            },
            id="single_wh_no_qmark",
        ),
        pytest.param(
            "Please explain this",
            {
                "has_question_mark": False,
                "interrogative_start": False,
                "instruction_prompt_start": True,
                "interrogative_wh_count": 0,
                "instruction_prompt_count": 1,
                "is_question": True,
                "contains_multiple_asks": False,
                "is_single_question": True,
            },
            id="single_instruction_prompt",
        ),
        pytest.param(
            "What happened and why did it happen?",
            {
                "has_question_mark": True,
                "interrogative_start": True,
                "instruction_prompt_start": False,
                "interrogative_wh_count": 2,
                "instruction_prompt_count": 0,
                "is_question": True,
                "contains_multiple_asks": True,
                "is_single_question": False,
            },
            id="multiple_wh_interrogatives",
        ),
        pytest.param(
            "Describe your role. Explain your responsibilities.",
            {
                "has_question_mark": False,
                "interrogative_start": False,
                "instruction_prompt_start": True,
                "interrogative_wh_count": 0,
                "instruction_prompt_count": 2,
                "is_question": True,
                "contains_multiple_asks": True,
                "is_single_question": False,
            },
            id="multiple_instruction_prompts",
        ),
        pytest.param(
            "What is your name? Where do you live?",
            {
                "has_question_mark": True,
                "interrogative_start": True,
                "instruction_prompt_start": False,
                "interrogative_wh_count": 2,
                "instruction_prompt_count": 0,
                "is_question": True,
                "contains_multiple_asks": True,
                "is_single_question": False,
            },
            id="multiple_separate_questions",
        ),
    ],
)
def test_get_question_structure_metrics_returns_expected_metrics(text, expected):
    """Returns expected question structure metrics."""
    assert (
        get_question_structure_metrics(text) == expected
    ), f"Expected question structure metrics for {text!r} to be {expected}"


@pytest.mark.parametrize("text", NON_QUESTION_TEXT_INPUTS)
def test_get_question_structure_metrics_non_question_text(text):
    """Returns falsey question structure metrics for non-question text."""
    assert (
        get_question_structure_metrics(text)
        == EXPECTED_FALSE_QUESTION_STRUCTURE_METRICS
    ), f"Expected falsey question structure metrics for non-question text: {text!r}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUTS)
def test_get_question_structure_metrics_empty_text(text):
    """Returns falsey question structure metrics for empty text inputs."""
    assert (
        get_question_structure_metrics(text)
        == EXPECTED_FALSE_QUESTION_STRUCTURE_METRICS
    ), f"Expected falsey question structure metrics for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUTS)
def test_get_question_structure_metrics_non_string(text):
    """Returns falsey question structure metrics for non-string inputs."""
    assert (
        get_question_structure_metrics(text)
        == EXPECTED_FALSE_QUESTION_STRUCTURE_METRICS
    ), (
        f"Expected falsey question structure metrics for non-string input: {text!r} "
        f"(type: {type(text).__name__})"
    )


def test_get_question_structure_metrics_returns_expected_keys():
    """Returns all expected question structure metric keys."""
    expected_keys = {
        "has_question_mark",
        "interrogative_start",
        "instruction_prompt_start",
        "interrogative_wh_count",
        "instruction_prompt_count",
        "is_question",
        "contains_multiple_asks",
        "is_single_question",
    }

    assert set(get_question_structure_metrics("What is your name?")) == (
        expected_keys
    ), "Expected get_question_structure_metrics to return all expected keys"
