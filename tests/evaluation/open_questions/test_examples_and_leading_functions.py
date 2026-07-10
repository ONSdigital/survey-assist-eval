"""Tests for example and leading question helper functions."""

import pytest

from survey_assist_eval.evaluation.open_questions.examples_and_leading_functions import (
    has_closed_category_options,
    has_closed_category_without_examples,
    has_definition_example_wording,
    has_examples,
    has_explicit_example_marker,
    has_including_example_phrase,
)

# ============================================================================
# Test Data - Shared between tests
# ============================================================================


def unique_texts(*groups: list[str]) -> list[str]:
    """Return unique texts while preserving their original order."""
    return list(dict.fromkeys(text for group in groups for text in group))


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


EXPLICIT_EXAMPLE_MARKER_CASES = [
    "What is your employer's main activity, for example, "
    " providing finance, retail or social services?",
    "What products does your company make, for example, furniture or toys?",
    "What services does your organisation provide, for example, teaching or training?",
    "Do you consider your job to be in retail? E.g. selling goods in a shop.",
    "Do you work in manufacturing? E.g. making furniture or clothing.",
    "Are you self-employed? I.e. you run your own business.",
    "Do you work full time? I.e. 35 or more hours per week.",
    "What is your employer's main activity, such as hair cutting or teeth cleaning?",
    "What services do you provide, such as accounting or bookkeeping?",
    "What products does your employer manufacture (e.g. bicycles)?",
    "What service does your organisation provide (for example, hairdressing)?",
    "What type of work do you do (such as bookkeeping)?",
    "What industry do you work in (e.g. retail)?",
    "What products do you make (such as furniture)?",
]

INCLUDING_EXAMPLE_PHRASE_CASES = [
    "Are you a student or a worker, including a lecturer as a worker?",
    "Do you work in healthcare, including nursing roles?",
    "Does your role involve administration, including scheduling meetings?",
    "Do you work in manufacturing, for instance, producing furniture?",
    "What services do you provide, for instance, accounting or payroll support?",
    "What products does your employer make, for instance, bicycles or clothing?",
    "What services do you provide (for instance, accounting)?",
]

DEFINITION_EXAMPLE_WORDING_CASES = [
    "Do you work in retail, meaning you sell goods directly to customers?",
    "Are you self-employed, meaning you run your own business?",
    "Do you work in education, meaning you teach or train people?",
    "Are you self-employed, which means you run your own business?",
    "Do you work in retail, which means selling goods to customers?",
    "Do you work in healthcare, namely nursing or physiotherapy?",
    "Do you provide professional services, namely accounting or legal advice?",
    "Do you work in education, that is teaching or training?",
    "Do you work in healthcare, that is nursing or physiotherapy?",
]

CLOSED_CATEGORY_OPTION_CASES = [
    "What is your employer's main activity: teaching or research?",
    "Are you a student or a worker?",
    "Is your organisation mainly public or private?",
    "Are you employed in retail or manufacturing?",
    "Do you mainly provide products or services?",
    "Are you a manager or a supervisor?",
    "Do you work in the public or private sector?",
    "Are you involved in teaching or administration?",
    "What type of organisation do you work for: school, hospital or university?",
    "What kind of teacher are you: primary, secondary or college?",
    "Are you employed either full-time or part-time?",
    "Are you a manager/supervisor?",
    "Which of the following best describes your role?",
    "Which of these sectors do you work in?",
    "Select one of the following options.",
    "Select the one that best describes your role.",
    "Choose one of the following categories.",
    "Choose the one that best describes your organisation.",
    "Pick one of the following options.",
    "Pick the one that best matches your role.",
]

CLOSED_CATEGORY_WITH_EXAMPLE_CASES = [
    "What is your employer's main activity, for example, "
    "providing finance, retail or social services?",
    "What is your employer's main activity, such as hair cutting or teeth cleaning?",
    "Are you a student or a worker, including a lecturer as a worker?",
    "Do you see yourself as teenager or adult? I.e. 13-19 years old or 20+ years old.",
]

NON_EXAMPLE_CASES = [
    "I work in retail.",
    "The organisation provides healthcare services.",
    "Customer service activities form part of the role.",
    "The company manufactures furniture.",
    "Teaching apprentices is a key responsibility.",
    "The organisation mainly supports local businesses.",
    "What would you like to do next?",
    "The organisation's activities are mainly retail focused.",
    "Customer service is included within the role description.",
]

NON_CLOSED_CATEGORY_CASES = [
    "What services does your organisation provide?",
    "What products does your employer manufacture?",
    "Please describe your main duties.",
    "What is your job title?",
    "What type of work do you do?",
    "What products does your employer manufacture, for example bicycles?",
    "What services do you provide, such as accounting?",
    "Do you work in healthcare, like a nurse?",
    "Are you self-employed? I.e. you run your own business.",
]

ALL_EXAMPLE_CASES = unique_texts(
    EXPLICIT_EXAMPLE_MARKER_CASES,
    INCLUDING_EXAMPLE_PHRASE_CASES,
    DEFINITION_EXAMPLE_WORDING_CASES,
)


# ============================================================================
# Test has_explicit_example_marker function
# ============================================================================


@pytest.mark.parametrize("text", EXPLICIT_EXAMPLE_MARKER_CASES)
def test_has_explicit_example_marker_detects_explicit_examples(text: str):
    """Detect explicit example wording such as e.g. and for example."""
    assert has_explicit_example_marker(
        text
    ), f"Expected explicit example marker to be detected for: {text}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUTS)
def test_has_explicit_example_marker_empty_text(text):
    """Returns False for empty text inputs."""
    assert (
        has_explicit_example_marker(text) is False
    ), f"Expected False for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUTS)
def test_has_explicit_example_marker_non_string_inputs(text):
    """Returns False for non-string inputs."""
    assert (
        has_explicit_example_marker(text) is False
    ), f"Expected False for non-string input: {text!r}"


# ============================================================================
# Test has_including_example_phrase function
# ============================================================================


@pytest.mark.parametrize("text", INCLUDING_EXAMPLE_PHRASE_CASES)
def test_has_including_example_phrase_detects_including_style_examples(text: str):
    """Detect including-style example wording."""
    assert has_including_example_phrase(
        text
    ), f"Expected including-style example wording to be detected for: {text}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUTS)
def test_has_including_example_phrase_empty_text(text):
    """Returns False for empty text inputs."""
    assert (
        has_including_example_phrase(text) is False
    ), f"Expected False for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUTS)
def test_has_including_example_phrase_non_string_inputs(text):
    """Returns False for non-string inputs."""
    assert (
        has_including_example_phrase(text) is False
    ), f"Expected False for non-string input: {text!r}"


# ============================================================================
# Test has_definition_example_wording function
# ============================================================================


@pytest.mark.parametrize("text", DEFINITION_EXAMPLE_WORDING_CASES)
def test_has_definition_example_wording_detects_definition_style_examples(
    text: str,
):
    """Detect definition-style example wording."""
    assert has_definition_example_wording(
        text
    ), f"Expected definition-style example wording to be detected for: {text}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUTS)
def test_has_definition_example_wording_empty_text(text):
    """Returns False for empty text inputs."""
    assert (
        has_definition_example_wording(text) is False
    ), f"Expected False for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUTS)
def test_has_definition_example_wording_non_string_inputs(text):
    """Returns False for non-string inputs."""
    assert (
        has_definition_example_wording(text) is False
    ), f"Expected False for non-string input: {text!r}"


# ============================================================================
# Test has_examples function
# ============================================================================


@pytest.mark.parametrize("text", ALL_EXAMPLE_CASES)
def test_has_examples_detects_supported_example_patterns(text: str):
    """Detect example wording across all supported example patterns."""
    assert has_examples(text), f"Expected example wording to be detected for: {text}"


@pytest.mark.parametrize("text", NON_EXAMPLE_CASES)
def test_has_examples_ignores_non_example_wording(text: str):
    """Avoid flagging ordinary statements and similar wording as examples."""
    assert not has_examples(text), f"Did not expect example wording in: {text}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUTS)
def test_has_examples_empty_text(text):
    """Returns False for empty text inputs."""
    assert has_examples(text) is False, f"Expected False for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUTS)
def test_has_examples_non_string_inputs(text):
    """Returns False for non-string inputs."""
    assert has_examples(text) is False, f"Expected False for non-string input: {text!r}"


# ============================================================================
# Test has_closed_category_options function
# ============================================================================


@pytest.mark.parametrize("text", CLOSED_CATEGORY_OPTION_CASES)
def test_has_closed_category_options_detects_predefined_categories(text: str):
    """Detect closed-category response options in question wording."""
    assert has_closed_category_options(
        text
    ), f"Expected closed-category wording to be detected for: {text}"


@pytest.mark.parametrize("text", NON_CLOSED_CATEGORY_CASES)
def test_has_closed_category_options_ignores_open_questions(text: str):
    """Avoid flagging open questions that do not provide response options."""
    assert not has_closed_category_options(
        text
    ), f"Did not expect closed-category wording to be detected for: {text}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUTS)
def test_has_closed_category_options_empty_text(text):
    """Returns False for empty text inputs."""
    assert (
        has_closed_category_options(text) is False
    ), f"Expected False for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUTS)
def test_has_closed_category_options_non_string_inputs(text):
    """Returns False for non-string inputs."""
    assert (
        has_closed_category_options(text) is False
    ), f"Expected False for non-string input: {text!r}"


# ============================================================================
# Test has_closed_category_without_examples function
# ============================================================================


@pytest.mark.parametrize("text", CLOSED_CATEGORY_OPTION_CASES)
def test_has_closed_category_without_examples_detects_pure_categories(text: str):
    """Detect closed-category questions that do not contain examples."""
    assert has_closed_category_without_examples(
        text
    ), f"Expected closed-category wording without examples for: {text}"


@pytest.mark.parametrize(
    "text",
    unique_texts(CLOSED_CATEGORY_WITH_EXAMPLE_CASES, NON_CLOSED_CATEGORY_CASES),
)
def test_has_closed_category_without_examples_rejects_examples_or_open_text(
    text: str,
):
    """Avoid flagging questions that either contain examples or remain open-ended."""
    assert not has_closed_category_without_examples(text), (
        "Did not expect pure closed-category wording without examples for: " f"{text}"
    )


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUTS)
def test_has_closed_category_without_examples_empty_text(text):
    """Returns False for empty text inputs."""
    assert (
        has_closed_category_without_examples(text) is False
    ), f"Expected False for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUTS)
def test_has_closed_category_without_examples_non_string_inputs(text):
    """Returns False for non-string inputs."""
    assert (
        has_closed_category_without_examples(text) is False
    ), f"Expected False for non-string input: {text!r}"
