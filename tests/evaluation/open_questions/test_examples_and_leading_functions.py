"""Tests for example and leading question helper functions."""

# pylint: disable=redefined-outer-name
# pylint: disable=duplicate-code

import pandas as pd
import pytest

from survey_assist_eval.evaluation.open_questions.examples_and_leading_functions import (
    ExampleLeadingQuestionMetrics,
    add_example_and_leading_columns,
    compute_example_and_leading_metrics,
    get_example_and_leading_metrics,
    has_closed_category_options,
    has_closed_category_without_examples,
    has_definition_example_wording,
    has_examples,
    has_explicit_example_marker,
    has_including_example_phrase,
    summarise_example_and_leading_columns,
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
    pytest.param(
        "What is your employer's main activity, for example, "
        "providing finance, retail or social services?",
        id="explicit_for_example_comma_list",
    ),
    pytest.param(
        "What products does your company make, for example, furniture or toys?",
        id="explicit_for_example_or_list",
    ),
    pytest.param(
        "What services does your organisation provide, for example, teaching or training?",
        id="explicit_for_example_activity_examples",
    ),
    pytest.param(
        "Do you consider your job to be in retail? E.g. selling goods in a shop.",
        id="explicit_eg_follow_on_sentence",
    ),
    pytest.param(
        "Do you work in manufacturing? E.g. making furniture or clothing.",
        id="explicit_eg_follow_on_or_list",
    ),
    pytest.param(
        "Are you self-employed? I.e. you run your own business.",
        id="explicit_ie_follow_on_definition",
    ),
    pytest.param(
        "Do you work full time? I.e. 35 or more hours per week.",
        id="explicit_ie_follow_on_hours_definition",
    ),
    pytest.param(
        "What is your employer's main activity, such as hair cutting or teeth cleaning?",
        id="explicit_such_as_or_list",
    ),
    pytest.param(
        "What services do you provide, such as accounting or bookkeeping?",
        id="explicit_such_as_service_examples",
    ),
    pytest.param(
        "What products does your employer manufacture (e.g. bicycles)?",
        id="explicit_parenthetical_eg",
    ),
    pytest.param(
        "What service does your organisation provide (for example, hairdressing)?",
        id="explicit_parenthetical_for_example",
    ),
    pytest.param(
        "What type of work do you do (such as bookkeeping)?",
        id="explicit_parenthetical_such_as",
    ),
    pytest.param(
        "What industry do you work in (e.g. retail)?",
        id="explicit_parenthetical_eg_single_example",
    ),
    pytest.param(
        "What products do you make (such as furniture)?",
        id="explicit_parenthetical_such_as_single_example",
    ),
]


INCLUDING_EXAMPLE_PHRASE_CASES = [
    pytest.param(
        "Are you a student or a worker, including a lecturer as a worker?",
        id="including_example_phrase_with_or",
    ),
    pytest.param(
        "Do you work in healthcare, including nursing roles?",
        id="including_example_phrase_single_role",
    ),
    pytest.param(
        "Does your role involve administration, including scheduling meetings?",
        id="including_example_phrase_activity",
    ),
    pytest.param(
        "Do you work in manufacturing, for instance, producing furniture?",
        id="for_instance_example_phrase_activity",
    ),
    pytest.param(
        "What services do you provide, for instance, accounting or payroll support?",
        id="for_instance_example_phrase_or_list",
    ),
    pytest.param(
        "What products does your employer make, for instance, bicycles or clothing?",
        id="for_instance_example_phrase_product_list",
    ),
    pytest.param(
        "What services do you provide (for instance, accounting)?",
        id="parenthetical_for_instance_example_phrase",
    ),
]


DEFINITION_EXAMPLE_WORDING_CASES = [
    pytest.param(
        "Do you work in retail, meaning you sell goods directly to customers?",
        id="definition_meaning_retail",
    ),
    pytest.param(
        "Are you self-employed, meaning you run your own business?",
        id="definition_meaning_self_employed",
    ),
    pytest.param(
        "Do you work in education, meaning you teach or train people?",
        id="definition_meaning_education",
    ),
    pytest.param(
        "Are you self-employed, which means you run your own business?",
        id="definition_which_means_self_employed",
    ),
    pytest.param(
        "Do you work in retail, which means selling goods to customers?",
        id="definition_which_means_retail",
    ),
    pytest.param(
        "Do you work in healthcare, namely nursing or physiotherapy?",
        id="definition_namely_healthcare_or_list",
    ),
    pytest.param(
        "Do you provide professional services, namely accounting or legal advice?",
        id="definition_namely_professional_services",
    ),
    pytest.param(
        "Do you work in education, that is teaching or training?",
        id="definition_that_is_education",
    ),
    pytest.param(
        "Do you work in healthcare, that is nursing or physiotherapy?",
        id="definition_that_is_healthcare",
    ),
]


CLOSED_CATEGORY_OPTION_CASES = [
    pytest.param(
        "What is your employer's main activity: teaching or research?",
        id="closed_category_colon_with_or",
    ),
    pytest.param("Are you a student or a worker?", id="closed_category_simple_or"),
    pytest.param(
        "Is your organisation mainly public or private?",
        id="closed_category_public_or_private",
    ),
    pytest.param(
        "Are you employed in retail or manufacturing?", id="closed_category_sector_or"
    ),
    pytest.param(
        "Do you mainly provide products or services?",
        id="closed_category_products_or_services",
    ),
    pytest.param("Are you a manager or a supervisor?", id="closed_category_role_or"),
    pytest.param(
        "Do you work in the public or private sector?",
        id="closed_category_sector_phrase",
    ),
    pytest.param(
        "Are you involved in teaching or administration?",
        id="closed_category_activity_or",
    ),
    pytest.param(
        "What type of organisation do you work for: school, hospital or university?",
        id="closed_category_colon_comma_or_list",
    ),
    pytest.param(
        "What kind of teacher are you: primary, secondary or college?",
        id="closed_category_colon_teacher_options",
    ),
    pytest.param(
        "Are you employed either full-time or part-time?",
        id="closed_category_either_or",
    ),
    pytest.param("Are you a manager/supervisor?", id="closed_category_slash_options"),
    pytest.param(
        "Which of the following best describes your role?",
        id="closed_category_which_of_the_following",
    ),
    pytest.param(
        "Which of these sectors do you work in?", id="closed_category_which_of_these"
    ),
    pytest.param(
        "Select one of the following options.", id="closed_category_select_one"
    ),
    pytest.param(
        "Select the one that best describes your role.",
        id="closed_category_select_the_one",
    ),
    pytest.param(
        "Choose one of the following categories.", id="closed_category_choose_one"
    ),
    pytest.param(
        "Choose the one that best describes your organisation.",
        id="closed_category_choose_the_one",
    ),
    pytest.param("Pick one of the following options.", id="closed_category_pick_one"),
    pytest.param(
        "Pick the one that best matches your role.", id="closed_category_pick_the_one"
    ),
]


CLOSED_CATEGORY_WITH_EXAMPLE_CASES = [
    pytest.param(
        "What is your employer's main activity, for example, "
        "providing finance, retail or social services?",
        id="closed_category_with_example_for_example_or_list",
    ),
    pytest.param(
        "What is your employer's main activity, such as hair cutting or teeth cleaning?",
        id="closed_category_with_example_such_as_or_list",
    ),
    pytest.param(
        "Are you a student or a worker, including a lecturer as a worker?",
        id="closed_category_with_example_including_and_or",
    ),
    pytest.param(
        "Do you see yourself as teenager or adult? I.e. 13-19 years old or 20+ years old.",
        id="closed_category_with_example_ie_follow_on_or_list",
    ),
]


NON_EXAMPLE_CASES = [
    pytest.param("I work in retail.", id="non_example_simple_statement"),
    pytest.param(
        "The organisation provides healthcare services.",
        id="non_example_healthcare_statement",
    ),
    pytest.param(
        "Customer service activities form part of the role.",
        id="non_example_customer_service_statement",
    ),
    pytest.param(
        "The company manufactures furniture.", id="non_example_manufacturing_statement"
    ),
    pytest.param(
        "Teaching apprentices is a key responsibility.",
        id="non_example_teaching_statement",
    ),
    pytest.param(
        "The organisation mainly supports local businesses.",
        id="non_example_local_business_statement",
    ),
    pytest.param("What would you like to do next?", id="non_example_open_question"),
    pytest.param(
        "The organisation's activities are mainly retail focused.",
        id="non_example_retail_focused_statement",
    ),
    pytest.param(
        "Customer service is included within the role description.",
        id="non_example_included_not_including",
    ),
]


NON_CLOSED_CATEGORY_CASES = [
    pytest.param(
        "What services does your organisation provide?",
        id="non_closed_open_services_question",
    ),
    pytest.param(
        "What products does your employer manufacture?",
        id="non_closed_open_products_question",
    ),
    pytest.param(
        "Please describe your main duties.", id="non_closed_describe_main_duties"
    ),
    pytest.param("What is your job title?", id="non_closed_job_title_question"),
    pytest.param(
        "What type of work do you do?", id="non_closed_open_type_of_work_question"
    ),
    pytest.param(
        "What products does your employer manufacture, for example bicycles?",
        id="non_closed_example_without_category_options",
    ),
    pytest.param(
        "What services do you provide, such as accounting?",
        id="non_closed_such_as_single_example",
    ),
    pytest.param(
        "Do you work in healthcare, like a nurse?",
        id="non_closed_like_example_not_supported",
    ),
    pytest.param(
        "Are you self-employed? I.e. you run your own business.",
        id="non_closed_definition_example_without_options",
    ),
]


ALL_EXAMPLE_CASES = [
    *EXPLICIT_EXAMPLE_MARKER_CASES,
    *INCLUDING_EXAMPLE_PHRASE_CASES,
    *DEFINITION_EXAMPLE_WORDING_CASES,
]


# ============================================================================
# Test Data - Shared between tests
# ============================================================================


@pytest.fixture
def example_and_leading_input_df():
    """Return input data for example and leading metric column tests."""
    return pd.DataFrame(
        {
            "follow_up_question": [
                "What products does your employer make, for example, furniture or toys?",
                "Are you a student or a worker?",
                "Please describe your main duties.",
                None,
            ],
            "respondent_id": [1, 2, 3, 4],
        }
    )


@pytest.fixture
def expected_example_and_leading_df():
    """Return expected output after adding example and leading metric columns."""
    return pd.DataFrame(
        {
            "follow_up_question": [
                "What products does your employer make, for example, furniture or toys?",
                "Are you a student or a worker?",
                "Please describe your main duties.",
                None,
            ],
            "respondent_id": [1, 2, 3, 4],
            "follow_up_question_has_examples": [True, False, False, False],
            "follow_up_question_has_closed_category_option": [
                True,
                True,
                False,
                False,
            ],
            "follow_up_question_has_closed_category_without_examples": [
                False,
                True,
                False,
                False,
            ],
        }
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


# ============================================================================
# Test get_example_and_leading_columns function
# ============================================================================

EXPECTED_FALSE_EXAMPLE_AND_LEADING_METRICS = {
    "has_examples": False,
    "has_closed_category_option": False,
    "has_closed_category_without_examples": False,
}


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param(
            "What products does your employer make, for example, furniture or toys?",
            {
                "has_examples": True,
                "has_closed_category_option": True,
                "has_closed_category_without_examples": False,
            },
            id="example_and_closed_category",
        ),
        pytest.param(
            "Are you a student or a worker?",
            {
                "has_examples": False,
                "has_closed_category_option": True,
                "has_closed_category_without_examples": True,
            },
            id="closed_category_without_example",
        ),
        pytest.param(
            "Are you a student or a worker, including a lecturer as a worker?",
            {
                "has_examples": True,
                "has_closed_category_option": True,
                "has_closed_category_without_examples": False,
            },
            id="closed_category_with_including_example",
        ),
        pytest.param(
            "Are you self-employed? I.e. you run your own business.",
            {
                "has_examples": True,
                "has_closed_category_option": False,
                "has_closed_category_without_examples": False,
            },
            id="example_without_closed_category",
        ),
        pytest.param(
            "What services do your organisation provide?",
            {
                "has_examples": False,
                "has_closed_category_option": False,
                "has_closed_category_without_examples": False,
            },
            id="open_question_without_example",
        ),
    ],
)
def test_get_example_and_leading_metrics_returns_expected_metrics(
    text,
    expected,
):
    """Returns expected example and leading question metrics."""
    assert get_example_and_leading_metrics(text) == expected, (
        f"Expected example and leading metrics for {text!r} " f"to be {expected}"
    )


@pytest.mark.parametrize("text", NON_EXAMPLE_CASES)
def test_get_example_and_leading_metrics_non_example_text(text):
    """Returns falsey example and leading metrics for non-example text."""
    assert (
        get_example_and_leading_metrics(text)
        == EXPECTED_FALSE_EXAMPLE_AND_LEADING_METRICS
    ), (
        f"Expected falsey example and leading metrics "
        f"for non-example text: {text!r}"
    )


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUTS)
def test_get_example_and_leading_metrics_empty_text(text):
    """Returns falsey example and leading metrics for empty text inputs."""
    assert (
        get_example_and_leading_metrics(text)
        == EXPECTED_FALSE_EXAMPLE_AND_LEADING_METRICS
    ), (
        f"Expected falsey example and leading metrics "
        f"for empty text input: {text!r}"
    )


@pytest.mark.parametrize("text", NON_STRING_INPUTS)
def test_get_example_and_leading_metrics_non_string(text):
    """Returns falsey example and leading metrics for non-string inputs."""
    assert (
        get_example_and_leading_metrics(text)
        == EXPECTED_FALSE_EXAMPLE_AND_LEADING_METRICS
    ), (
        f"Expected falsey example and leading metrics "
        f"for non-string input: {text!r} "
        f"(type: {type(text).__name__})"
    )


def test_get_example_and_leading_metrics_returns_expected_keys():
    """Returns all expected example and leading metric keys."""
    expected_keys = {
        "has_examples",
        "has_closed_category_option",
        "has_closed_category_without_examples",
    }

    assert (
        set(
            get_example_and_leading_metrics(
                "What products does your employer make, for example, furniture or toys?"
            )
        )
        == expected_keys
    ), "Expected get_example_and_leading_metrics to return all expected keys"


# ============================================================================
# Test add_example_and_leading_columns function
# ============================================================================


def test_add_example_and_leading_columns_returns_expected_dataframe(
    example_and_leading_input_df,
    expected_example_and_leading_df,
):
    """Adds expected example and leading metric columns and values."""
    result = add_example_and_leading_columns(
        example_and_leading_input_df,
        text_column="follow_up_question",
    )

    pd.testing.assert_frame_equal(result, expected_example_and_leading_df)


# ============================================================================
# Test summarise_example_and_leading_columns function
# ============================================================================


def test_summarise_example_and_leading_columns_returns_expected_summary(
    expected_example_and_leading_df,
):
    """Returns expected summary statistics from precomputed metric columns."""
    result = summarise_example_and_leading_columns(
        expected_example_and_leading_df,
        prefix="follow_up_question_",
    ).__dict__

    assert result["n_count"] == 4, "Expected n_count to equal the number of rows"

    assert result["pct_with_examples"] == pytest.approx(25, rel=1e-2), (
        "Expected pct_with_examples to equal the percentage of rows "
        "containing example wording"
    )

    assert result["pct_with_closed_category_option"] == pytest.approx(50, rel=1e-2), (
        "Expected pct_with_closed_category_option to equal the percentage "
        "of rows containing closed-category wording"
    )

    assert result["pct_with_closed_category_without_examples"] == pytest.approx(
        25, rel=1e-2
    ), (
        "Expected pct_with_closed_category_without_examples to equal the "
        "percentage of rows containing closed-category wording without examples"
    )


def test_summarise_example_and_leading_columns_uses_prefix():
    """Uses the supplied prefix to locate metric columns."""
    df = pd.DataFrame(
        {
            "test_has_examples": [True],
            "test_has_closed_category_option": [True],
            "test_has_closed_category_without_examples": [False],
        }
    )

    result = summarise_example_and_leading_columns(
        df,
        prefix="test_",
    ).__dict__

    assert result["n_count"] == 1, (
        "Expected summarise_example_and_leading_columns to use the supplied "
        "prefix when locating metric columns"
    )

    assert (
        result["pct_with_examples"] == 100.0
    ), "Expected pct_with_examples to be calculated from prefixed columns"

    assert result["pct_with_closed_category_option"] == 100.0, (
        "Expected pct_with_closed_category_option to be calculated from "
        "prefixed columns"
    )

    assert result["pct_with_closed_category_without_examples"] == 0.0, (
        "Expected pct_with_closed_category_without_examples to be calculated "
        "from prefixed columns"
    )


def test_summarise_example_and_leading_columns_missing_column_raises_key_error():
    """Raises KeyError when a required metric column is missing."""
    df = pd.DataFrame(
        {
            "question_has_examples": [True],
        }
    )

    with pytest.raises(
        KeyError,
        match="question_has_closed_category_option",
    ):
        summarise_example_and_leading_columns(
            df,
            prefix="question_",
        )


def test_summarise_example_and_leading_columns_returns_metrics_model(
    expected_example_and_leading_df,
):
    """Returns an ExampleLeadingQuestionMetrics model."""
    result = summarise_example_and_leading_columns(
        expected_example_and_leading_df,
        prefix="follow_up_question_",
    )

    assert isinstance(result, ExampleLeadingQuestionMetrics), (
        "Expected summarise_example_and_leading_columns to return an "
        "ExampleLeadingQuestionMetrics instance"
    )


# ============================================================================
# Test ExampleLeadingQuestionMetrics function
# ============================================================================


def test_example_leading_question_metrics_stores_values():
    """Stores the supplied example and leading question metric values."""
    metrics = ExampleLeadingQuestionMetrics(
        n_count=4,
        pct_with_examples=50.0,
        pct_with_closed_category_option=75.0,
        pct_with_closed_category_without_examples=25.0,
    )

    assert metrics.n_count == 4, "Expected n_count to be stored"
    assert metrics.pct_with_examples == 50.0, "Expected pct_with_examples to be stored"
    assert (
        metrics.pct_with_closed_category_option == 75.0
    ), "Expected pct_with_closed_category_option to be stored"
    assert (
        metrics.pct_with_closed_category_without_examples == 25.0
    ), "Expected pct_with_closed_category_without_examples to be stored"


def test_example_leading_question_metrics_report_metrics_returns_expected_text():
    """Returns formatted example and leading question metrics as text."""
    metrics = ExampleLeadingQuestionMetrics(
        n_count=4,
        pct_with_examples=50.0,
        pct_with_closed_category_option=75.0,
        pct_with_closed_category_without_examples=25.0,
    )

    result = metrics.report_metrics()

    expected = "\n".join(
        [
            "\nExample and leading question metrics:",
            " Number of follow-up questions: 4",
            " Percentage with examples: 50.00%",
            " Percentage with closed category options: 75.00%",
            " Percentage with closed category options without examples: 25.00%",
        ]
    )

    assert (
        result == expected
    ), "Expected report_metrics to return correctly formatted metric text"


# ============================================================================
# Test compute_example_and_leading_metrics function
# ============================================================================


def test_compute_example_and_leading_metrics_returns_metrics_model(
    example_and_leading_input_df,
):
    """Returns an ExampleLeadingQuestionMetrics model."""
    result = compute_example_and_leading_metrics(
        example_and_leading_input_df,
        text_column="follow_up_question",
    )

    assert isinstance(result, ExampleLeadingQuestionMetrics), (
        "Expected compute_example_and_leading_metrics to return an "
        "ExampleLeadingQuestionMetrics instance"
    )


def test_compute_example_and_leading_metrics_returns_expected_values(
    example_and_leading_input_df,
):
    """Returns expected example and leading question summary values."""
    result = compute_example_and_leading_metrics(
        example_and_leading_input_df,
        text_column="follow_up_question",
    )

    assert (
        result.n_count == 4
    ), "Expected n_count to equal the number of rows in the input DataFrame"

    assert result.pct_with_examples == pytest.approx(25, rel=1e-2), (
        "Expected pct_with_examples to equal the percentage of rows "
        "containing example wording"
    )

    assert result.pct_with_closed_category_option == pytest.approx(50, rel=1e-2), (
        "Expected pct_with_closed_category_option to equal the percentage "
        "of rows containing closed-category wording"
    )

    assert result.pct_with_closed_category_without_examples == pytest.approx(
        25, rel=1e-2
    ), (
        "Expected pct_with_closed_category_without_examples to equal the "
        "percentage of rows containing closed-category wording without examples"
    )


def test_compute_example_and_leading_metrics_returns_zero_percentages():
    """Returns zero percentages when no rows match any metrics."""
    df = pd.DataFrame(
        {
            "follow_up_question": [
                "Please describe your main duties.",
                "What is your job title?",
            ]
        }
    )

    result = compute_example_and_leading_metrics(
        df,
        text_column="follow_up_question",
    )

    assert result.n_count == 2
    assert result.pct_with_examples == 0.0
    assert result.pct_with_closed_category_option == 0.0
    assert result.pct_with_closed_category_without_examples == 0.0


def test_compute_example_and_leading_metrics_uses_default_prefix(
    example_and_leading_input_df,
):
    """Uses the internal evaluation prefix when computing metrics."""
    result = compute_example_and_leading_metrics(
        example_and_leading_input_df,
        text_column="follow_up_question",
    )

    assert isinstance(result, ExampleLeadingQuestionMetrics), (
        "Expected compute_example_and_leading_metrics to return metrics when "
        "using the default evaluation prefix"
    )

    assert (
        result.n_count == 4
    ), "Expected n_count to equal the number of rows when using the default prefix"


def test_compute_example_and_leading_metrics_missing_text_column_raises_key_error(
    example_and_leading_input_df,
):
    """Raises KeyError when the text column is missing."""
    with pytest.raises(KeyError, match="missing_column"):
        compute_example_and_leading_metrics(
            example_and_leading_input_df,
            text_column="missing_column",
        )
