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


def unique_texts(*groups: tuple[tuple[str, str], ...]) -> list[str]:
    """Return unique texts while preserving their original order."""
    return list(dict.fromkeys(text for group in groups for text, _ in group))


def as_pytest_params(cases: tuple[tuple[object, str], ...]) -> tuple:
    """Convert (value, id) tuples into pytest.param entries."""
    return tuple(pytest.param(value, id=case_id) for value, case_id in cases)


NON_STRING_INPUTS = (
    (None, "none_input"),
    (123, "integer_input"),
    (12.5, "float_input"),
    ([], "list_input"),
    ({}, "dict_input"),
    (True, "bool_input"),
)
NON_STRING_INPUT_PARAMS = as_pytest_params(NON_STRING_INPUTS)

EMPTY_TEXT_INPUTS = (
    ("", "empty_string"),
    ("   ", "whitespace_only"),
)
EMPTY_TEXT_INPUT_PARAMS = as_pytest_params(EMPTY_TEXT_INPUTS)

EXPLICIT_EXAMPLE_MARKER_CASES = (
    (
        "What is your employer's main activity, for example, "
        "providing finance, retail or social services?",
        "explicit_for_example_comma_list",
    ),
    (
        "What products does your company make, for example, furniture or toys?",
        "explicit_for_example_or_list",
    ),
    (
        "What services does your organisation provide, for example, teaching or training?",
        "explicit_for_example_activity_examples",
    ),
    (
        "Do you consider your job to be in retail? E.g. selling goods in a shop.",
        "explicit_eg_follow_on_sentence",
    ),
    (
        "Do you work in manufacturing? E.g. making furniture or clothing.",
        "explicit_eg_follow_on_or_list",
    ),
    (
        "Are you self-employed? I.e. you run your own business.",
        "explicit_ie_follow_on_definition",
    ),
    (
        "Do you work full time? I.e. 35 or more hours per week.",
        "explicit_ie_follow_on_hours_definition",
    ),
    (
        "What is your employer's main activity, such as hair cutting or teeth cleaning?",
        "explicit_such_as_or_list",
    ),
    (
        "What services do you provide, such as accounting or bookkeeping?",
        "explicit_such_as_service_examples",
    ),
    (
        "What products does your employer manufacture (e.g. bicycles)?",
        "explicit_parenthetical_eg",
    ),
    (
        "What service does your organisation provide (for example, hairdressing)?",
        "explicit_parenthetical_for_example",
    ),
    (
        "What type of work do you do (such as bookkeeping)?",
        "explicit_parenthetical_such_as",
    ),
    (
        "What industry do you work in (e.g. retail)?",
        "explicit_parenthetical_eg_single_example",
    ),
    (
        "What products do you make (such as furniture)?",
        "explicit_parenthetical_such_as_single_example",
    ),
)
EXPLICIT_EXAMPLE_MARKER_PARAMS = as_pytest_params(EXPLICIT_EXAMPLE_MARKER_CASES)

INCLUDING_EXAMPLE_PHRASE_CASES = (
    (
        "Are you a student or a worker, including a lecturer as a worker?",
        "including_example_phrase_with_or",
    ),
    (
        "Do you work in healthcare, including nursing roles?",
        "including_example_phrase_single_role",
    ),
    (
        "Does your role involve administration, including scheduling meetings?",
        "including_example_phrase_activity",
    ),
    (
        "Do you work in manufacturing, for instance, producing furniture?",
        "for_instance_example_phrase_activity",
    ),
    (
        "What services do you provide, for instance, accounting or payroll support?",
        "for_instance_example_phrase_or_list",
    ),
    (
        "What products does your employer make, for instance, bicycles or clothing?",
        "for_instance_example_phrase_product_list",
    ),
    (
        "What services do you provide (for instance, accounting)?",
        "parenthetical_for_instance_example_phrase",
    ),
)
INCLUDING_EXAMPLE_PHRASE_PARAMS = as_pytest_params(INCLUDING_EXAMPLE_PHRASE_CASES)

DEFINITION_EXAMPLE_WORDING_CASES = (
    (
        "Do you work in retail, meaning you sell goods directly to customers?",
        "definition_meaning_retail",
    ),
    (
        "Are you self-employed, meaning you run your own business?",
        "definition_meaning_self_employed",
    ),
    (
        "Do you work in education, meaning you teach or train people?",
        "definition_meaning_education",
    ),
    (
        "Are you self-employed, which means you run your own business?",
        "definition_which_means_self_employed",
    ),
    (
        "Do you work in retail, which means selling goods to customers?",
        "definition_which_means_retail",
    ),
    (
        "Do you work in healthcare, namely nursing or physiotherapy?",
        "definition_namely_healthcare_or_list",
    ),
    (
        "Do you provide professional services, namely accounting or legal advice?",
        "definition_namely_professional_services",
    ),
    (
        "Do you work in education, that is teaching or training?",
        "definition_that_is_education",
    ),
    (
        "Do you work in healthcare, that is nursing or physiotherapy?",
        "definition_that_is_healthcare",
    ),
)
DEFINITION_EXAMPLE_WORDING_PARAMS = as_pytest_params(DEFINITION_EXAMPLE_WORDING_CASES)

CLOSED_CATEGORY_OPTION_CASES = (
    (
        "What is your employer's main activity: teaching or research?",
        "closed_category_colon_with_or",
    ),
    ("Are you a student or a worker?", "closed_category_simple_or"),
    (
        "Is your organisation mainly public or private?",
        "closed_category_public_or_private",
    ),
    (
        "Are you employed in retail or manufacturing?",
        "closed_category_sector_or",
    ),
    (
        "Do you mainly provide products or services?",
        "closed_category_products_or_services",
    ),
    ("Are you a manager or a supervisor?", "closed_category_role_or"),
    (
        "Do you work in the public or private sector?",
        "closed_category_sector_phrase",
    ),
    (
        "Are you involved in teaching or administration?",
        "closed_category_activity_or",
    ),
    (
        "What type of organisation do you work for: school, hospital or university?",
        "closed_category_colon_comma_or_list",
    ),
    (
        "What kind of teacher are you: primary, secondary or college?",
        "closed_category_colon_teacher_options",
    ),
    (
        "Are you employed either full-time or part-time?",
        "closed_category_either_or",
    ),
    ("Are you a manager/supervisor?", "closed_category_slash_options"),
    (
        "Which of the following best describes your role?",
        "closed_category_which_of_the_following",
    ),
    (
        "Which of these sectors do you work in?",
        "closed_category_which_of_these",
    ),
    (
        "Select one of the following options.",
        "closed_category_select_one",
    ),
    (
        "Select the one that best describes your role.",
        "closed_category_select_the_one",
    ),
    (
        "Choose one of the following categories.",
        "closed_category_choose_one",
    ),
    (
        "Choose the one that best describes your organisation.",
        "closed_category_choose_the_one",
    ),
    ("Pick one of the following options.", "closed_category_pick_one"),
    (
        "Pick the one that best matches your role.",
        "closed_category_pick_the_one",
    ),
)
CLOSED_CATEGORY_OPTION_PARAMS = as_pytest_params(CLOSED_CATEGORY_OPTION_CASES)

CLOSED_CATEGORY_WITH_EXAMPLE_CASES = (
    (
        "What is your employer's main activity, for example, "
        "providing finance, retail or social services?",
        "closed_category_with_example_for_example_or_list",
    ),
    (
        "What is your employer's main activity, such as hair cutting or teeth cleaning?",
        "closed_category_with_example_such_as_or_list",
    ),
    (
        "Are you a student or a worker, including a lecturer as a worker?",
        "closed_category_with_example_including_and_or",
    ),
    (
        "Do you see yourself as teenager or adult? I.e. 13-19 years old or 20+ years old.",
        "closed_category_with_example_ie_follow_on_or_list",
    ),
)

NON_EXAMPLE_CASES = (
    ("I work in retail.", "non_example_simple_statement"),
    (
        "The organisation provides healthcare services.",
        "non_example_healthcare_statement",
    ),
    (
        "Customer service activities form part of the role.",
        "non_example_customer_service_statement",
    ),
    (
        "The company manufactures furniture.",
        "non_example_manufacturing_statement",
    ),
    (
        "Teaching apprentices is a key responsibility.",
        "non_example_teaching_statement",
    ),
    (
        "The organisation mainly supports local businesses.",
        "non_example_local_business_statement",
    ),
    ("What would you like to do next?", "non_example_open_question"),
    (
        "The organisation's activities are mainly retail focused.",
        "non_example_retail_focused_statement",
    ),
    (
        "Customer service is included within the role description.",
        "non_example_included_not_including",
    ),
)
NON_EXAMPLE_PARAMS = as_pytest_params(NON_EXAMPLE_CASES)

NON_CLOSED_CATEGORY_CASES = (
    (
        "What services does your organisation provide?",
        "non_closed_open_services_question",
    ),
    (
        "What products does your employer manufacture?",
        "non_closed_open_products_question",
    ),
    (
        "Please describe your main duties.",
        "non_closed_describe_main_duties",
    ),
    ("What is your job title?", "non_closed_job_title_question"),
    (
        "What type of work do you do?",
        "non_closed_open_type_of_work_question",
    ),
    (
        "What products does your employer manufacture, for example bicycles?",
        "non_closed_example_without_category_options",
    ),
    (
        "What services do you provide, such as accounting?",
        "non_closed_such_as_single_example",
    ),
    (
        "Do you work in healthcare, like a nurse?",
        "non_closed_like_example_not_supported",
    ),
    (
        "Are you self-employed? I.e. you run your own business.",
        "non_closed_definition_example_without_options",
    ),
)
NON_CLOSED_CATEGORY_PARAMS = as_pytest_params(NON_CLOSED_CATEGORY_CASES)

ALL_EXAMPLE_CASES = (
    *EXPLICIT_EXAMPLE_MARKER_CASES,
    *INCLUDING_EXAMPLE_PHRASE_CASES,
    *DEFINITION_EXAMPLE_WORDING_CASES,
)
ALL_EXAMPLE_PARAMS = as_pytest_params(ALL_EXAMPLE_CASES)

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


@pytest.mark.parametrize("text", EXPLICIT_EXAMPLE_MARKER_PARAMS)
def test_has_explicit_example_marker_detects_explicit_examples(text: str):
    """Detect explicit example wording such as e.g. and for example."""
    assert has_explicit_example_marker(
        text
    ), f"Expected explicit example marker to be detected for: {text}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUT_PARAMS)
def test_has_explicit_example_marker_empty_text(text):
    """Returns False for empty text inputs."""
    assert (
        has_explicit_example_marker(text) is False
    ), f"Expected False for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUT_PARAMS)
def test_has_explicit_example_marker_non_string_inputs(text):
    """Returns False for non-string inputs."""
    assert (
        has_explicit_example_marker(text) is False
    ), f"Expected False for non-string input: {text!r}"


# ============================================================================
# Test has_including_example_phrase function
# ============================================================================


@pytest.mark.parametrize("text", INCLUDING_EXAMPLE_PHRASE_PARAMS)
def test_has_including_example_phrase_detects_including_style_examples(text: str):
    """Detect including-style example wording."""
    assert has_including_example_phrase(
        text
    ), f"Expected including-style example wording to be detected for: {text}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUT_PARAMS)
def test_has_including_example_phrase_empty_text(text):
    """Returns False for empty text inputs."""
    assert (
        has_including_example_phrase(text) is False
    ), f"Expected False for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUT_PARAMS)
def test_has_including_example_phrase_non_string_inputs(text):
    """Returns False for non-string inputs."""
    assert (
        has_including_example_phrase(text) is False
    ), f"Expected False for non-string input: {text!r}"


# ============================================================================
# Test has_definition_example_wording function
# ============================================================================


@pytest.mark.parametrize("text", DEFINITION_EXAMPLE_WORDING_PARAMS)
def test_has_definition_example_wording_detects_definition_style_examples(
    text: str,
):
    """Detect definition-style example wording."""
    assert has_definition_example_wording(
        text
    ), f"Expected definition-style example wording to be detected for: {text}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUT_PARAMS)
def test_has_definition_example_wording_empty_text(text):
    """Returns False for empty text inputs."""
    assert (
        has_definition_example_wording(text) is False
    ), f"Expected False for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUT_PARAMS)
def test_has_definition_example_wording_non_string_inputs(text):
    """Returns False for non-string inputs."""
    assert (
        has_definition_example_wording(text) is False
    ), f"Expected False for non-string input: {text!r}"


# ============================================================================
# Test has_examples function
# ============================================================================


@pytest.mark.parametrize("text", ALL_EXAMPLE_PARAMS)
def test_has_examples_detects_supported_example_patterns(text: str):
    """Detect example wording across all supported example patterns."""
    assert has_examples(text), f"Expected example wording to be detected for: {text}"


@pytest.mark.parametrize("text", NON_EXAMPLE_PARAMS)
def test_has_examples_ignores_non_example_wording(text: str):
    """Avoid flagging ordinary statements and similar wording as examples."""
    assert not has_examples(text), f"Did not expect example wording in: {text}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUT_PARAMS)
def test_has_examples_empty_text(text):
    """Returns False for empty text inputs."""
    assert has_examples(text) is False, f"Expected False for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUT_PARAMS)
def test_has_examples_non_string_inputs(text):
    """Returns False for non-string inputs."""
    assert has_examples(text) is False, f"Expected False for non-string input: {text!r}"


# ============================================================================
# Test has_closed_category_options function
# ============================================================================


@pytest.mark.parametrize("text", CLOSED_CATEGORY_OPTION_PARAMS)
def test_has_closed_category_options_detects_predefined_categories(text: str):
    """Detect closed-category response options in question wording."""
    assert has_closed_category_options(
        text
    ), f"Expected closed-category wording to be detected for: {text}"


@pytest.mark.parametrize("text", NON_CLOSED_CATEGORY_PARAMS)
def test_has_closed_category_options_ignores_open_questions(text: str):
    """Avoid flagging open questions that do not provide response options."""
    assert not has_closed_category_options(
        text
    ), f"Did not expect closed-category wording to be detected for: {text}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUT_PARAMS)
def test_has_closed_category_options_empty_text(text):
    """Returns False for empty text inputs."""
    assert (
        has_closed_category_options(text) is False
    ), f"Expected False for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUT_PARAMS)
def test_has_closed_category_options_non_string_inputs(text):
    """Returns False for non-string inputs."""
    assert (
        has_closed_category_options(text) is False
    ), f"Expected False for non-string input: {text!r}"


# ============================================================================
# Test has_closed_category_without_examples function
# ============================================================================


@pytest.mark.parametrize("text", CLOSED_CATEGORY_OPTION_PARAMS)
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


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUT_PARAMS)
def test_has_closed_category_without_examples_empty_text(text):
    """Returns False for empty text inputs."""
    assert (
        has_closed_category_without_examples(text) is False
    ), f"Expected False for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUT_PARAMS)
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


@pytest.mark.parametrize("text", NON_EXAMPLE_PARAMS)
def test_get_example_and_leading_metrics_non_example_text(text):
    """Returns falsey example and leading metrics for non-example text."""
    assert (
        get_example_and_leading_metrics(text)
        == EXPECTED_FALSE_EXAMPLE_AND_LEADING_METRICS
    ), (
        f"Expected falsey example and leading metrics "
        f"for non-example text: {text!r}"
    )


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUT_PARAMS)
def test_get_example_and_leading_metrics_empty_text(text):
    """Returns falsey example and leading metrics for empty text inputs."""
    assert (
        get_example_and_leading_metrics(text)
        == EXPECTED_FALSE_EXAMPLE_AND_LEADING_METRICS
    ), (
        f"Expected falsey example and leading metrics "
        f"for empty text input: {text!r}"
    )


@pytest.mark.parametrize("text", NON_STRING_INPUT_PARAMS)
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
