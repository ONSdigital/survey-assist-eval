"""Tests for question structure wrapper functions."""

# pylint: disable=redefined-outer-name
# pylint: disable=duplicate-code

import numpy as np
import pandas as pd
import pytest

from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    QuestionStructureMetrics,
    add_question_structure_columns,
    compute_question_structure_metrics,
    summarise_question_structure_columns,
)

# ============================================================================
# Test Data - Shared between tests
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


QUESTION_STRUCTURE_METRIC_COLUMNS = {
    "has_question_mark",
    "interrogative_start",
    "instruction_prompt_start",
    "interrogative_wh_count",
    "instruction_prompt_count",
    "is_question",
    "contains_multiple_asks",
    "is_single_question",
}


@pytest.fixture
def question_structure_input_df():
    """Return input data for question structure column tests."""
    return pd.DataFrame(
        {
            "follow_up_question": [
                "What is your name?",
                "This is a statement.",
                None,
            ],
            "respondent_id": [1, 2, 3],
        }
    )


@pytest.fixture
def expected_default_question_structure_columns():
    """Return expected question structure columns with default prefix."""
    return {
        f"follow_up_question_{column}" for column in QUESTION_STRUCTURE_METRIC_COLUMNS
    }


@pytest.fixture
def expected_custom_question_structure_columns():
    """Return expected question structure columns with custom prefix."""
    return {f"question_{column}" for column in QUESTION_STRUCTURE_METRIC_COLUMNS}


@pytest.fixture
def expected_question_structure_df():
    """Return expected output after adding question structure columns."""
    return pd.DataFrame(
        {
            "follow_up_question": [
                "What is your name?",
                "This is a statement.",
                None,
            ],
            "respondent_id": [1, 2, 3],
            "follow_up_question_has_question_mark": [True, False, False],
            "follow_up_question_interrogative_start": [True, False, False],
            "follow_up_question_instruction_prompt_start": [False, False, False],
            "follow_up_question_interrogative_wh_count": [1, 0, 0],
            "follow_up_question_instruction_prompt_count": [0, 0, 0],
            "follow_up_question_is_question": [True, False, False],
            "follow_up_question_contains_multiple_asks": [False, False, False],
            "follow_up_question_is_single_question": [True, False, False],
        }
    )


# ============================================================================
# Test add_question_structure_columns function
# ============================================================================


def test_add_question_structure_columns_returns_expected_dataframe(
    question_structure_input_df,
    expected_question_structure_df,
):
    """Adds expected question structure metric columns and values."""
    result = add_question_structure_columns(
        question_structure_input_df,
        text_column="follow_up_question",
    )

    pd.testing.assert_frame_equal(result, expected_question_structure_df)


def test_add_question_structure_columns_adds_default_prefixed_columns(
    question_structure_input_df,
    expected_default_question_structure_columns,
):
    """Adds question structure columns using the text column name as the default prefix."""
    result = add_question_structure_columns(
        question_structure_input_df,
        text_column="follow_up_question",
    )

    assert expected_default_question_structure_columns.issubset(result.columns), (
        "Expected add_question_structure_columns to add all question structure "
        "metric columns with the default text column prefix"
    )


def test_add_question_structure_columns_preserves_original_columns(
    question_structure_input_df,
):
    """Preserves original columns when adding question structure columns."""
    result = add_question_structure_columns(
        question_structure_input_df,
        text_column="follow_up_question",
    )

    original_columns = set(question_structure_input_df.columns)

    assert original_columns.issubset(
        result.columns
    ), "Expected add_question_structure_columns to preserve all original columns"


def test_add_question_structure_columns_uses_custom_prefix(
    question_structure_input_df,
    expected_custom_question_structure_columns,
):
    """Adds question structure columns using a custom prefix."""
    result = add_question_structure_columns(
        question_structure_input_df,
        text_column="follow_up_question",
        prefix="question_",
    )

    assert expected_custom_question_structure_columns.issubset(result.columns), (
        "Expected add_question_structure_columns to add all question structure "
        "metric columns with the custom prefix"
    )


def test_add_question_structure_columns_does_not_add_default_prefix_when_custom_prefix_used(
    question_structure_input_df,
    expected_default_question_structure_columns,
):
    """Does not add default-prefixed columns when a custom prefix is supplied."""
    result = add_question_structure_columns(
        question_structure_input_df,
        text_column="follow_up_question",
        prefix="question_",
    )

    assert expected_default_question_structure_columns.isdisjoint(result.columns), (
        "Expected default-prefixed question structure columns not to be added "
        "when a custom prefix is supplied"
    )


def test_add_question_structure_columns_handles_missing_values_as_empty_text(
    question_structure_input_df,
):
    """Treats missing text values as empty strings when creating metrics."""
    result = add_question_structure_columns(
        question_structure_input_df,
        text_column="follow_up_question",
    )

    expected_values = {
        f"follow_up_question_{metric}": value
        for metric, value in EXPECTED_FALSE_QUESTION_STRUCTURE_METRICS.items()
    }

    for column, expected_value in expected_values.items():
        assert (
            result.loc[2, column] == expected_value
        ), f"Expected {column} to be {expected_value} for missing text input"


def test_add_question_structure_columns_raises_key_error_for_missing_text_column():
    """Raises KeyError when the text column is not present in the DataFrame."""
    df = pd.DataFrame({"other_column": ["What is your name?"]})

    with pytest.raises(KeyError):
        add_question_structure_columns(
            df,
            text_column="follow_up_question",
        )


# ============================================================================
# Test summarise_question_structure_columns function
# ============================================================================


def test_summarise_question_structure_columns_returns_expected_summary(
    expected_question_structure_df,
):
    """Returns expected summary statistics from precomputed metric columns."""
    result = summarise_question_structure_columns(
        expected_question_structure_df,
        prefix="follow_up_question_",
    )

    assert result["n_count"] == 3, "Expected n_count to equal the number of rows"
    assert result["pct_is_question"] == pytest.approx(
        33.33, rel=1e-2
    ), "Expected pct_is_question to equal the percentage of questions"
    assert result["pct_is_single_question"] == pytest.approx(
        33.33, rel=1e-2
    ), "Expected pct_is_single_question to equal the percentage of single questions"
    assert result["pct_with_instruction_prompt_start"] == 0.0, (
        "Expected pct_with_instruction_prompt_start to equal the percentage of "
        "questions starting with an instruction prompt"
    )
    assert result["pct_with_interrogative_start"] == pytest.approx(33.33, rel=1e-2), (
        "Expected pct_with_interrogative_start to equal the percentage of "
        "questions starting with an interrogative"
    )
    assert result["pct_has_question_mark"] == pytest.approx(33.33, rel=1e-2), (
        "Expected pct_has_question_mark to equal the percentage of questions "
        "containing a question mark"
    )
    assert np.isnan(result["mean_instruction_prompt_count_excluding_zero"]), (
        "Expected mean_instruction_prompt_count_excluding_zero to be NaN when "
        "there are no non-zero counts"
    )
    assert result["mean_interrogative_wh_count_excluding_zero"] == 1.0, (
        "Expected mean_interrogative_wh_count_excluding_zero to equal the mean "
        "interrogative count excluding zero values"
    )


def test_summarise_question_structure_columns_uses_prefix():
    """Uses the supplied prefix to locate metric columns."""
    df = pd.DataFrame(
        {
            "test_is_question": [True],
            "test_is_single_question": [True],
            "test_instruction_prompt_start": [False],
            "test_instruction_prompt_count": [1],
            "test_interrogative_start": [False],
            "test_interrogative_wh_count": [1],
            "test_has_question_mark": [True],
        }
    )

    result = summarise_question_structure_columns(
        df,
        prefix="test_",
    )

    assert result["n_count"] == 1, (
        "Expected summarise_question_structure_columns to use the supplied "
        "prefix when locating metric columns"
    )
    assert (
        result["pct_is_question"] == 100.0
    ), "Expected pct_is_question to be calculated from prefixed columns"


def test_summarise_question_structure_columns_returns_nan_when_no_non_zero_counts():
    """Returns NaN for mean counts when all counts are zero."""
    df = pd.DataFrame(
        {
            "question_is_question": [True],
            "question_is_single_question": [True],
            "question_instruction_prompt_start": [False],
            "question_instruction_prompt_count": [0],
            "question_interrogative_start": [False],
            "question_interrogative_wh_count": [0],
            "question_has_question_mark": [True],
        }
    )

    result = summarise_question_structure_columns(
        df,
        prefix="question_",
    )

    assert np.isnan(result["mean_instruction_prompt_count_excluding_zero"]), (
        "Expected mean_instruction_prompt_count_excluding_zero to be NaN when "
        "all instruction prompt counts are zero"
    )
    assert np.isnan(result["mean_interrogative_wh_count_excluding_zero"]), (
        "Expected mean_interrogative_wh_count_excluding_zero to be NaN when "
        "all interrogative counts are zero"
    )


def test_summarise_question_structure_columns_missing_column_raises_key_error():
    """Raises KeyError when a required metric column is missing."""
    df = pd.DataFrame(
        {
            "question_is_question": [True],
        }
    )

    with pytest.raises(
        KeyError,
        match="question_is_single_question",
    ):
        summarise_question_structure_columns(
            df,
            prefix="question_",
        )


# ============================================================================
# Test QuestionStructureMetrics function
# ============================================================================


def test_question_structure_metrics_stores_values():
    """Stores the supplied question structure metric values."""
    metrics = QuestionStructureMetrics(
        n_count=3,
        pct_is_question=66.67,
        pct_is_single_question=33.33,
        pct_with_instruction_prompt_start=10.0,
        mean_instruction_prompt_count_excluding_zero=1.5,
        pct_with_interrogative_start=50.0,
        mean_interrogative_wh_count_excluding_zero=2.0,
        pct_has_question_mark=75.0,
    )

    assert metrics.n_count == 3, "Expected n_count to be stored"
    assert metrics.pct_is_question == 66.67, "Expected pct_is_question to be stored"
    assert (
        metrics.pct_is_single_question == 33.33
    ), "Expected pct_is_single_question to be stored"
    assert (
        metrics.pct_with_instruction_prompt_start == 10.0
    ), "Expected pct_with_instruction_prompt_start to be stored"
    assert (
        metrics.mean_instruction_prompt_count_excluding_zero == 1.5
    ), "Expected mean_instruction_prompt_count_excluding_zero to be stored"
    assert (
        metrics.pct_with_interrogative_start == 50.0
    ), "Expected pct_with_interrogative_start to be stored"
    assert (
        metrics.mean_interrogative_wh_count_excluding_zero == 2.0
    ), "Expected mean_interrogative_wh_count_excluding_zero to be stored"
    assert (
        metrics.pct_has_question_mark == 75.0
    ), "Expected pct_has_question_mark to be stored"


def test_question_structure_metrics_report_metrics_returns_expected_text():
    """Returns formatted question structure metrics as text."""
    metrics = QuestionStructureMetrics(
        n_count=3,
        pct_is_question=66.6667,
        pct_is_single_question=33.3333,
        pct_with_instruction_prompt_start=10.0,
        mean_instruction_prompt_count_excluding_zero=1.5,
        pct_with_interrogative_start=50.0,
        mean_interrogative_wh_count_excluding_zero=2.0,
        pct_has_question_mark=75.0,
    )

    result = metrics.report_metrics()

    expected = "\n".join(
        [
            "\nQuestion structure metrics:",
            " Number of follow-up questions: 3",
            " Percentage is_question: 66.67%",
            " Percentage is_single_question: 33.33%",
            " Percentage with instruction_prompt_start: 10.00%",
            " Mean instruction prompt count (excluding zero): 1.50",
            " Percentage with interrogative_start: 50.00%",
            " Mean interrogative WH count (excluding zero): 2.00",
            " Percentage with question mark: 75.00%",
        ]
    )

    assert (
        result == expected
    ), "Expected report_metrics to return correctly formatted metric text"


def test_question_structure_metrics_report_metrics_returns_na_for_none_means():
    """Returns N/A for mean metrics when values are None."""
    metrics = QuestionStructureMetrics(
        n_count=1,
        pct_is_question=100.0,
        pct_is_single_question=100.0,
        pct_with_instruction_prompt_start=0.0,
        mean_instruction_prompt_count_excluding_zero=None,
        pct_with_interrogative_start=0.0,
        mean_interrogative_wh_count_excluding_zero=None,
        pct_has_question_mark=100.0,
    )

    result = metrics.report_metrics()

    assert (
        " Mean instruction prompt count (excluding zero): N/A" in result
    ), "Expected instruction prompt mean to be shown as N/A when None"
    assert (
        " Mean interrogative WH count (excluding zero): N/A" in result
    ), "Expected interrogative WH mean to be shown as N/A when None"


# ============================================================================
# Test compute_question_structure_metrics function
# ============================================================================


def test_compute_question_structure_metrics_returns_metrics_model(
    question_structure_input_df,
):
    """Returns a QuestionStructureMetrics model."""
    result = compute_question_structure_metrics(
        question_structure_input_df,
        text_column="follow_up_question",
        prefix="follow_up_question_",
    )

    assert isinstance(result, QuestionStructureMetrics), (
        "Expected compute_question_structure_metrics to return a "
        "QuestionStructureMetrics instance"
    )


def test_compute_question_structure_metrics_returns_expected_values(
    question_structure_input_df,
):
    """Returns expected question structure summary values."""
    result = compute_question_structure_metrics(
        question_structure_input_df,
        text_column="follow_up_question",
        prefix="follow_up_question_",
    )

    assert (
        result.n_count == 3
    ), "Expected n_count to equal the number of rows in the input DataFrame"
    assert result.pct_is_question == pytest.approx(
        33.33, rel=1e-2
    ), "Expected pct_is_question to equal the percentage of questions"
    assert result.pct_is_single_question == pytest.approx(
        33.33, rel=1e-2
    ), "Expected pct_is_single_question to equal the percentage of single questions"
    assert result.pct_with_instruction_prompt_start == 0.0, (
        "Expected pct_with_instruction_prompt_start to equal the percentage of "
        "questions starting with an instruction prompt"
    )
    assert np.isnan(result.mean_instruction_prompt_count_excluding_zero), (
        "Expected mean_instruction_prompt_count_excluding_zero to be NaN when "
        "there are no non-zero instruction prompt counts"
    )
    assert result.pct_with_interrogative_start == pytest.approx(33.33, rel=1e-2), (
        "Expected pct_with_interrogative_start to equal the percentage of "
        "questions starting with an interrogative"
    )
    assert result.mean_interrogative_wh_count_excluding_zero == 1.0, (
        "Expected mean_interrogative_wh_count_excluding_zero to equal the mean "
        "interrogative count excluding zero values"
    )
    assert result.pct_has_question_mark == pytest.approx(33.33, rel=1e-2), (
        "Expected pct_has_question_mark to equal the percentage of questions "
        "containing a question mark"
    )


def test_compute_question_structure_metrics_uses_default_prefix(
    question_structure_input_df,
):
    """Uses the default prefix when no prefix is supplied."""
    result = compute_question_structure_metrics(
        question_structure_input_df,
        text_column="follow_up_question",
    )

    assert isinstance(result, QuestionStructureMetrics), (
        "Expected compute_question_structure_metrics to return metrics when using "
        "the default prefix"
    )
    assert (
        result.n_count == 3
    ), "Expected n_count to equal the number of rows when using the default prefix"


def test_compute_question_structure_metrics_uses_custom_prefix(
    question_structure_input_df,
):
    """Uses the supplied prefix when computing metrics."""
    result = compute_question_structure_metrics(
        question_structure_input_df,
        text_column="follow_up_question",
        prefix="question_",
    )

    assert isinstance(result, QuestionStructureMetrics), (
        "Expected compute_question_structure_metrics to return metrics when using "
        "a custom prefix"
    )
    assert (
        result.n_count == 3
    ), "Expected n_count to equal the number of rows when using a custom prefix"


def test_compute_question_structure_metrics_missing_text_column_raises_key_error(
    question_structure_input_df,
):
    """Raises KeyError when the text column is missing."""
    with pytest.raises(KeyError, match="missing_column"):
        compute_question_structure_metrics(
            question_structure_input_df,
            text_column="missing_column",
            prefix="follow_up_question_",
        )
