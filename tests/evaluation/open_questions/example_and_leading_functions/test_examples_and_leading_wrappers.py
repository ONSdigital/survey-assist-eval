"""Tests for example and leading question helper functions."""

# pylint: disable=redefined-outer-name
# pylint: disable=duplicate-code

import pandas as pd
import pytest

from survey_assist_eval.evaluation.open_questions.examples_and_leading_functions import (
    ExampleLeadingQuestionMetrics,
    add_example_and_leading_columns,
    compute_example_and_leading_metrics,
    summarise_example_and_leading_columns,
)

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
