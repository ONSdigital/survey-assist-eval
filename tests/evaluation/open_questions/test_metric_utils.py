"""Tests for metric utils functions."""

# pylint: disable=redefined-outer-name

import pandas as pd
import pytest

from survey_assist_eval.evaluation.open_questions.metric_utils import (
    add_metrics_columns,
)

# ============================================================================
# Test add_metrics_columns function
# ============================================================================


def dummy_metrics(text: str) -> dict[str, int | bool]:
    """Return simple metrics for testing."""
    return {
        "length": len(text),
        "is_empty": text == "",
    }


@pytest.fixture
def metrics_input_df():
    """Return input data for add_metrics_columns tests."""
    return pd.DataFrame(
        {
            "text": [
                "Hello",
                "World",
                None,
            ],
            "id": [1, 2, 3],
        }
    )


@pytest.fixture
def expected_metrics_df():
    """Return expected output DataFrame with default-prefixed columns."""
    return pd.DataFrame(
        {
            "text": ["Hello", "World", None],
            "id": [1, 2, 3],
            "text_length": [5, 5, 0],
            "text_is_empty": [False, False, True],
        }
    )


EXPECTED_DEFAULT_COLUMNS = {
    "text_length",
    "text_is_empty",
}

EXPECTED_CUSTOM_COLUMNS = {
    "metric_length",
    "metric_is_empty",
}


def test_add_metrics_columns_returns_expected_dataframe(
    metrics_input_df,
    expected_metrics_df,
):
    """Adds expected metric columns and values."""
    result = add_metrics_columns(
        metrics_input_df,
        text_column="text",
        metrics_func=dummy_metrics,
    )

    pd.testing.assert_frame_equal(
        result,
        expected_metrics_df,
        obj="Expected add_metrics_columns to return the expected DataFrame "
        "containing the original columns and computed metric columns",
    )


def test_add_metrics_columns_adds_default_prefixed_columns(
    metrics_input_df,
):
    """Uses the text column name as the default prefix."""
    result = add_metrics_columns(
        metrics_input_df,
        text_column="text",
        metrics_func=dummy_metrics,
    )

    assert EXPECTED_DEFAULT_COLUMNS.issubset(result.columns), (
        "Expected add_metrics_columns to add all metric columns using "
        "the text column name as the default prefix"
    )


def test_add_metrics_columns_preserves_original_columns(
    metrics_input_df,
):
    """Preserves original columns."""
    result = add_metrics_columns(
        metrics_input_df,
        text_column="text",
        metrics_func=dummy_metrics,
    )

    assert set(metrics_input_df.columns).issubset(
        result.columns
    ), "Expected add_metrics_columns to preserve all original DataFrame columns"


def test_add_metrics_columns_uses_custom_prefix(
    metrics_input_df,
):
    """Uses a custom prefix when supplied."""
    result = add_metrics_columns(
        metrics_input_df,
        text_column="text",
        metrics_func=dummy_metrics,
        prefix="metric_",
    )

    assert EXPECTED_CUSTOM_COLUMNS.issubset(result.columns), (
        "Expected add_metrics_columns to add all metric columns using "
        "the supplied custom prefix"
    )


def test_add_metrics_columns_does_not_add_default_prefix_when_custom_prefix_used(
    metrics_input_df,
):
    """Does not add default-prefixed columns when a custom prefix is supplied."""
    result = add_metrics_columns(
        metrics_input_df,
        text_column="text",
        metrics_func=dummy_metrics,
        prefix="metric_",
    )

    assert EXPECTED_DEFAULT_COLUMNS.isdisjoint(result.columns), (
        "Expected default-prefixed metric columns not to be added when "
        "a custom prefix is supplied"
    )


def test_add_metrics_columns_handles_missing_values_as_empty_text(
    metrics_input_df,
):
    """Treats missing text values as empty strings."""
    result = add_metrics_columns(
        metrics_input_df,
        text_column="text",
        metrics_func=dummy_metrics,
    )

    assert result.loc[2, "text_length"] == 0, (
        "Expected missing text values to be converted to empty strings "
        "before applying the metrics function"
    )

    assert result.loc[2, "text_is_empty"], (
        "Expected missing text values to be treated as empty strings "
        "when computing metrics"
    )


def test_add_metrics_columns_raises_key_error_for_missing_text_column():
    """Raises KeyError when the text column is missing."""
    df = pd.DataFrame({"other_column": ["hello"]})

    with pytest.raises(
        KeyError,
        match="text",
    ):
        add_metrics_columns(
            df,
            text_column="text",
            metrics_func=dummy_metrics,
        )


def test_add_metrics_columns_uses_default_prefix_when_prefix_is_none(
    metrics_input_df,
):
    """Uses the text column name when prefix=None."""
    result = add_metrics_columns(
        metrics_input_df,
        text_column="text",
        metrics_func=dummy_metrics,
        prefix=None,
    )

    assert EXPECTED_DEFAULT_COLUMNS.issubset(result.columns), (
        "Expected add_metrics_columns to use the text column name as the "
        "default prefix when prefix=None"
    )
