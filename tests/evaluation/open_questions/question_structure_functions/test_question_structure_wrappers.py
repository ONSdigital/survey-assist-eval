"""Tests for question structure wrapper functions."""

# pylint: disable=redefined-outer-name
# pylint: disable=duplicate-code

import pandas as pd
import pytest

from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    add_question_structure_columns,
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
# Test add_question_structure_columns function
# ============================================================================
