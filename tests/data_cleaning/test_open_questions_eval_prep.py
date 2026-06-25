"""Tests for open question evaluation prep functions."""

import pandas as pd

from survey_assist_eval.data_cleaning.open_questions_eval_prep import (
    filter_nonempty_object_column,
)


def test_filter_nonempty_object_column_removes_empty_and_null_values():
    """Verify that empty and null text values are filtered out."""
    df = pd.DataFrame(
        {
            "text": ["hello", "", None, "world"],
            "other": [1, 2, 3, 4],
        }
    )

    filtered = filter_nonempty_object_column(df, "text")

    assert list(filtered["text"]) == ["hello", "world"]
    assert list(filtered["other"]) == [1, 4]
