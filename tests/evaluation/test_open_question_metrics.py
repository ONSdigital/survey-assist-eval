"""Tests for open question evaluation metrics."""

import pandas as pd
import pytest

from survey_assist_eval.evaluation.open_questions_metrics import (
    add_text_stats_columns,
    filter_nonempty_object_column,
    get_text_stats,
    summarise_text_stats,
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


def test_get_text_stats_returns_expected_values():
    """Ensure text statistics are computed correctly for a sample string."""
    stats = get_text_stats("Hello world. This is a test.")

    assert stats["word_count"] == 6
    assert stats["sentence_count"] == 1
    assert stats["syllable_count"] == 7
    assert stats["character_count"] == 23
    assert stats["letter_count"] == 21
    assert stats["mean_words_per_sentence"] == pytest.approx(6.0)
    assert stats["mean_syllables_per_word"] == pytest.approx(1.1666666666666667)


def test_add_text_stats_columns_adds_prefixed_columns():
    """Check that text stat columns are added and prefixed correctly."""
    df = pd.DataFrame({"answer": ["One two three.", "Another short answer."]})

    result = add_text_stats_columns(df, text_column="answer", result_prefix="answer_")

    assert "answer_word_count" in result.columns
    assert "answer_sentence_count" in result.columns
    assert "answer_mean_words_per_sentence" in result.columns
    assert list(result["answer_word_count"]) == [3, 3]
    assert list(result["answer_sentence_count"]) == [1, 1]

    # original DataFrame should be unchanged when inplace=False
    assert "answer_word_count" not in df.columns


def test_add_text_stats_columns_inplace_modifies_dataframe():
    """Validate that inplace=True updates the original DataFrame."""
    df = pd.DataFrame({"answer": ["One two."]})

    result = add_text_stats_columns(
        df, text_column="answer", result_prefix="answer_", inplace=True
    )

    assert result is df
    assert "answer_word_count" in df.columns
    assert df.loc[0, "answer_word_count"] == 2


def test_summarise_text_stats_computes_summary_using_text_column():
    """Validate summary statistics are computed when a text column is provided."""
    df = pd.DataFrame({"answer": ["One two three.", "A single sentence answer."]})

    summary = summarise_text_stats(df, prefix=None, text_column="answer")

    assert summary["n_outputs"] == 2
    assert summary["mean_word_count"] == pytest.approx(3.5)
    assert summary["median_word_count"] == 3.5
    assert summary["mean_sentence_count"] == pytest.approx(1.0)
    assert summary["mean_words_per_sentence"] == pytest.approx(3.5)
    assert summary["pct_over_25_words"] == pytest.approx(0.0)
    assert summary["pct_blank_or_too_short"] == pytest.approx(0.0)


def test_summarise_text_stats_uses_existing_prefix_columns():
    """Verify summary statistics are computed from pre-existing prefixed columns."""
    df = pd.DataFrame(
        {
            "answer_word_count": [1, 30, 5],
            "answer_sentence_count": [1, 3, 1],
            "answer_mean_words_per_sentence": [1.0, 25.0, 5.0],
            "answer_mean_syllables_per_word": [1.0, 1.2, 1.1],
        }
    )

    summary = summarise_text_stats(df, prefix="answer_")

    assert summary["n_outputs"] == 3
    assert summary["mean_word_count"] == pytest.approx(12.0)
    assert summary["pct_over_25_words"] == pytest.approx(1 / 3 * 100)
    assert summary["pct_over_2_sentences"] == pytest.approx(1 / 3 * 100)
    assert summary["pct_long_sentence"] == pytest.approx(1 / 3 * 100)


def test_summarise_text_stats_raises_when_no_prefix_or_text_column():
    """Confirm that a ValueError is raised when neither prefix nor text_column is provided."""
    df = pd.DataFrame({"text": ["one two"]})

    with pytest.raises(ValueError, match="Provide either text_column or prefix"):
        summarise_text_stats(df, prefix=None, text_column=None)
