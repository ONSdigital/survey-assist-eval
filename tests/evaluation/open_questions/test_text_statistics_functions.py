"""Tests for text statistics functions."""

import pandas as pd
import pytest

from survey_assist_eval.evaluation.open_questions.text_statistics_functions import (
    add_text_stats_columns,
    compare_text_stats,
    get_text_stats,
    summarise_text_stats,
    word_counts_per_setence,
)


def test_word_counts_per_setence_single_sentence():
    """Verify word counts for a single sentence."""
    text = "Hello world. How are you?"
    result = word_counts_per_setence(text)

    assert result == [2, 3]


def test_word_counts_per_setence_multiple_sentences():
    """Validate word counts across multiple sentences separated by periods."""
    text = "Hello world. This is a test."
    result = word_counts_per_setence(text)

    assert result == [2, 4]


def test_word_counts_per_setence_mixed_delimiters():
    """Ensure sentences are split on period, exclamation mark, and question mark."""
    text = "Hello! How are you? I am fine."
    result = word_counts_per_setence(text)

    assert result == [1, 3, 3]


def test_word_counts_per_setence_ignores_whitespace():
    """Confirm that leading and trailing whitespace is handled correctly."""
    text = "One.  Two   . Three."
    result = word_counts_per_setence(text)

    assert result == [1, 1, 1]


def test_word_counts_per_setence_empty_sentences_excluded():
    """Verify that empty sentences are not included in the result."""
    text = "Hello...World."
    result = word_counts_per_setence(text)

    assert result == [1, 1]


def test_word_counts_per_setence_empty_string():
    """Check that an empty string returns an empty list."""
    text = ""
    result = word_counts_per_setence(text)

    assert result == []


def test_get_text_stats_returns_expected_values():
    """Ensure text statistics are computed correctly for a sample string."""
    stats = get_text_stats("Hello world. This is a test.")

    assert stats["word_count"] == 6
    assert stats["sentence_count"] == 1
    assert stats["syllable_count"] == 7
    assert stats["character_count"] == 23
    assert stats["letter_count"] == 21
    assert stats["words_per_sentence"] == [2, 4]
    assert stats["mean_words_per_sentence"] == pytest.approx(6)
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
    df = pd.DataFrame(
        {
            "answer": [
                "One two three.",
                "A single sentence answer.",
                "Good morning. This sentence is longer then 10 words to as a test.",
            ]
        }
    )

    summary = summarise_text_stats(
        df, prefix=None, text_column="answer", long_sentence_threshold=10
    )

    assert summary["n_count"] == 3
    assert summary["mean_word_count"] == pytest.approx(6.666666666666667)
    assert summary["sd_word_count"] == pytest.approx(5.507570547286102)
    assert summary["median_word_count"] == 4
    assert summary["mean_sentence_count"] == pytest.approx(1.0)
    assert summary["mean_word_count_per_sentence"] == pytest.approx(5)
    assert summary["mean_of_mean_syllables_per_word"] == pytest.approx(
        1.3269230769230769
    )
    assert summary["pct_over_sentence_count_threshold"] == pytest.approx(0.0)
    assert summary["pct_with_long_sentence_over_word_count_threshold"] == pytest.approx(
        1 / 3 * 100
    )
    assert summary["pct_over_word_count_threshold"] == pytest.approx(0.0)
    assert summary["pct_blank_or_too_short"] == pytest.approx(0.0)


def test_summarise_text_stats_uses_existing_prefix_columns():
    """Verify summary statistics are computed from pre-existing prefixed columns."""
    df = pd.DataFrame(
        {
            "answer_word_count": [1, 30, 5],
            "answer_sentence_count": [1, 3, 1],
            "answer_words_per_sentence": [[1], [3, 21, 6], [5]],
            "answer_mean_words_per_sentence": [1.0, 25.0, 5.0],
            "answer_mean_syllables_per_word": [1.0, 1.2, 1.1],
        }
    )

    summary = summarise_text_stats(df, prefix="answer_")

    assert summary["n_count"] == 3
    assert summary["mean_word_count"] == pytest.approx(12.0)
    assert summary["pct_over_word_count_threshold"] == pytest.approx(1 / 3 * 100)
    assert summary["pct_over_sentence_count_threshold"] == pytest.approx(1 / 3 * 100)
    assert summary["pct_with_long_sentence_over_word_count_threshold"] == pytest.approx(
        1 / 3 * 100
    )


def test_summarise_text_stats_raises_when_no_prefix_or_text_column():
    """Confirm that a ValueError is raised when neither prefix nor text_column is provided."""
    df = pd.DataFrame({"text": ["one two"]})

    with pytest.raises(ValueError, match="Provide either text_column or prefix"):
        summarise_text_stats(df, prefix=None, text_column=None)


def test_compare_text_stats_dict_input_returns_dataframe():
    """Ensure the comparison helper returns a DataFrame with proper labels from dict input."""
    df_a = pd.DataFrame({"answer": ["One two.", "Another sentence."]})
    df_b = pd.DataFrame({"answer": ["Short.", "Longer response here."]})

    result = compare_text_stats(
        {"group_a": df_a, "group_b": df_b}, text_column="answer"
    )

    assert list(result.index) == ["group_a", "group_b"]
    assert "mean_word_count" in result.columns
    assert result.loc["group_a", "n_count"] == 2
    assert result.loc["group_b", "n_count"] == 2


def test_compare_text_stats_preserves_labels_in_output():
    """Validate that dataset labels are preserved in the result index."""
    df_a = pd.DataFrame({"answer": ["One two."]})
    df_b = pd.DataFrame({"answer": ["Short."]})

    result = compare_text_stats({"A": df_a, "B": df_b}, text_column="answer")

    assert list(result.index) == ["A", "B"]
    assert result.loc["A", "mean_word_count"] == pytest.approx(2.0)
    assert result.loc["B", "mean_word_count"] == pytest.approx(1.0)
