"""Tests for text statistics functions."""

# pylint: disable=redefined-outer-name

import pandas as pd
import pytest

from survey_assist_eval.evaluation.open_questions.text_statistics_functions import (
    OpenQuestionTextStatistics,
    add_text_stats_columns,
    compare_text_statistics,
    compute_text_statistics,
    get_text_stats,
    summarise_text_stat_columns,
    word_counts_per_sentence,
)

# ============================================================================
# Test Data - Shared between tests
# ============================================================================


@pytest.fixture
def sample_text_statistics_df():
    """Provide sample text rows for compute_text_statistics wrapper tests."""
    return pd.DataFrame(
        {
            "text": [
                "One two three.",
                "Short sentence.",
                "This sentence has more than five words to test thresholds.",
            ]
        }
    )


# ============================================================================
# Test word_counts_per_sentence function
# ============================================================================


def test_word_counts_per_sentence_single_sentence():
    """Verify word counts for a single sentence."""
    text = "Hello world, how are you?"
    result = word_counts_per_sentence(text)

    assert result == [5]


def test_word_counts_per_sentence_multiple_sentences():
    """Validate word counts across multiple sentences separated by periods."""
    text = "Hello world. This is a test."
    result = word_counts_per_sentence(text)

    assert result == [2, 4]


def test_word_counts_per_sentence_mixed_delimiters():
    """Ensure sentences are split on period, exclamation mark, and question mark."""
    text = "Hello! How are you? I am fine."
    result = word_counts_per_sentence(text)

    assert result == [1, 3, 3]


def test_word_counts_per_sentence_ignores_whitespace():
    """Confirm that leading and trailing whitespace is handled correctly."""
    text = "One.  Two   . Three."
    result = word_counts_per_sentence(text)

    assert result == [1, 1, 1]


def test_word_counts_per_sentence_empty_sentences_excluded():
    """Verify that empty sentences are not included in the result."""
    text = "Hello...World."
    result = word_counts_per_sentence(text)

    assert result == [1, 1]


def test_word_counts_per_sentence_empty_string():
    """Check that an empty string returns an empty list."""
    text = ""
    result = word_counts_per_sentence(text)

    assert result == []


# ============================================================================
# Test get_text_stats function
# ============================================================================


def test_get_text_stats_returns_expected_values():
    """Ensure text statistics are computed correctly for a sample string."""
    stats = get_text_stats("Hello world. This is a test.")

    assert stats["word_count"] == 6
    assert stats["sentence_count"] == 1
    assert stats["character_count"] == 23
    assert stats["letter_count"] == 21
    assert stats["words_per_sentence"] == [2, 4]
    assert stats["mean_words_per_sentence"] == pytest.approx(6)


# ============================================================================
# Test add_text_stats_columns function
# ============================================================================


def test_add_text_stats_columns_adds_prefixed_columns():
    """Check that text stat columns are added and prefixed correctly."""
    df = pd.DataFrame({"answer": ["One two three.", "Another short answer."]})

    result = add_text_stats_columns(df, text_column="answer", prefix="answer_")

    assert "answer_word_count" in result.columns
    assert "answer_sentence_count" in result.columns
    assert "answer_mean_words_per_sentence" in result.columns
    assert list(result["answer_word_count"]) == [3, 3]
    assert list(result["answer_sentence_count"]) == [1, 1]


# ============================================================================
# Test summarise_text_stat_columns function
# ============================================================================


def test_summarise_text_stat_columns_computes_summary():
    """Validate summary statistics are computed from precomputed columns."""
    df = pd.DataFrame(
        {
            "answer": [
                "One two three.",
                "A single sentence answer.",
                "Good morning. This sentence is longer then 10 words to as a test.",
            ]
        }
    )

    df_with_stats = add_text_stats_columns(
        df,
        text_column="answer",
        prefix="answer_",
    )

    summary = summarise_text_stat_columns(
        df_with_stats,
        prefix="answer_",
        long_sentence_threshold=10,
    )

    assert summary["n_count"] == 3
    assert summary["mean_word_count"] == pytest.approx(6.666666666666667)
    assert summary["sd_word_count"] == pytest.approx(5.507570547286102)
    assert summary["median_word_count"] == 4
    assert summary["mean_sentence_count"] == pytest.approx(1.0)
    assert summary["mean_word_count_per_sentence"] == pytest.approx(5)
    assert summary["pct_over_sentence_count_threshold"] == pytest.approx(0.0)
    assert summary["pct_with_long_sentence_over_word_count_threshold"] == pytest.approx(
        1 / 3 * 100
    )
    assert summary["pct_over_word_count_threshold"] == pytest.approx(0.0)
    assert summary["pct_blank_or_too_short"] == pytest.approx(0.0)


def test_summarise_text_stat_columns_uses_existing_prefix_columns():
    """Verify summary statistics are computed from pre-existing prefixed columns."""
    df = pd.DataFrame(
        {
            "answer_word_count": [1, 30, 5],
            "answer_sentence_count": [1, 3, 1],
            "answer_words_per_sentence": [[1], [3, 21, 6], [5]],
            "answer_mean_words_per_sentence": [1.0, 25.0, 5.0],
        }
    )

    summary = summarise_text_stat_columns(df, prefix="answer_")

    assert summary["n_count"] == 3
    assert summary["mean_word_count"] == pytest.approx(12.0)
    assert summary["pct_over_word_count_threshold"] == pytest.approx(1 / 3 * 100)
    assert summary["pct_over_sentence_count_threshold"] == pytest.approx(1 / 3 * 100)
    assert summary["pct_with_long_sentence_over_word_count_threshold"] == pytest.approx(
        1 / 3 * 100
    )


def test_summarise_text_stat_columns_raises_when_columns_missing():
    """Confirm error when required stat columns are not present."""
    df = pd.DataFrame({"text": ["one two"]})

    with pytest.raises(KeyError):
        summarise_text_stat_columns(df, prefix="text_")


# ============================================================================
# Test compare_text_statistics function
# ============================================================================


def test_compare_text_statistics_dict_input_returns_dataframe():
    """Ensure the comparison helper returns a DataFrame with proper labels from dict input."""
    df_a = pd.DataFrame({"answer": ["One two.", "Another sentence."]})
    df_b = pd.DataFrame({"answer": ["Short.", "Longer response here."]})

    result = compare_text_statistics(
        {"group_a": df_a, "group_b": df_b}, text_column="answer"
    )

    assert list(result.index) == ["group_a", "group_b"]
    assert "mean_word_count" in result.columns
    assert result.loc["group_a", "n_count"] == 2
    assert result.loc["group_b", "n_count"] == 2


def test_compare_text_statistics_preserves_labels_in_output():
    """Validate that dataset labels are preserved in the result index."""
    df_a = pd.DataFrame({"answer": ["One two."]})
    df_b = pd.DataFrame({"answer": ["Short."]})

    result = compare_text_statistics({"A": df_a, "B": df_b}, text_column="answer")

    assert list(result.index) == ["A", "B"]
    assert result.loc["A", "mean_word_count"] == pytest.approx(2.0)
    assert result.loc["B", "mean_word_count"] == pytest.approx(1.0)


def test_compare_text_statistics_raises_when_no_inputs_are_provided():
    """Confirm comparison requires either raw text or precomputed columns."""
    with pytest.raises(
        ValueError, match="Provide at least one of 'text_column' or 'prefix'."
    ):
        compare_text_statistics({})


def test_compare_text_statistics_uses_existing_stat_columns():
    """Uses existing text statistic columns when a prefix is provided."""
    df = pd.DataFrame(
        {
            "stat_word_count": [2, 4],
            "stat_sentence_count": [1, 1],
            "stat_mean_words_per_sentence": [2, 4],
            "stat_has_long_sentence": [False, False],
            "stat_is_blank_or_too_short": [False, False],
            "stat_words_per_sentence": [[5], [4]],
        }
    )

    result = compare_text_statistics(
        {"group_a": df},
        prefix="stat_",
    )

    assert result.loc["group_a", "n_count"] == 2


def test_compare_text_statistics_uses_custom_prefix_with_text_column():
    """Uses the supplied prefix when computing text statistics."""
    df = pd.DataFrame({"answer": ["One two.", "Another sentence."]})

    result = compare_text_statistics(
        {"group_a": df},
        text_column="answer",
        prefix="custom_",
    )

    assert result.loc["group_a", "n_count"] == 2


# ============================================================================
# Test compute_text_statistics function
# ============================================================================


def test_compute_text_statistics_returns_metrics_model(sample_text_statistics_df):
    """Validate that the wrapper computes metrics and returns a structured model."""
    metrics = compute_text_statistics(
        sample_text_statistics_df,
        text_column="text",
        word_threshold=3,
        sentence_threshold=1,
        long_sentence_threshold=4,
        short_word_count_threshold=2,
    )

    assert isinstance(metrics, OpenQuestionTextStatistics)
    assert metrics.n_count == 3, "Expected the wrapper to count all input rows."
    assert metrics.mean_sentence_count == pytest.approx(1.0)
    assert metrics.pct_blank_or_too_short == pytest.approx(33.33333333333333)
    assert metrics.pct_over_word_count_threshold == pytest.approx(33.33333333333333)


def test_compute_text_statistics_returns_expected_model_type(
    sample_text_statistics_df,
):
    """Returns an OpenQuestionTextStatistics object."""
    result = compute_text_statistics(
        sample_text_statistics_df,
        text_column="text",
    )

    assert isinstance(result, OpenQuestionTextStatistics)


def test_compute_text_statistics_populates_all_metrics(
    sample_text_statistics_df,
):
    """Returned model contains all expected metrics."""
    result = compute_text_statistics(
        sample_text_statistics_df,
        text_column="text",
    )

    assert result.n_count > 0
    assert isinstance(result.median_word_count, float)
    assert isinstance(result.sd_word_count, float)
    assert isinstance(result.mean_sentence_count, float)
    assert isinstance(result.mean_word_count_per_sentence, float)
    assert isinstance(result.pct_over_word_count_threshold, float)
    assert isinstance(result.pct_over_sentence_count_threshold, float)
    assert isinstance(
        result.pct_with_long_sentence_over_word_count_threshold,
        float,
    )
    assert isinstance(result.pct_blank_or_too_short, float)


def test_compute_text_statistics_respects_custom_thresholds(
    sample_text_statistics_df,
):
    """Runs successfully when custom thresholds are provided."""
    result = compute_text_statistics(
        sample_text_statistics_df,
        text_column="text",
        word_threshold=10,
        sentence_threshold=1,
        long_sentence_threshold=5,
        short_word_count_threshold=1,
    )

    assert isinstance(result, OpenQuestionTextStatistics)


# ============================================================================
# Test OpenQuestionTextStatistics function
# ============================================================================


def test_open_question_text_statistics_report_metrics_formats_output():
    """Ensure the report helper renders the expected metric labels."""
    metrics = OpenQuestionTextStatistics(
        n_count=3,
        median_word_count=3.0,
        sd_word_count=1.5,
        mean_sentence_count=1.0,
        mean_word_count_per_sentence=2.5,
        pct_over_word_count_threshold=33.3,
        pct_over_sentence_count_threshold=0.0,
        pct_with_long_sentence_over_word_count_threshold=66.7,
        pct_blank_or_too_short=33.3,
    )

    report = metrics.report_metrics()

    assert "Text statistics:" in report
    assert "Number of open questions: 3" in report
    assert "Median Word Count: 3.00" in report
    assert "Percent Over Word Threshold Count: 33.30%" in report
    assert "Percent with Blank or Too Short Sentences: 33.30%" in report


def test_open_question_text_statistics_report_metrics_returns_string():
    """Report metrics returns a formatted string."""
    metrics = OpenQuestionTextStatistics(
        n_count=10,
        median_word_count=5.0,
        sd_word_count=1.0,
        mean_sentence_count=1.5,
        mean_word_count_per_sentence=4.0,
        pct_over_word_count_threshold=10.0,
        pct_over_sentence_count_threshold=20.0,
        pct_with_long_sentence_over_word_count_threshold=30.0,
        pct_blank_or_too_short=5.0,
    )

    result = metrics.report_metrics()

    assert isinstance(result, str)


def test_open_question_text_statistics_report_metrics_contains_metrics():
    """Report output contains key metric values."""
    metrics = OpenQuestionTextStatistics(
        n_count=10,
        median_word_count=5.0,
        sd_word_count=1.0,
        mean_sentence_count=1.5,
        mean_word_count_per_sentence=4.0,
        pct_over_word_count_threshold=10.0,
        pct_over_sentence_count_threshold=20.0,
        pct_with_long_sentence_over_word_count_threshold=30.0,
        pct_blank_or_too_short=5.0,
    )

    report = metrics.report_metrics()

    assert "Number of open questions: 10" in report
    assert "Median Word Count: 5.00" in report
    assert "Percent with Blank or Too Short Sentences: 5.00%" in report
