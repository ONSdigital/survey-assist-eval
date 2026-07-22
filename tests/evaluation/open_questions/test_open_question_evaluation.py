"""Tests for open question evaluation functions."""

import pandas as pd

from survey_assist_eval.evaluation.open_questions.open_questions_evaluation import (
    OpenQuestionEvaluation,
    add_open_question_evaluation_columns,
    evaluate_open_questions,
    filter_nonempty_object_column,
)
from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    QuestionStructureMetrics,
)
from survey_assist_eval.evaluation.open_questions.simple_language_functions import (
    SimpleLanguageMetrics,
)
from survey_assist_eval.evaluation.open_questions.text_statistics_functions import (
    OpenQuestionTextStatistics,
)

# ============================================================================
# Test OpenQuestionEvaluation model
# ============================================================================


def test_open_question_evaluation_as_dict_returns_expected_structure():
    """Return metrics as a nested dictionary."""
    df = pd.DataFrame(
        {
            "question": [
                "What do you do?",
                "Describe your role.",
            ]
        }
    )

    evaluation = evaluate_open_questions(
        df,
        text_column="question",
    )

    result = evaluation.as_dict()

    assert "text_statistics" in result
    assert "question_structure" in result
    assert "simple_language" in result
    assert isinstance(result["text_statistics"], dict)
    assert isinstance(result["question_structure"], dict)
    assert isinstance(result["simple_language"], dict)


def test_open_question_evaluation_report_metrics_returns_string():
    """Return a formatted metrics report."""
    df = pd.DataFrame(
        {
            "question": [
                "What do you do?",
                "Describe your role.",
            ]
        }
    )

    evaluation = evaluate_open_questions(
        df,
        text_column="question",
    )

    report = evaluation.report_metrics()

    assert isinstance(report, str)
    assert "Open Question Evaluation metrics summary:" in report


# ============================================================================
# Test evaluate_open_questions function
# ============================================================================


def test_evaluate_open_questions_returns_expected_model():
    """Return an OpenQuestionEvaluation object."""
    df = pd.DataFrame(
        {
            "question": [
                "What do you do?",
                "Describe your responsibilities.",
            ]
        }
    )

    result = evaluate_open_questions(
        df,
        text_column="question",
    )

    assert isinstance(result, OpenQuestionEvaluation)
    assert isinstance(result.text_statistics, OpenQuestionTextStatistics)
    assert isinstance(result.question_structure, QuestionStructureMetrics)
    assert isinstance(result.simple_language, SimpleLanguageMetrics)


def test_evaluate_open_questions_filters_empty_rows():
    """Exclude blank and null responses before evaluation."""
    df = pd.DataFrame(
        {
            "question": [
                "What do you do?",
                "",
                None,
                "Describe your responsibilities.",
            ]
        }
    )

    result = evaluate_open_questions(
        df,
        text_column="question",
    )

    assert result.text_statistics.n_count == 2
    assert result.question_structure.n_count == 2
    assert result.simple_language.n_count == 2


def test_evaluate_open_questions_uses_default_text_statistics_config():
    """Use default thresholds when no config is supplied."""
    df = pd.DataFrame(
        {
            "question": [
                "What do you do?",
            ]
        }
    )

    result = evaluate_open_questions(
        df,
        text_column="question",
    )

    assert isinstance(result, OpenQuestionEvaluation)


def test_evaluate_open_questions_uses_custom_text_statistics_config():
    """Custom threshold settings should be forwarded to text statistics evaluation."""
    df = pd.DataFrame(
        {
            "question": [
                "This sentence has many words and should exceed the custom threshold.",
            ]
        }
    )

    result = evaluate_open_questions(
        df,
        text_column="question",
        text_statistics_config={"word_threshold": 3},
    )

    assert result.text_statistics.pct_over_word_count_threshold == 100.0
    assert result.simple_language.n_count == 1


def test_evaluate_open_questions_counts_contractions_and_hyphens_end_to_end():
    """Contractions and hyphenated words should count as split tokens end-to-end."""
    df = pd.DataFrame(
        {
            "question": [
                "Don't re-enter high-quality data.",
            ]
        }
    )

    result = evaluate_open_questions(
        df,
        text_column="question",
        text_statistics_config={"word_threshold": 6},
    )

    assert result.text_statistics.n_count == 1
    assert result.text_statistics.mean_word_count == 7.0
    assert result.text_statistics.median_word_count == 7.0
    assert result.text_statistics.pct_over_word_count_threshold == 100.0


# ============================================================================
# Test add_open_question_evaluation_columns function
# ============================================================================


def test_add_open_question_evaluation_columns_adds_expected_metric_columns():
    """Add all expected open question metric columns using text-column prefix."""
    df = pd.DataFrame(
        {
            "question": [
                "What do you do?",
                "Describe your role.",
            ],
            "id": [1, 2],
        }
    )

    result = add_open_question_evaluation_columns(df, text_column="question")

    expected_columns = {
        "question_has_question_mark",
        "question_interrogative_start",
        "question_instruction_prompt_start",
        "question_interrogative_wh_count",
        "question_instruction_prompt_count",
        "question_is_question",
        "question_contains_multiple_asks",
        "question_is_single_question",
        "question_n_acronyms",
        "question_avg_syllables_per_word",
        "question_syllable_counts",
        "question_word_count",
        "question_sentence_count",
        "question_character_count",
        "question_letter_count",
        "question_words_per_sentence",
        "question_mean_words_per_sentence",
    }

    assert expected_columns.issubset(set(result.columns))
    assert "question" in result.columns
    assert "id" in result.columns


def test_add_open_question_evaluation_columns_uses_provided_text_column_name_as_prefix():
    """Use the provided text column name when building metric column prefixes."""
    df = pd.DataFrame(
        {
            "prompt_text": [
                "What is your role?",
            ]
        }
    )

    result = add_open_question_evaluation_columns(df, text_column="prompt_text")

    assert "prompt_text_is_question" in result.columns
    assert "prompt_text_n_acronyms" in result.columns
    assert "prompt_text_word_count" in result.columns


def test_add_open_question_evaluation_columns_preserves_rows_including_empty_and_null():
    """Preserve all input rows and compute metrics even when text is null/blank."""
    df = pd.DataFrame(
        {
            "question": [
                "What do you do?",
                "",
                None,
            ],
            "id": [10, 20, 30],
        }
    )

    result = add_open_question_evaluation_columns(df, text_column="question")

    assert len(result) == 3
    assert list(result["id"]) == [10, 20, 30]
    assert not bool(result.loc[1, "question_is_question"])
    assert not bool(result.loc[2, "question_is_question"])


# ============================================================================
# Test filter_nonempty_object_column function
# ============================================================================


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
