"""Tests for open question evaluation functions."""

import pandas as pd

from survey_assist_eval.evaluation.open_questions.open_questions_evaluation import (
    OpenQuestionEvaluation,
    evaluate_open_questions,
)
from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    QuestionStructureMetrics,
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
    assert isinstance(result["text_statistics"], dict)
    assert isinstance(result["question_structure"], dict)


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
