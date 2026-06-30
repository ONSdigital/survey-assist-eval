"""Functions for evaluating question quality in open questions by
combining structure, language, and text-based checks.
"""

import pandas as pd
from pydantic import BaseModel

from survey_assist_eval.data_cleaning.open_questions_eval_prep import (
    filter_nonempty_object_column,
)
from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    QuestionStructureMetrics,
    compute_question_structure_metrics,
)
from survey_assist_eval.evaluation.open_questions.text_statistics_functions import (
    OpenQuestionTextStatistics,
    compute_text_statistics,
)


class OpenQuestionEvaluation(BaseModel):
    """Container for all open question evaluation metrics."""

    text_statistics: OpenQuestionTextStatistics
    question_structure: QuestionStructureMetrics

    def report_metrics(self):
        """Pretty print all simple metrics."""
        lines = [
            "Open Question Evaluation metrics summary:",
            self.text_statistics.report_metrics(),
            self.question_structure.report_metrics(),
        ]
        return "\n".join(lines)

    def as_dict(self):
        """Return open question evaluation metrics as a dictionary."""
        return {
            "text_statistics": self.text_statistics.__dict__,
            "question_structure": self.question_structure.__dict__,
        }


def evaluate_open_questions(
    df: pd.DataFrame, text_column: str, text_statistics_config: dict | None = None
) -> OpenQuestionEvaluation:
    """Evaluate open questions using structure, language, and text-statistics checks.

    Args:
        df: DataFrame containing open question text.
        text_column: Column containing the open questions.
        text_statistics_config: Optional dictionary of keyword arguments passed to
            `compute_text_statistics` (e.g. thresholds such as word count or sentence count).

    Returns:
        OpenQuestionEvaluationResult containing text statistics, question structure
        checks, and simple language checks for the input data.

    Notes:
        - Each config dictionary is optional; default values are used when not provided.
        - Config dictionaries are unpacked and passed directly to the corresponding
        evaluation functions.
    """
    text_statistics_config = text_statistics_config or {}

    df = filter_nonempty_object_column(df, column=text_column)

    text_stats = compute_text_statistics(
        df, text_column=text_column, **text_statistics_config
    )

    question_struct_metrics = compute_question_structure_metrics(
        df, text_column=text_column
    )

    return OpenQuestionEvaluation(
        text_statistics=text_stats,
        question_structure=question_struct_metrics,
    )
