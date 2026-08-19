"""Functions for computing suggestions hard limit metrics for SAYT evaluation."""

# pylint: disable=duplicate-code
import pandas as pd
from pydantic import BaseModel

from notebooks.sayt.sayt_utils import get_codes_from_suggestions
from survey_assist_eval.evaluation.sayt.performance_metrics_functions import (
    get_rank_of_correct_code,
)


class SAYTHardLimitMetrics(BaseModel):
    """Metrics for evaluating SAYT performance under a hard cutoff limit,
    including analysis of boundary behavior when suggestion scores are tied.
    Aggregated across a dataframe of queries.
    """

    num_queries: int  # Total number of queries evaluated
    cutoff_k: int  # The hard limit (e.g., 9)
    actual_max_returned: int  # Max returned in this dataset
    pct_queries_exceeding_limit: float  # % of queries returning > cutoff_k suggestions
    pct_correct_outside_limit: float  # % of correct answers beyond the limit
    pct_correct_within_limit: float  # % of correct answers within the limit
    pct_correct_total: float  # % of correct answers overall
    pct_correct_tied_at_boundary: float  # % with correct answer only at boundary by tie


def compute_hard_limit_suggestions_metrics_from_suggestions(  # noqa: PLR0913 pylint: disable = R0913, R0917
    df,
    correct_code_col: str,
    suggestions_col: str,
    cutoff_k: int,
    code_length: int,
    score_col: str = "score",
) -> SAYTHardLimitMetrics:
    """Compute the hard limit metrics for a given suggester and prefix length.

    Args:
        df: The DataFrame containing the data.
        correct_code_col: The name of the column with the correct codes.
        suggestions_col: The name of the column with the suggestions.
        cutoff_k: The hard limit on the number of suggestions returned.
        code_length: The length of the code prefix used for suggestions.
        score_col: The name of the column with the suggestion scores.

    Returns:
        SAYTHardLimitMetrics: The computed hard limit metrics.
    """
    df = df.copy()

    df["_retrieved_codes"] = df.apply(
        get_codes_from_suggestions,
        suggestions_col=suggestions_col,
        code_length=code_length,
        axis=1,
    )

    return summary_hardlimit_metrics(
        df=df,
        correct_code_col=correct_code_col,
        cutoff_k=cutoff_k,
        score_col=score_col,
    )


def is_correct_code_tied_with_boundary(
    retrieved_codes_scores: list, correct_code_rank: int | None, cutoff_k: int
) -> bool:
    """Check if a correct code within the hard limit has the same score as the boundary suggestion.

    Args:
        retrieved_codes_scores: List of scores for retrieved suggestions.
        correct_code_rank: Rank of the correct code (1-indexed), may be NaN or None.
        cutoff_k: The hard limit on the number of suggestions returned.

    Returns:
        bool: True if correct code is within limit and shares boundary score, False otherwise.
    """
    # If suggestions don't exceed the limit, it's not a tie-at-boundary situation
    if len(retrieved_codes_scores) <= cutoff_k:
        return False

    # Check for NaN/None first before any indexing
    if pd.isna(correct_code_rank) or correct_code_rank > cutoff_k:
        return False

    # Now safe to convert to int for indexing
    rank_idx = int(correct_code_rank) - 1
    boundary_idx = cutoff_k - 1

    return retrieved_codes_scores[rank_idx] == retrieved_codes_scores[boundary_idx]


def summary_hardlimit_metrics(
    df, correct_code_col: str, cutoff_k: int, score_col: str
) -> SAYTHardLimitMetrics:
    """Compute summary metrics for hard limit suggestions evaluation.

    Args:
        df: DataFrame containing the suggestions and correct codes.
        correct_code_col: The name of the column with the correct codes.
        cutoff_k: The hard limit on the number of suggestions returned.
        score_col: The name of the column with the suggestion scores.

    Returns:
        SAYTHardLimitMetrics: The computed hard limit metrics.
    """
    df["correct_code_rank"] = df.apply(
        lambda row: get_rank_of_correct_code(
            row["_retrieved_codes"], row[correct_code_col]
        ),
        axis=1,
    )

    pct_queries_exceeding_limit = (df["_retrieved_codes"].apply(len) > cutoff_k).mean()

    df["correct_in_cutoff_by_default"] = df.apply(
        lambda row: is_correct_code_tied_with_boundary(
            row[score_col], row["correct_code_rank"], cutoff_k
        ),
        axis=1,
    )

    summary = {
        "num_queries": df.shape[0],
        "cutoff_k": cutoff_k,
        "actual_max_returned": df["_retrieved_codes"].apply(len).max(),
        "pct_queries_exceeding_limit": pct_queries_exceeding_limit,
        "pct_correct_outside_limit": df[df["correct_code_rank"] > cutoff_k].shape[0]
        / df.shape[0],
        "pct_correct_within_limit": df[df["correct_code_rank"] <= cutoff_k].shape[0]
        / df.shape[0],
        "pct_correct_total": df[df["correct_code_rank"] > 0].shape[0] / df.shape[0],
        "pct_correct_tied_at_boundary": df["correct_in_cutoff_by_default"].mean(),
    }

    return SAYTHardLimitMetrics(**summary)
