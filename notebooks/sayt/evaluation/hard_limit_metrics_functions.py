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

    suggestions_col: str
    num_queries: int  # Total number of queries evaluated
    cutoff_k: int  # The hard limit (e.g., 9)
    max_suggestions_returned: int  # Max returned in this dataset
    pct_exceeding_limit: float  # % of queries returning > cutoff_k suggestions
    pct_correct_overall: float  # % of correct answers overall
    pct_correct_within_limit: float  # % of correct answers within the limit
    pct_correct_outside_limit: float  # % of correct answers beyond the limit
    pct_within_limit_at_tie_score: (
        float  # % with correct answer only at boundary by tie
    )

    def report_metrics(self) -> str:
        """Generate a human-readable report of the metrics."""
        report = (
            f"SAYT Hard Limit Metrics for column '{self.suggestions_col}':\n"
            f" Number of queries: {self.num_queries}\n"
            f" Suggestions limit (k): {self.cutoff_k}\n"
            f" Maximum suggestions returned: {self.max_suggestions_returned}\n"
            f" % of queries exceeding limit: {self.pct_exceeding_limit:.2%}\n"
            f" % of queries with a correct suggestion: "
            f"{self.pct_correct_overall:.2%}\n"
            " % of queries with a correct suggestion within limit: "
            f"{self.pct_correct_within_limit:.2%}\n"
            " % of queries with a correct suggestion outside limit: "
            f"{self.pct_correct_outside_limit:.2%}\n"
            "% of queries with a correct in-limit suggestion tied across the limit boundary: "
            f"{self.pct_within_limit_at_tie_score:.2%}"
        )
        return report


def compute_suggestions_hard_limit_metrics(  # noqa: PLR0913 pylint: disable = R0913, R0917
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

    if score_col not in df.columns:
        raise ValueError(
            f"Score column '{score_col}' not found in DataFrame. Please "
            f"ensure with_scores is set to True when generating suggestions."
        )

    df["_retrieved_codes"] = df.apply(
        get_codes_from_suggestions,
        suggestions_col=suggestions_col,
        code_length=code_length,
        axis=1,
    )

    df = add_sayt_hard_limit_metrics_columns(
        df=df,
        retrieved_codes_col="_retrieved_codes",
        correct_code_col=correct_code_col,
        cutoff_k=cutoff_k,
        score_col=score_col,
    )

    return summarise_hard_limit_metrics(
        df=df,
        suggestions_col=suggestions_col,
        cutoff_k=cutoff_k,
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


def add_sayt_hard_limit_metrics_columns(  # noqa: PLR0913 pylint: disable = R0913, R0917
    df,
    retrieved_codes_col: str,
    correct_code_col: str,
    cutoff_k: int,
    score_col: str = "score",
    prefix: str | None = None,
):
    """Add columns to the DataFrame for hard limit metrics evaluation.

    Args:
        df: DataFrame containing the suggestions and correct codes.
        retrieved_codes_col: The name of the column with the retrieved codes.
        correct_code_col: The name of the column with the correct codes.
        cutoff_k: The hard limit on the number of suggestions returned.
        score_col: The name of the column with the suggestion scores.
        prefix: Optional prefix for the new columns.

    Returns:
        DataFrame: The DataFrame with added hard limit metrics columns.
    """
    if prefix is None:
        prefix = ""

    df = df.copy()

    df[f"{prefix}correct_code_rank"] = df.apply(
        lambda row: get_rank_of_correct_code(
            row[retrieved_codes_col], row[correct_code_col]
        ),
        axis=1,
    )

    df[f"{prefix}num_retrieved_codes"] = df[retrieved_codes_col].apply(len)
    df[f"{prefix}correct_code_within_cutoff"] = (
        df[f"{prefix}correct_code_rank"] <= cutoff_k
    )
    df[f"{prefix}correct_code_outside_cutoff"] = (
        df[f"{prefix}correct_code_rank"] > cutoff_k
    )
    df[f"{prefix}correct_in_cutoff_by_default"] = df.apply(
        lambda row: is_correct_code_tied_with_boundary(
            row[score_col], row[f"{prefix}correct_code_rank"], cutoff_k
        ),
        axis=1,
    )
    return df


def summarise_hard_limit_metrics(
    df,
    suggestions_col: str,
    cutoff_k: int,
    prefix: str = "",
) -> SAYTHardLimitMetrics:
    """Compute summary metrics for hard limit suggestions evaluation.

    Args:
        df: DataFrame containing the suggestions and correct codes.
        suggestions_col: The name of the column with the suggestions.
        cutoff_k: The hard limit on the number of suggestions returned.
        prefix: Optional prefix for the metric columns.

    Returns:
        SAYTHardLimitMetrics: The computed hard limit metrics.
    """
    summary = {
        "suggestions_col": suggestions_col,
        "num_queries": len(df),
        "cutoff_k": cutoff_k,
        "max_suggestions_returned": df[f"{prefix}num_retrieved_codes"].max(),
        "pct_exceeding_limit": df[f"{prefix}num_retrieved_codes"].gt(cutoff_k).mean(),
        "pct_correct_overall": (df[f"{prefix}correct_code_rank"] > 0).mean(),
        "pct_correct_within_limit": df[f"{prefix}correct_code_within_cutoff"].mean(),
        "pct_correct_outside_limit": df[f"{prefix}correct_code_outside_cutoff"].mean(),
        "pct_within_limit_at_tie_score": df[
            f"{prefix}correct_in_cutoff_by_default"
        ].mean(),
    }

    return SAYTHardLimitMetrics(**summary)


def build_sayt_hard_limit_metrics_comparison_table(
    df,
    suggestions_cols_to_compare: list[str],
    correct_code_col: str,
    cutoff_k: int,
):
    """Build a comparison table of performance metrics across suggestion columns.

    Args:
        df: DataFrame containing the retrieved suggestions and correct codes.
        suggestions_cols_to_compare: List of column names containing
            the retrieved suggestions to compare.
        correct_code_col: Column name containing the correct code.
        cutoff_k: The hard limit on the number of suggestions returned.

    Returns:
        pd.DataFrame: One row per suggestion column with all hard limit metrics.
    """
    performance_metrics = pd.DataFrame()
    code_length = len(df[correct_code_col].iloc[0])  # inferred from first row

    for col in suggestions_cols_to_compare:
        score_col = col.replace("suggestions_", "scores_")
        performance_metrics_tmp = {
            **compute_suggestions_hard_limit_metrics(
                df,
                correct_code_col=correct_code_col,
                suggestions_col=col,
                cutoff_k=cutoff_k,
                code_length=code_length,
                score_col=score_col,
            ).__dict__,
        }

        performance_metrics = pd.concat(
            [performance_metrics, pd.DataFrame([performance_metrics_tmp])],
            ignore_index=True,
        )
    return performance_metrics
