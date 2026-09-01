"""Performance metrics functions for SAYT evaluation."""

import pandas as pd
from pydantic import BaseModel

from notebooks.sayt.sayt_utils import get_codes_from_suggestions


class SAYTPerformanceMetrics(BaseModel):
    """Class to compute performance metrics for SAYT evaluation."""

    suggestions_col: str
    code_digit_match_length: int
    total_queries: int
    ave_time_per_query_ms: float
    unmatched_query_count: int
    mrr: float
    mean_rank: float
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]

    def report_metrics(self) -> str:
        """Pretty print the performance metrics."""
        lines = [
            f"\nSAYT Performance Metrics for column {self.suggestions_col}:",
            f" Code digit match length: {self.code_digit_match_length}",
            f" Total queries: {self.total_queries}",
            f" Average time per query: {self.ave_time_per_query_ms:.2f} ms",
            f" Unmatched query count: {self.unmatched_query_count}",
            f" Mean reciprocal rank: {self.mrr:.4f}",
            f" Mean rank: {self.mean_rank:.2f}",
        ]
        for k, val in sorted(self.precision_at_k.items()):
            lines.append(f" Precision@{k}: {val:.4f}")
        for k, val in sorted(self.recall_at_k.items()):
            lines.append(f" Recall@{k}: {val:.4f}")
        return "\n".join(lines)


def compute_performance_metrics_from_suggestions(  # noqa: PLR0913 pylint: disable = R0913, R0917
    df,
    correct_codes_col: str,
    suggestions_col: str,
    code_length: int,
    ave_time_per_query: float,
    k_values: list[int] | None = None,
    code_digit_match_length: int | None = None,
) -> SAYTPerformanceMetrics:
    """Compute performance metrics from raw suggestion strings.

    Args:
        df: DataFrame containing the queries and suggestions.
        correct_codes_col: Column name containing correct code(s) (string or list).
        suggestions_col: Column name containing the list of suggestion strings.
        code_length: Number of trailing characters to extract as a code.
        k_values: List of k values for which to compute Precision@K and Recall@K.
        ave_time_per_query: Average time taken per query in milliseconds.
        code_digit_match_length: Optional length of the code to match for evaluation.

    Returns:
        SAYTPerformanceMetrics: Computed performance metrics.
    """
    df = df.copy()
    df["_retrieved_codes"] = df.apply(
        get_codes_from_suggestions,
        suggestions_col=suggestions_col,
        code_length=code_length,
        axis=1,
    )

    if code_digit_match_length is not None:
        df[correct_codes_col] = df[correct_codes_col].apply(
            lambda codes: (
                codes[:code_digit_match_length]
                if isinstance(codes, str)
                else list({code[:code_digit_match_length] for code in codes})
            )
        )
        df["_retrieved_codes"] = df["_retrieved_codes"].apply(
            lambda codes: [code[:code_digit_match_length] for code in codes],
        )

    df = add_sayt_metrics_columns(
        df,
        retrieved_codes_col="_retrieved_codes",
        correct_codes_col=correct_codes_col,
        k_values=k_values,
    )

    return summarise_performance_metrics(
        df,
        suggestions_col=suggestions_col,
        code_digit_match_length=(
            code_digit_match_length
            if code_digit_match_length is not None
            else code_length
        ),
        k_values=k_values if k_values is not None else [],
        ave_time_per_query=ave_time_per_query,
    )


def compute_precision_at_k(
    retrieved_codes: list[str], correct_codes: str | list[str], k: int
) -> float:
    """Compute Precision@K for a single query.

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_codes: A single correct code or list of correct codes to match against.
        k: The cutoff rank at which to compute precision.

    Returns:
        float: Precision@K value.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    if isinstance(correct_codes, str):
        correct_codes = [correct_codes]

    top_k_retrieved = retrieved_codes[:k]
    relevant_count = sum(1 for item in top_k_retrieved if item in correct_codes)
    return relevant_count / k


def compute_recall_at_k(
    retrieved_codes: list[str], correct_codes: str | list[str], k: int
) -> float:
    """Compute Recall@K for a single query.

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_codes: A single correct code or list of correct codes to match against.
        k: The cutoff rank at which to compute recall.

    Returns:
        float: Recall@K value (relevant codes in top-k / total correct codes).
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    if isinstance(correct_codes, str):
        correct_codes = [correct_codes]

    top_k_retrieved = retrieved_codes[:k]

    relevant_count = sum(1 for item in top_k_retrieved if item in correct_codes)
    total_correct = len(correct_codes)
    return relevant_count / total_correct if total_correct > 0 else 0.0


def compute_reciprocal_rank(
    retrieved_codes: list[str], correct_codes: str | list[str]
) -> float:
    """Compute Reciprocal Rank for a single query.

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_codes: A single correct code or list of correct codes to match against.

    Returns:
        float: Reciprocal Rank value (1/rank of first match, 0 if no match).
    """
    if isinstance(correct_codes, str):
        correct_codes = [correct_codes]

    for rank, item in enumerate(retrieved_codes, start=1):
        if item in correct_codes:
            return 1 / rank
    return 0.0


def get_rank_of_first_matching_code(
    retrieved_codes: list[str], correct_codes: str | list[str]
) -> int | None:
    """Get the rank of the first retrieved code matching correct code(s).

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_codes: A single correct code or list of correct codes to match against.

    Returns:
        int: Rank of the first matching code, or None if no match found.
    """
    if isinstance(correct_codes, str):
        correct_codes = [correct_codes]

    for rank, item in enumerate(retrieved_codes, start=1):
        if item in correct_codes:
            return int(rank)
    return None


def add_sayt_metrics_columns(
    df,
    retrieved_codes_col: str,
    correct_codes_col: str,
    k_values: list[int] | None = None,
    prefix: str | None = None,
):
    """Add performance metric columns to the DataFrame.

    Args:
        df: DataFrame containing the retrieved codes and correct codes.
        retrieved_codes_col: Column name containing the list of retrieved codes.
        correct_codes_col: Column name containing correct code(s) (string or list).
        k_values: List of k values for which to compute Precision@K and Recall@K.
        prefix: Optional prefix for the new metric columns. Defaults to None.

    Returns:
        pd.DataFrame: Copy of df with metric columns added.
    """
    if prefix is None:
        prefix = ""

    df = df.copy()

    if k_values:
        for k in k_values:
            df[f"{prefix}precision_at_{k}"] = df.apply(
                lambda row, k=k: compute_precision_at_k(
                    row[retrieved_codes_col], row[correct_codes_col], k=k
                ),
                axis=1,
            )
            df[f"{prefix}recall_at_{k}"] = df.apply(
                lambda row, k=k: compute_recall_at_k(
                    row[retrieved_codes_col], row[correct_codes_col], k=k
                ),
                axis=1,
            )

    df[f"{prefix}reciprocal_rank"] = df.apply(
        lambda row: compute_reciprocal_rank(
            row[retrieved_codes_col], row[correct_codes_col]
        ),
        axis=1,
    )
    df[f"{prefix}correct_code_rank"] = df.apply(
        lambda row: get_rank_of_first_matching_code(
            row[retrieved_codes_col], row[correct_codes_col]
        ),
        axis=1,
    )
    return df


def summarise_performance_metrics(  # noqa: PLR0913 pylint: disable = R0913, R0917
    df,
    suggestions_col: str,
    code_digit_match_length: int,
    ave_time_per_query: float,
    k_values: list[int] | None = None,
    prefix: str | None = None,
) -> SAYTPerformanceMetrics:
    """Summarize performance metrics across the DataFrame.

    Args:
        df: DataFrame containing the performance metric columns.
        suggestions_col: Column name containing the retrieved suggestions.
        code_digit_match_length: Length of the code to match for evaluation.
        k_values: List of k values for which Precision@K and Recall@K were computed.
        ave_time_per_query: Average time taken per query in milliseconds.
        prefix: Optional prefix for the metric columns. Defaults to None.

    Returns:
        dict: Summary statistics for each performance metric.
    """
    if prefix is None:
        prefix = ""

    if k_values is None:
        k_values = []

    summary = {
        "code_digit_match_length": code_digit_match_length,
        "suggestions_col": suggestions_col,
        "total_queries": len(df),
        "ave_time_per_query_ms": ave_time_per_query,
        "unmatched_query_count": df[f"{prefix}correct_code_rank"].isna().sum(),
        "mrr": df[f"{prefix}reciprocal_rank"].mean(),
        "mean_rank": df[f"{prefix}correct_code_rank"].mean(),
        "precision_at_k": {k: df[f"{prefix}precision_at_{k}"].mean() for k in k_values},
        "recall_at_k": {k: df[f"{prefix}recall_at_{k}"].mean() for k in k_values},
    }

    return SAYTPerformanceMetrics(**summary)


def build_sayt_metrics_comparison_table(
    df,
    suggestions_cols_to_compare: list[str],
    correct_codes_col: str,
    ave_time_per_query_dict: dict[str, float],
    k_values: list[int] | None = None,
):
    """Build a comparison table of performance metrics across suggestion columns.

    Args:
        df: DataFrame containing the retrieved suggestions and correct codes.
        suggestions_cols_to_compare: List of column names containing
            the retrieved suggestions to compare.
        correct_codes_col: Column name containing correct code(s) (string or list).
        k_values: List of k values for which to compute Precision@K and Recall@K.
        ave_time_per_query_dict: Average time per query (ms) for each suggestion column,
            keyed by the suggestion column name.

    Returns:
        pd.DataFrame: One row per suggestion column with all performance metrics.
    """
    performance_metrics = pd.DataFrame()
    code_length = len(df[correct_codes_col].iloc[0])  # inferred from first row

    for col in suggestions_cols_to_compare:
        performance_metrics_tmp = {
            **compute_performance_metrics_from_suggestions(
                df,
                correct_codes_col=correct_codes_col,
                suggestions_col=col,
                code_length=code_length,
                k_values=k_values if k_values is not None else [],
                ave_time_per_query=ave_time_per_query_dict[col],
            ).__dict__,
        }

        performance_metrics = pd.concat(
            [performance_metrics, pd.DataFrame([performance_metrics_tmp])],
            ignore_index=True,
        )
    return performance_metrics
