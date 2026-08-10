"""Performance metrics functions for SAYT evaluation."""

import pandas as pd
from pydantic import BaseModel

from notebooks.sayt.sayt_utils import get_codes_from_suggestions


class SAYTPerformanceMetrics(BaseModel):
    """Class to compute performance metrics for SAYT evaluation."""

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
            "\nSAYT Performance Metrics:",
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
    correct_code_col: str,
    suggestions_col: str,
    code_length: int,
    k_values: list[int],
    ave_time_per_query: float,
) -> SAYTPerformanceMetrics:
    """Compute performance metrics from raw suggestion strings.

    Args:
        df: DataFrame containing the queries and suggestions.
        correct_code_col: Column name containing the correct code.
        suggestions_col: Column name containing the list of suggestion strings.
        code_length: Number of trailing characters to extract as a code.
        k_values: List of k values for which to compute Precision@K and Recall@K.
        ave_time_per_query: Average time taken per query in milliseconds.

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
    df = add_sayt_metrics_columns(
        df,
        retrieved_codes_col="_retrieved_codes",
        correct_code_col=correct_code_col,
        k_values=k_values,
    )

    return summarise_performance_metrics(
        df,
        k_values=k_values,
        ave_time_per_query=ave_time_per_query,
    )


def compute_precision_at_k(
    retrieved_codes: list[str], correct_code: str, k: int
) -> float:
    """Compute Precision@K for a single query.

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_code: The correct code for the query.
        k: The cutoff rank at which to compute precision.

    Returns:
        float: Precision@K value.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    top_k_retrieved = retrieved_codes[:k]
    relevant_count = sum(1 for item in top_k_retrieved if item == correct_code)
    return relevant_count / k


def compute_recall_at_k(retrieved_codes: list[str], correct_code: str, k: int) -> float:
    """Compute Recall@K for a single query.

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_code: The correct code for the query.
        k: The cutoff rank at which to compute recall.

    Returns:
        float: Recall@K value.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    top_k_retrieved = retrieved_codes[:k]
    return 1.0 if correct_code in top_k_retrieved else 0.0


def compute_reciprocal_rank(retrieved_codes: list[str], correct_code: str) -> float:
    """Compute Reciprocal Rank for a single query.

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_code: The correct code for the query.

    Returns:
        float: Reciprocal Rank value.
    """
    for rank, item in enumerate(retrieved_codes, start=1):
        if item == correct_code:
            return 1 / rank
    return 0.0


def get_rank_of_correct_code(
    retrieved_codes: list[str], correct_code: str
) -> float | None:
    """Get the rank of the correct code in the retrieved list for a single query.

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_code: The correct code for the query.

    Returns:
        float: Rank of the correct code, or None if not found.
    """
    for rank, item in enumerate(retrieved_codes, start=1):
        if item == correct_code:
            return float(rank)
    return None


def add_sayt_metrics_columns(
    df,
    retrieved_codes_col: str,
    correct_code_col: str,
    k_values: list[int],
):
    """Add performance metric columns to the DataFrame.

    Args:
        df: DataFrame containing the retrieved codes and correct codes.
        retrieved_codes_col: Column name containing the list of retrieved codes.
        correct_code_col: Column name containing the correct code.
        k_values: List of k values for which to compute Precision@K and Recall@K.

    Returns:
        pd.DataFrame: Copy of df with metric columns added.
    """
    df = df.copy()
    for k in k_values:
        df[f"precision_at_{k}"] = df.apply(
            lambda row, k=k: compute_precision_at_k(
                row[retrieved_codes_col], row[correct_code_col], k=k
            ),
            axis=1,
        )
        df[f"recall_at_{k}"] = df.apply(
            lambda row, k=k: compute_recall_at_k(
                row[retrieved_codes_col], row[correct_code_col], k=k
            ),
            axis=1,
        )
    df["reciprocal_rank"] = df.apply(
        lambda row: compute_reciprocal_rank(
            row[retrieved_codes_col], row[correct_code_col]
        ),
        axis=1,
    )
    df["correct_code_rank"] = df.apply(
        lambda row: get_rank_of_correct_code(
            row[retrieved_codes_col], row[correct_code_col]
        ),
        axis=1,
    )
    return df


def summarise_performance_metrics(
    df, k_values: list[int], ave_time_per_query: float
) -> SAYTPerformanceMetrics:
    """Summarize performance metrics across the DataFrame.

    Args:
        df: DataFrame containing the performance metric columns.
        k_values: List of k values for which Precision@K and Recall@K were computed.
        ave_time_per_query: Average time taken per query in milliseconds.

    Returns:
        dict: Summary statistics for each performance metric.
    """
    summary = {
        "total_queries": len(df),
        "ave_time_per_query_ms": ave_time_per_query,
        "unmatched_query_count": df["correct_code_rank"].isna().sum(),
        "mrr": df["reciprocal_rank"].mean(),
        "mean_rank": df["correct_code_rank"].mean(),
        "precision_at_k": {k: df[f"precision_at_{k}"].mean() for k in k_values},
        "recall_at_k": {k: df[f"recall_at_{k}"].mean() for k in k_values},
    }

    return SAYTPerformanceMetrics(**summary)


def build_sayt_metrics_comparison_table(
    df,
    suggestions_cols_to_compare: list[str],
    correct_code_col: str,
    k_values: list[int],
    ave_time_per_query_list: list[float],
):
    """Build a comparison table of performance metrics across suggestion columns.

    Args:
        df: DataFrame containing the retrieved suggestions and correct codes.
        suggestions_cols_to_compare: List of column names containing
            the retrieved suggestions to compare.
        correct_code_col: Column name containing the correct code.
        k_values: List of k values for which to compute Precision@K and Recall@K.
        ave_time_per_query_list: Average time per query (ms) for each suggestion column,
            in the same order as suggestions_cols_to_compare.

    Returns:
        pd.DataFrame: One row per suggestion column with all performance metrics.
    """
    performance_metrics = pd.DataFrame()
    code_length = len(df[correct_code_col].iloc[0])  # inferred from first row

    for i, col in enumerate(suggestions_cols_to_compare):
        performance_metrics_tmp = {
            "model": col.removeprefix("suggestions_"),
            **compute_performance_metrics_from_suggestions(
                df,
                correct_code_col=correct_code_col,
                suggestions_col=col,
                code_length=code_length,
                k_values=k_values,
                ave_time_per_query=ave_time_per_query_list[i],
            ).__dict__,
        }

        performance_metrics = pd.concat(
            [performance_metrics, pd.DataFrame([performance_metrics_tmp])],
            ignore_index=True,
        )
    return performance_metrics
