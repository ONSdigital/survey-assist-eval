"""Performance metrics functions for SAYT evaluation."""

import math

import pandas as pd
from pydantic import BaseModel

from survey_assist_eval.data_cleaning.code_standard import (
    validate_n_digits_for_code_type,
)
from survey_assist_eval.evaluation.sayt.suggestion_ranking_functions import (
    clean_codes_columns,
    get_codes_from_suggestions,
    get_rank_of_first_matching_code,
    get_ranks_of_correct_codes,
    is_correct_codes_empty,
)


class SAYTPerformanceMetrics(BaseModel):
    """Class to compute performance metrics for SAYT evaluation."""

    suggestions_col: str
    code_digit_match_length: int
    total_queries: int
    queries_with_ground_truth: int
    queries_missing_ground_truth: int
    ave_time_per_query_ms: float
    unmatched_query_count: int
    mrr: float
    mean_last_reciprocal_rank: float
    mean_rank: float
    mean_rank_penalised: float
    median_rank: float
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    ndcg_all_at_k: dict[int, float]

    def report_metrics(self) -> str:
        """Pretty print the performance metrics."""
        lines = [
            f"\nSAYT Performance Metrics for column {self.suggestions_col}",
            "=" * 50,
            "",
            "Configuration",
            f"  Code digit match length: {self.code_digit_match_length}",
            "",
            "Dataset",
            f"  Total queries: {self.total_queries}",
            f"  Queries with ground truth: {self.queries_with_ground_truth}",
            f"  Queries missing ground truth: {self.queries_missing_ground_truth}",
            f"  Unmatched query count: {self.unmatched_query_count}",
            "",
            "Performance",
            f"  Average time per query: {self.ave_time_per_query_ms:.2f} ms",
            "",
            "Ranking Metrics",
            f"  MRR: {self.mrr:.4f}",
            f"  Mean last reciprocal rank: {self.mean_last_reciprocal_rank:.4f}",
            f"  Mean rank: {self.mean_rank:.2f}",
            f"  Mean rank (penalised): {self.mean_rank_penalised:.2f}",
            f"  Median rank: {self.median_rank:.2f}",
        ]
        lines.append("")
        lines.append("Precision")
        for k, val in sorted(self.precision_at_k.items()):
            lines.append(f" Precision@{k}: {val:.4f}")
        lines.append("")
        lines.append("Recall")
        for k, val in sorted(self.recall_at_k.items()):
            lines.append(f" Recall@{k}: {val:.4f}")

        lines.append("")
        lines.append("NDCG (first rank per code)")
        for k, val in sorted(self.ndcg_at_k.items()):
            lines.append(f" NDCG@{k}: {val:.4f}")

        if self.ndcg_all_at_k:
            lines.append("")
            lines.append("NDCG (all relevant ranks)")
            for k, val in sorted(self.ndcg_all_at_k.items()):
                lines.append(f" NDCG_All@{k}: {val:.4f}")
        else:
            lines.append("")
            lines.append(
                "NDCG (all relevant ranks) not available, "
                "please provide a sayt_corpus_df."
            )

        return "\n".join(lines)


def compute_performance_metrics_from_suggestions(  # noqa: PLR0913 pylint: disable = R0913, R0917
    df,
    correct_codes_col: str,
    suggestions_col: str,
    ave_time_per_query: float,
    code_type: str = "sic",
    k_values: list[int] | None = None,
    sayt_corpus_df: pd.DataFrame | None = None,
    code_digit_match_length: int | None = None,
) -> SAYTPerformanceMetrics:
    """Compute performance metrics from raw suggestion strings.

    Args:
        df: DataFrame containing the queries and suggestions.
        correct_codes_col: Column name containing correct code(s).
        suggestions_col: Column name containing the list of suggestion strings.
        code_type: Type of code ('sic' or 'soc'). Defaults to 'sic'.
        k_values: List of k values for which to compute Precision@K and Recall@K.
        ave_time_per_query: Average time taken per query in milliseconds.
        sayt_corpus_df: Optional DataFrame containing the SAYT corpus.
            Required for computing NDCG considering all relevant ranks.
        code_digit_match_length: Optional length of the code to match for evaluation.

    Returns:
        SAYTPerformanceMetrics: Computed performance metrics.
    """
    df = df.copy()

    df["_retrieved_codes"] = df.apply(
        get_codes_from_suggestions,
        suggestions_col=suggestions_col,
        code_type=code_type,
        axis=1,
    )

    df = clean_codes_columns(
        df,
        code_digit_match_length=code_digit_match_length,
        code_type=code_type,
        correct_codes_col=correct_codes_col,
        retrieved_codes_col="_retrieved_codes",
    )

    if sayt_corpus_df is not None:
        sayt_corpus_df = clean_codes_columns(
            sayt_corpus_df,
            code_digit_match_length=code_digit_match_length,
            code_type=code_type,
            retrieved_codes_col="code",
        )

        total_relevant_ranks_dict = get_sayt_corpus_code_counts(
            sayt_corpus_df.explode("code_clean"), code_col="code_clean"
        )
    else:
        total_relevant_ranks_dict = None

    df = add_sayt_metrics_columns(
        df,
        retrieved_codes_col="_retrieved_codes_clean",
        correct_codes_col=f"{correct_codes_col}_clean",
        k_values=k_values,
        total_relevant_ranks_dict=total_relevant_ranks_dict,
    )

    return summarise_performance_metrics(
        df,
        suggestions_col=suggestions_col,
        correct_codes_col=correct_codes_col,
        code_digit_match_length=validate_n_digits_for_code_type(
            code_digit_match_length, code_type
        ),
        k_values=k_values if k_values is not None else [],
        ave_time_per_query=ave_time_per_query,
    )


def get_sayt_corpus_code_counts(
    sayt_corpus_df: pd.DataFrame, code_col: str
) -> dict[str, int]:
    """Compute the total relevant ranks for each code in the SAYT corpus.

    Args:
        sayt_corpus_df: DataFrame containing the SAYT corpus with a column specified by `code_col`.
        code_col: Name of the column containing the codes in the SAYT corpus.

    Returns:
        dict[str, int]: Dictionary mapping each code to its total count in the corpus.
    """
    code_counts = sayt_corpus_df[code_col].value_counts().to_dict()
    return code_counts


def compute_precision_at_k(
    retrieved_codes: list[str], correct_codes: str | list[str] | set[str] | None, k: int
) -> float:
    """Compute Precision@K for a single query.

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_codes: A single correct code or set of correct codes to match against.
        k: The cutoff rank at which to compute precision.

    Returns:
        float: Precision@K value.
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    if correct_codes is None or is_correct_codes_empty(correct_codes):
        return 0.0

    if isinstance(correct_codes, str):
        correct_codes = {correct_codes}

    top_k_retrieved = retrieved_codes[:k]
    relevant_count = sum(1 for item in top_k_retrieved if item in correct_codes)
    return relevant_count / k


def compute_recall_at_k(
    retrieved_codes: list[str], correct_codes: str | list[str] | set[str] | None, k: int
) -> float:
    """Compute Recall@K for a single query.

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_codes: A single correct code or set of correct codes to match against.
        k: The cutoff rank at which to compute recall.

    Returns:
        float: Recall@K value (relevant codes in top-k / total correct codes).
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    if correct_codes is None or is_correct_codes_empty(correct_codes):
        return 0.0

    if isinstance(correct_codes, str):
        correct_codes = {correct_codes}

    top_k_retrieved = retrieved_codes[:k]

    relevant_count = sum(1 for item in correct_codes if item in top_k_retrieved)
    total_correct = len(correct_codes)
    return relevant_count / total_correct if total_correct > 0 else 0.0


def compute_reciprocal_rank(
    retrieved_codes: list[str], correct_codes: str | list[str] | set[str] | None
) -> float:
    """Compute Reciprocal Rank for a single query.

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_codes: A single correct code, list of correct codes, or set of
            correct codes to match against.

    Returns:
        float: Reciprocal Rank value (1/rank of first match, 0 if no match).
    """
    if correct_codes is None or is_correct_codes_empty(correct_codes):
        return 0.0

    if isinstance(correct_codes, str):
        correct_codes = {correct_codes}

    for rank, item in enumerate(retrieved_codes, start=1):
        if item in correct_codes:
            return 1 / rank
    return 0.0


def compute_median_with_none_as_inf(
    values: list[float | None] | pd.Series,
) -> float:
    """Compute the median of a list of values, treating None as infinity.

    Args:
        values: Numeric values that may contain missing values.

    Returns:
        The median value, with None treated as infinity.
    """
    return float(pd.Series(values, dtype="float64").fillna(float("inf")).median())


def compute_reciprocal_rank_of_final_correct_code(
    correct_codes_ranks_dict: dict[str, list[int]],
) -> float:
    """Compute the reciprocal rank of the final distinct correct code retrieved.

    Uses the first retrieved rank for each correct code, then takes the reciprocal
    of the latest of those first-hit ranks. Returns 0.0 if any correct code has no
    retrieved rank.

    Args:
        correct_codes_ranks_dict: Mapping of each correct code to its retrieved ranks.

    Returns:
        float: Reciprocal rank of the final correct code, or 0.0 if any code is
            missing from the retrieved ranks.
    """
    if not correct_codes_ranks_dict:
        return 0.0

    first_ranks = []

    for ranks in correct_codes_ranks_dict.values():
        if ranks == []:
            return 0.0

        first_ranks.append(min(ranks))

    return 1 / max(first_ranks) if first_ranks and max(first_ranks) > 0 else 0.0


def get_total_relevant_ranks(
    correct_codes: list[str],
    total_relevant_ranks_dict: dict[str, int] | None,
) -> int:
    """Retrieve the total relevant ranks for a given set of correct codes.

    Args:
        correct_codes: List of correct codes.
        total_relevant_ranks_dict: Mapping from correct code to total relevant ranks.

    Returns:
        int: Total relevant ranks for the given correct codes, or 0 if not found.
    """
    if total_relevant_ranks_dict is None:
        return 0
    return sum(total_relevant_ranks_dict.get(code, 0) for code in correct_codes)


def compute_normalized_discounted_cumulative_gain_at_k(
    correct_codes_ranks_dict: dict[str, list[int]],
    k: int,
    include_all_relevant_ranks: bool = False,
    total_relevant_ranks: int | None = None,
) -> float:
    """Compute NDCG, optionally counting every relevant rank for each code.

    Args:
        correct_codes_ranks_dict: Mapping of every correct code to its retrieved ranks.
            An empty rank list means that code was not retrieved.
        k: The cutoff rank for computing NDCG.
        include_all_relevant_ranks: Whether to count every matching rank instead of
            only the first rank for each correct code.
        total_relevant_ranks: Total relevant ranks for the ideal ranking. Required
            when include_all_relevant_ranks is True.

    Returns:
        float: NDCG value (0.0 if no correct codes are retrieved).
    """
    if not correct_codes_ranks_dict:
        return 0.0

    if include_all_relevant_ranks:
        if total_relevant_ranks is None:
            raise ValueError(
                "total_relevant_ranks must be provided when "
                "include_all_relevant_ranks is True."
            )
        relevant_ranks = [
            rank
            for ranks in correct_codes_ranks_dict.values()
            for rank in ranks
            if rank <= k
        ]
        ideal_ranks = range(1, min(total_relevant_ranks, k) + 1)
    else:
        relevant_ranks = [
            min(ranks)
            for ranks in correct_codes_ranks_dict.values()
            if ranks and min(ranks) <= k
        ]
        ideal_ranks = range(1, min(len(correct_codes_ranks_dict), k) + 1)

    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
    idcg = sum(1 / math.log2(rank + 1) for rank in ideal_ranks)
    return dcg / idcg if idcg > 0 else 0.0


def add_sayt_metrics_columns(  # noqa: PLR0913 pylint: disable = R0913, R0917
    df,
    retrieved_codes_col: str,
    correct_codes_col: str,
    k_values: list[int] | None = None,
    total_relevant_ranks_dict: dict[str, int] | None = None,
    prefix: str | None = None,
):
    """Add performance metric columns to the DataFrame.

    Args:
        df: DataFrame containing the retrieved codes and correct codes.
        retrieved_codes_col: Column name containing the list of retrieved codes.
        correct_codes_col: Column name containing correct code(s).
        k_values: List of k values for which to compute Precision@K and Recall@K.
        total_relevant_ranks_dict: Optional dictionary mapping correct codes to their total
            relevant ranks. Required for computing NDCG considering all relevant ranks.
        prefix: Optional prefix for the new metric columns. Defaults to None.

    Returns:
        pd.DataFrame: Copy of df with metric columns added.
    """
    if prefix is None:
        prefix = ""

    df = df.copy()

    df[f"{prefix}ranks_of_correct_codes"] = df.apply(
        lambda row: get_ranks_of_correct_codes(
            row[retrieved_codes_col], row[correct_codes_col]
        ),
        axis=1,
    )

    if total_relevant_ranks_dict is not None:
        df[f"{prefix}total_relevant_ranks"] = df.apply(
            lambda row: get_total_relevant_ranks(
                row[correct_codes_col], total_relevant_ranks_dict
            ),
            axis=1,
        )

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
            df[f"{prefix}ndcg_at_{k}"] = df.apply(
                lambda row, k=k: compute_normalized_discounted_cumulative_gain_at_k(
                    row[f"{prefix}ranks_of_correct_codes"], k=k
                ),
                axis=1,
            )

            if total_relevant_ranks_dict is not None:
                df[f"{prefix}ndcg_all_at_{k}"] = df.apply(
                    lambda row, k=k: compute_normalized_discounted_cumulative_gain_at_k(
                        row[f"{prefix}ranks_of_correct_codes"],
                        k=k,
                        include_all_relevant_ranks=True,
                        total_relevant_ranks=row[f"{prefix}total_relevant_ranks"],
                    ),
                    axis=1,
                )

    df[f"{prefix}reciprocal_rank"] = df.apply(
        lambda row: compute_reciprocal_rank(
            row[retrieved_codes_col], row[correct_codes_col]
        ),
        axis=1,
    )

    df[f"{prefix}last_reciprocal_rank"] = df.apply(
        lambda row: compute_reciprocal_rank_of_final_correct_code(
            row[f"{prefix}ranks_of_correct_codes"]
        ),
        axis=1,
    )
    df[f"{prefix}correct_code_rank"] = df.apply(
        lambda row: get_rank_of_first_matching_code(
            row[retrieved_codes_col], row[correct_codes_col]
        ),
        axis=1,
    )

    df[f"{prefix}correct_code_rank_penalised"] = df.apply(
        lambda row: get_rank_of_first_matching_code(
            row[retrieved_codes_col], row[correct_codes_col], penalise_if_not_found=True
        ),
        axis=1,
    )

    return df


def summarise_performance_metrics(  # noqa: PLR0913 pylint: disable = R0913, R0917
    df,
    suggestions_col: str,
    correct_codes_col: str,
    code_digit_match_length: int | str,
    ave_time_per_query: float,
    k_values: list[int] | None = None,
    prefix: str | None = None,
) -> SAYTPerformanceMetrics:
    """Summarize performance metrics across the DataFrame.

    Args:
        df: DataFrame containing the performance metric columns.
        suggestions_col: Column name containing the retrieved suggestions.
        correct_codes_col: Column name containing correct code(s).
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

    if correct_codes_col not in df.columns:
        raise ValueError(f"Column '{correct_codes_col}' not found in DataFrame.")

    total_queries = len(df)

    no_ground_truth = df[correct_codes_col].apply(is_correct_codes_empty)

    df = df.loc[~no_ground_truth].copy()

    summary = {
        "code_digit_match_length": code_digit_match_length,
        "suggestions_col": suggestions_col,
        "total_queries": total_queries,
        "queries_with_ground_truth": len(df),
        "queries_missing_ground_truth": total_queries - len(df),
        "ave_time_per_query_ms": ave_time_per_query,
        "unmatched_query_count": df[f"{prefix}correct_code_rank"].isna().sum(),
        "mrr": df[f"{prefix}reciprocal_rank"].mean(),
        "mean_last_reciprocal_rank": df[f"{prefix}last_reciprocal_rank"].mean(),
        "mean_rank": df[f"{prefix}correct_code_rank"].mean(),
        "mean_rank_penalised": df[f"{prefix}correct_code_rank_penalised"].mean(),
        "median_rank": compute_median_with_none_as_inf(
            df[f"{prefix}correct_code_rank"]
        ),
        "precision_at_k": {k: df[f"{prefix}precision_at_{k}"].mean() for k in k_values},
        "recall_at_k": {k: df[f"{prefix}recall_at_{k}"].mean() for k in k_values},
        "ndcg_at_k": {k: df[f"{prefix}ndcg_at_{k}"].mean() for k in k_values},
        "ndcg_all_at_k": {
            k: df[f"{prefix}ndcg_all_at_{k}"].mean()
            for k in k_values
            if f"{prefix}ndcg_all_at_{k}" in df.columns
        },
    }

    return SAYTPerformanceMetrics(**summary)


def build_sayt_metrics_comparison_table(  # noqa: PLR0913 pylint: disable = R0913, R0917
    df,
    suggestions_cols_to_compare: list[str],
    correct_codes_col: str,
    ave_time_per_query_dict: dict[str, float],
    code_type: str = "sic",
    code_digit_match_length: int | None = None,
    sayt_corpus_df: pd.DataFrame | None = None,
    k_values: list[int] | None = None,
):
    """Build a comparison table of performance metrics across suggestion columns.

    Args:
        df: DataFrame containing the retrieved suggestions and correct codes.
        suggestions_cols_to_compare: List of column names containing
            the retrieved suggestions to compare.
        correct_codes_col: Column name containing correct code(s).
        ave_time_per_query_dict: Average time per query (ms) for each suggestion column,
            keyed by the suggestion column name.
        code_type: Type of code ('sic' or 'soc'). Defaults to 'sic'. Defaults to 'sic'.
        code_digit_match_length: Length of the code digit match to consider (default is None).
        sayt_corpus_df: DataFrame containing the full SAYT corpus (default is None).
        k_values: List of k values for which to compute Precision@K and Recall@K.

    Returns:
        pd.DataFrame: One row per suggestion column with all performance metrics.
    """
    performance_metrics = pd.DataFrame()

    for col in suggestions_cols_to_compare:
        performance_metrics_tmp = {
            **compute_performance_metrics_from_suggestions(
                df,
                correct_codes_col=correct_codes_col,
                suggestions_col=col,
                code_type=code_type,
                k_values=k_values if k_values is not None else [],
                sayt_corpus_df=sayt_corpus_df,
                ave_time_per_query=ave_time_per_query_dict[col],
                code_digit_match_length=code_digit_match_length,
            ).__dict__,
        }

        performance_metrics = pd.concat(
            [performance_metrics, pd.DataFrame([performance_metrics_tmp])],
            ignore_index=True,
        )
    return performance_metrics
