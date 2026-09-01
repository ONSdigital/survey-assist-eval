"""Tests for SAYT performance metric helper functions."""

# pylint: disable=redefined-outer-name, too-many-lines


import math

import pandas as pd
import pytest
from pydantic import ValidationError

from survey_assist_eval.evaluation.sayt.performance_metrics_functions import (
    SAYTPerformanceMetrics,
    add_sayt_metrics_columns,
    build_sayt_metrics_comparison_table,
    compute_performance_metrics_from_suggestions,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    get_rank_of_first_matching_code,
    summarise_performance_metrics,
)


@pytest.fixture
def sayt_metrics_input_df():
    """Representative SAYT retrieval results for DataFrame-based metric tests."""
    return pd.DataFrame(
        {
            "retrieved_codes": [
                ["1111", "2222", "3333"],
                ["3333", "1111"],
                [],
            ],
            "correct_code": ["1111", "4444", "5555"],
        }
    )


# ============================================================================
# Test compute_precision_at_k function
# ============================================================================


@pytest.mark.parametrize("k", [0, -1])
def test_compute_precision_at_k_raises_for_non_positive_k(k):
    """Precision@k should reject non-positive cutoffs."""
    with pytest.raises(ValueError, match="k must be a positive integer"):
        compute_precision_at_k(["1234", "5678"], "1234", k)


def test_compute_precision_at_k_returns_fraction_of_top_k_matches():
    """Precision@k should count relevant results within the cutoff."""
    precision = compute_precision_at_k(["1111", "2222", "1111"], "1111", 2)

    assert precision == pytest.approx(0.5), (
        "Expected Precision@2 to be 0.5 when one of two top-ranked results matches "
        "the correct code."
    )


def test_compute_precision_at_k_counts_duplicate_correct_codes_in_top_k():
    """Precision@k counts all matching entries present in the top-k list."""
    precision = compute_precision_at_k(["1111", "1111", "2222"], "1111", 2)

    assert precision == pytest.approx(1.0), (
        "Expected Precision@2 to be 1.0 when both top-ranked results match the "
        "correct code."
    )


def test_compute_precision_at_k_returns_zero_for_empty_retrieved_codes():
    """Precision@k should be zero when no results are retrieved."""
    precision = compute_precision_at_k([], "1111", 3)

    assert precision == pytest.approx(
        0.0
    ), "Expected Precision@3 to be 0.0 when the retrieved code list is empty."


def test_compute_precision_at_k_uses_requested_k_when_fewer_results_returned():
    """Precision@k keeps k as the denominator even for short result lists."""
    precision = compute_precision_at_k(["1111"], "1111", 3)

    assert precision == pytest.approx(1 / 3), (
        "Expected Precision@3 to divide by k even when fewer than k results are "
        "returned."
    )


def test_compute_precision_at_k_with_list_of_correct_codes():
    """Precision@k should count matches against any code in the list."""
    precision = compute_precision_at_k(["1111", "2222", "3333"], ["2222", "4444"], 2)

    assert precision == pytest.approx(
        0.5
    ), "Expected Precision@2 to count one match (2222) from the list in top 2."


def test_compute_precision_at_k_with_multiple_matches_in_list():
    """Precision@k should count all matches when list codes appear in top-k."""
    precision = compute_precision_at_k(["1111", "2222", "3333"], ["1111", "2222"], 2)

    assert precision == pytest.approx(
        1.0
    ), "Expected Precision@2 to count two matches from the list in top 2."


def test_compute_precision_at_k_with_empty_correct_codes_list():
    """Precision@k should return 0 when correct_codes list is empty."""
    precision = compute_precision_at_k(["1111", "2222", "3333"], [], 3)

    assert precision == pytest.approx(
        0.0
    ), "Expected Precision@3 to be 0.0 when no correct codes to match."


# ============================================================================
# Test compute_recall_at_k function
# ============================================================================


@pytest.mark.parametrize("k", [0, -1])
def test_compute_recall_at_k_raises_for_non_positive_k(k):
    """Recall@k should reject non-positive cutoffs."""
    with pytest.raises(ValueError, match="k must be a positive integer"):
        compute_recall_at_k(["1234", "5678"], "1234", k)


def test_compute_recall_at_k_returns_one_when_correct_code_in_top_k():
    """Recall@k should be 1 when the correct code is retrieved within k."""
    recall = compute_recall_at_k(["1111", "2222", "3333"], "2222", 2)

    assert recall == pytest.approx(1.0), (
        "Expected Recall@2 to be 1.0 when the correct code appears within the top "
        "two results."
    )


def test_compute_recall_at_k_returns_zero_when_correct_code_not_in_top_k():
    """Recall@k should be 0 when the correct code falls outside the cutoff."""
    recall = compute_recall_at_k(["1111", "2222", "3333"], "3333", 2)

    assert recall == pytest.approx(
        0.0
    ), "Expected Recall@2 to be 0.0 when the correct code falls outside the cutoff."


def test_compute_recall_at_k_returns_zero_for_empty_retrieved_codes():
    """Recall@k should be zero when no results are retrieved."""
    recall = compute_recall_at_k([], "1111", 3)

    assert recall == pytest.approx(
        0.0
    ), "Expected Recall@3 to be 0.0 when the retrieved code list is empty."


def test_compute_recall_at_k_handles_k_larger_than_retrieved_results():
    """Recall@k should still find a match when k exceeds the result count."""
    recall = compute_recall_at_k(["1111"], "1111", 3)

    assert recall == pytest.approx(1.0), (
        "Expected Recall@3 to be 1.0 (1 found / 1 total correct) when the correct "
        "code is present even if fewer than k results are returned."
    )


def test_compute_recall_at_k_with_list_of_correct_codes():
    """Recall@k should be relevant_found / total_correct using a list."""
    recall = compute_recall_at_k(["1111", "2222", "3333"], ["2222", "4444", "5555"], 2)

    assert recall == pytest.approx(
        1 / 3
    ), "Expected Recall@2 to be 1/3 (1 found: 2222 / 3 total correct codes)."


def test_compute_recall_at_k_with_multiple_matches_in_list():
    """Recall@k should count all matching codes from the list in top-k."""
    recall = compute_recall_at_k(["1111", "2222", "3333"], ["1111", "2222", "4444"], 3)

    assert recall == pytest.approx(
        2 / 3
    ), "Expected Recall@3 to be 2/3 (2 found: 1111, 2222 / 3 total correct codes)."


def test_compute_recall_at_k_with_empty_correct_codes_list():
    """Recall@k should return 0 when correct_codes list is empty."""
    recall = compute_recall_at_k(["1111", "2222", "3333"], [], 3)

    assert recall == pytest.approx(
        0.0
    ), "Expected Recall@3 to be 0.0 when no correct codes to recall."


def test_compute_recall_at_k_with_no_matches_in_top_k():
    """Recall@k should be 0 when none of the correct codes appear in top-k."""
    recall = compute_recall_at_k(["1111", "2222"], ["3333", "4444"], 2)

    assert recall == pytest.approx(
        0.0
    ), "Expected Recall@2 to be 0.0 (0 found / 2 total correct codes)."


# ============================================================================
# Test compute_reciprocal_rank function
# ============================================================================


def test_compute_reciprocal_rank_returns_inverse_of_first_matching_rank():
    """Reciprocal rank should use the first matching position."""
    reciprocal_rank = compute_reciprocal_rank(["1111", "2222", "1111"], "1111")

    assert reciprocal_rank == pytest.approx(1.0), (
        "Expected reciprocal rank to use the first matching result when duplicates "
        "exist later in the list."
    )


def test_compute_reciprocal_rank_returns_zero_when_code_not_found():
    """Reciprocal rank should be zero when there is no match."""
    reciprocal_rank = compute_reciprocal_rank(["1111", "2222", "3333"], "4444")

    assert reciprocal_rank == pytest.approx(0.0), (
        "Expected reciprocal rank to be 0.0 when the correct code is absent from "
        "the retrieved list."
    )


def test_compute_reciprocal_rank_returns_inverse_for_match_beyond_first_position():
    """Reciprocal rank should use the first matching position even when later."""
    reciprocal_rank = compute_reciprocal_rank(["1111", "2222", "3333"], "3333")

    assert reciprocal_rank == pytest.approx(1 / 3), (
        "Expected reciprocal rank to equal 1/3 when the correct code is first found "
        "at rank 3."
    )


def test_compute_reciprocal_rank_with_list_of_correct_codes():
    """Reciprocal rank should find first match in list."""
    reciprocal_rank = compute_reciprocal_rank(
        ["1111", "2222", "3333"], ["3333", "4444"]
    )

    assert reciprocal_rank == pytest.approx(1 / 3), (
        "Expected reciprocal rank to equal 1/3 when first matching code from list "
        "is at rank 3."
    )


def test_compute_reciprocal_rank_finds_earliest_in_list():
    """Reciprocal rank should return earliest matching position from list."""
    reciprocal_rank = compute_reciprocal_rank(
        ["1111", "2222", "3333", "4444"], ["3333", "2222"]
    )

    assert reciprocal_rank == pytest.approx(1 / 2), (
        "Expected reciprocal rank to equal 1/2 when earliest match from list "
        "is 2222 at rank 2."
    )


def test_compute_reciprocal_rank_with_empty_correct_codes_list():
    """Reciprocal rank should return 0 when correct_codes list is empty."""
    reciprocal_rank = compute_reciprocal_rank(["1111", "2222", "3333"], [])

    assert reciprocal_rank == pytest.approx(
        0.0
    ), "Expected reciprocal rank to be 0.0 when no correct codes in list."


# ============================================================================
# Test get_rank_of_first_matching_code function
# ============================================================================


def test_get_rank_of_first_matching_code_returns_first_matching_rank():
    """Rank should report the first position containing the correct code."""
    rank = get_rank_of_first_matching_code(["1111", "2222", "1111"], "1111")

    assert rank == pytest.approx(1.0), (
        "Expected rank to report the first matching position when duplicates appear "
        "later in the list."
    )


def test_get_rank_of_first_matching_code_returns_zero_when_code_not_found():
    """Rank should be None when the correct code is absent."""
    rank = get_rank_of_first_matching_code(["1111", "2222", "3333"], "4444")

    assert rank is None, (
        "Expected rank to be None when the correct code is absent from the retrieved "
        "list."
    )


def test_get_rank_of_first_matching_code_returns_rank_beyond_first_position():
    """Rank should reflect the first matching position when it is not first."""
    rank = get_rank_of_first_matching_code(["1111", "2222", "3333"], "3333")

    assert rank == pytest.approx(3.0), (
        "Expected rank to equal 3.0 when the correct code is first found in the "
        "third position."
    )


def test_get_rank_of_first_matching_code_works_with_list_of_correct_codes():
    """Rank should find first match when given a list of correct codes."""
    rank = get_rank_of_first_matching_code(["1111", "2222", "3333"], ["3333", "4444"])

    assert rank == 3, (
        "Expected rank to equal 3 when the first code in the list matches at "
        "position 3."
    )


def test_get_rank_of_first_matching_code_finds_first_match_in_list():
    """Rank should return the earliest matching position from a list of codes."""
    rank = get_rank_of_first_matching_code(
        ["1111", "2222", "3333", "4444"], ["3333", "2222"]
    )

    assert rank == 2, (
        "Expected rank to equal 2 when the earliest match from the list is at "
        "position 2."
    )


def test_get_rank_of_first_matching_code_returns_none_when_list_has_no_match():
    """Rank should be None when no codes in the list are found."""
    rank = get_rank_of_first_matching_code(["1111", "2222", "3333"], ["4444", "5555"])

    assert rank is None, (
        "Expected rank to be None when none of the codes in the list match the "
        "retrieved codes."
    )


def test_get_rank_of_first_matching_code_handles_empty_list_of_correct_codes():
    """Rank should be None when given an empty list of correct codes."""
    rank = get_rank_of_first_matching_code(["1111", "2222", "3333"], [])

    assert (
        rank is None
    ), "Expected rank to be None when the list of correct codes is empty."


def test_get_rank_of_first_matching_code_single_code_in_list():
    """Rank should work with a single-item list equivalently to a string."""
    rank_from_list = get_rank_of_first_matching_code(["1111", "2222", "3333"], ["2222"])
    rank_from_string = get_rank_of_first_matching_code(["1111", "2222", "3333"], "2222")

    assert (
        rank_from_list == rank_from_string == 2
    ), "Expected single-item list and string to produce the same rank."


# ============================================================================
# Test add_sayt_metrics_columns function
# ============================================================================


def test_add_sayt_metrics_columns_does_not_mutate_input_dataframe(
    sayt_metrics_input_df,
):
    """The helper should add metric columns to a copy rather than the input."""
    original_df = sayt_metrics_input_df.copy(deep=True)

    add_sayt_metrics_columns(
        sayt_metrics_input_df,
        retrieved_codes_col="retrieved_codes",
        correct_codes_col="correct_code",
        k_values=[1, 2],
    )

    assert sayt_metrics_input_df.equals(
        original_df
    ), "Expected add_sayt_metrics_columns to leave the input DataFrame unchanged."


def test_add_sayt_metrics_columns_adds_precision_and_recall_columns(
    sayt_metrics_input_df,
):
    """The helper should create one precision and recall column per requested k."""
    result_df = add_sayt_metrics_columns(
        sayt_metrics_input_df,
        retrieved_codes_col="retrieved_codes",
        correct_codes_col="correct_code",
        k_values=[1, 2],
    )

    assert ["precision_at_1", "precision_at_2", "recall_at_1", "recall_at_2"] == [
        "precision_at_1",
        "precision_at_2",
        "recall_at_1",
        "recall_at_2",
    ], "Sanity check failed for expected metric column names."
    assert (
        "precision_at_1" in result_df.columns
    ), "Expected a precision_at_1 column to be added for k=1."
    assert (
        "precision_at_2" in result_df.columns
    ), "Expected a precision_at_2 column to be added for k=2."
    assert (
        "recall_at_1" in result_df.columns
    ), "Expected a recall_at_1 column to be added for k=1."
    assert (
        "recall_at_2" in result_df.columns
    ), "Expected a recall_at_2 column to be added for k=2."
    assert result_df["precision_at_1"].tolist() == pytest.approx(
        [1.0, 0.0, 0.0]
    ), "Expected Precision@1 to be computed independently for each row."
    assert result_df["precision_at_2"].tolist() == pytest.approx(
        [0.5, 0.0, 0.0]
    ), "Expected Precision@2 to divide row-wise matches in the top two results by 2."
    assert result_df["recall_at_1"].tolist() == pytest.approx(
        [1.0, 0.0, 0.0]
    ), "Expected Recall@1 to indicate whether each correct code appears in rank 1."
    assert result_df["recall_at_2"].tolist() == pytest.approx([1.0, 0.0, 0.0]), (
        "Expected Recall@2 to indicate whether each correct code appears within the "
        "top two results."
    )


def test_add_sayt_metrics_columns_adds_rank_based_summary_columns(
    sayt_metrics_input_df,
):
    """The helper should add reciprocal-rank and rank columns for each row."""
    result_df = add_sayt_metrics_columns(
        sayt_metrics_input_df,
        retrieved_codes_col="retrieved_codes",
        correct_codes_col="correct_code",
        k_values=[1],
    )

    assert result_df["reciprocal_rank"].tolist() == pytest.approx([1.0, 0.0, 0.0]), (
        "Expected reciprocal_rank values to equal the reciprocal rank "
        "of the first matching code or 0.0 when there is no match."
    )
    assert result_df["correct_code_rank"].iloc[0] == pytest.approx(1.0)
    assert result_df["correct_code_rank"].isna().tolist() == [
        False,
        True,
        True,
    ], "Expected correct_code_rank to be NaN when the correct code is absent."


def test_add_sayt_metrics_columns_handles_empty_k_values(sayt_metrics_input_df):
    """The helper should still add rank-based metrics when no k metrics are requested."""
    result_df = add_sayt_metrics_columns(
        sayt_metrics_input_df,
        retrieved_codes_col="retrieved_codes",
        correct_codes_col="correct_code",
        k_values=[],
    )

    assert "reciprocal_rank" in result_df.columns, (
        "Expected reciprocal_rank to be added even when no "
        "precision/recall cutoffs are provided."
    )
    assert "correct_code_rank" in result_df.columns, (
        "Expected correct_code_rank to be added even when no precision/recall "
        "cutoffs are provided."
    )
    assert not any(
        column.startswith("precision_at_") for column in result_df.columns
    ), "Expected no precision_at_k columns when k_values is empty."
    assert not any(
        column.startswith("recall_at_") for column in result_df.columns
    ), "Expected no recall_at_k columns when k_values is empty."


@pytest.mark.parametrize("k", [0, -1])
def test_add_sayt_metrics_columns_raises_for_non_positive_k(sayt_metrics_input_df, k):
    """The helper should propagate invalid k errors from the metric functions."""
    with pytest.raises(ValueError, match="k must be a positive integer"):
        add_sayt_metrics_columns(
            sayt_metrics_input_df,
            retrieved_codes_col="retrieved_codes",
            correct_codes_col="correct_code",
            k_values=[k],
        )


def test_add_sayt_metrics_columns_uses_prefix_for_column_names(sayt_metrics_input_df):
    """Column names should be prepended with the prefix when one is provided."""
    result_df = add_sayt_metrics_columns(
        sayt_metrics_input_df,
        retrieved_codes_col="retrieved_codes",
        correct_codes_col="correct_code",
        k_values=[1],
        prefix="model_a_",
    )

    assert (
        "model_a_precision_at_1" in result_df.columns
    ), "Expected prefix to be prepended to precision column names."
    assert (
        "model_a_recall_at_1" in result_df.columns
    ), "Expected prefix to be prepended to recall column names."
    assert (
        "model_a_reciprocal_rank" in result_df.columns
    ), "Expected prefix to be prepended to reciprocal_rank column name."
    assert (
        "model_a_correct_code_rank" in result_df.columns
    ), "Expected prefix to be prepended to correct_code_rank column name."
    assert (
        "precision_at_1" not in result_df.columns
    ), "Expected unprefixed precision column to be absent when prefix is used."


def test_add_sayt_metrics_columns_default_prefix_produces_unprefixed_columns(
    sayt_metrics_input_df,
):
    """Omitting prefix should produce columns without any prefix."""
    result_df = add_sayt_metrics_columns(
        sayt_metrics_input_df,
        retrieved_codes_col="retrieved_codes",
        correct_codes_col="correct_code",
        k_values=[1],
    )

    assert (
        "precision_at_1" in result_df.columns
    ), "Expected unprefixed precision column when no prefix is provided."
    assert (
        "reciprocal_rank" in result_df.columns
    ), "Expected unprefixed reciprocal_rank column when no prefix is provided."


# ============================================================================
# Test summarise_performance_metrics function
# ============================================================================


@pytest.fixture
def sayt_metrics_df():
    """Pre-computed per-row metric DataFrame for summarise_performance_metrics tests.

    Row 0: correct code found at rank 1 (full match).
    Row 1: correct code found at rank 2 (partial match).
    Row 2: correct code not found (unmatched).
    """
    return pd.DataFrame(
        {
            "reciprocal_rank": [1.0, 0.5, 0.0],
            "correct_code_rank": [1.0, 2.0, None],
            "precision_at_1": [1.0, 0.0, 0.0],
            "precision_at_3": [1 / 3, 1 / 3, 0.0],
            "recall_at_1": [1.0, 0.0, 0.0],
            "recall_at_3": [1.0, 1.0, 0.0],
        }
    )


def test_summarise_performance_metrics_returns_sayt_performance_metrics_instance(
    sayt_metrics_df,
):
    """The function should return a SAYTPerformanceMetrics instance."""
    result = summarise_performance_metrics(
        sayt_metrics_df,
        suggestions_col="suggestions",
        code_digit_match_length=5,
        k_values=[1, 3],
        ave_time_per_query=12.5,
    )

    assert isinstance(result, SAYTPerformanceMetrics), (
        "Expected summarise_performance_metrics to return a SAYTPerformanceMetrics "
        "instance."
    )


def test_summarise_performance_metrics_total_queries_equals_row_count(
    sayt_metrics_df,
):
    """total_queries should equal the number of rows in the input DataFrame."""
    result = summarise_performance_metrics(
        sayt_metrics_df,
        suggestions_col="suggestions",
        code_digit_match_length=5,
        k_values=[1],
        ave_time_per_query=10.0,
    )

    assert (
        result.total_queries == 3
    ), "Expected total_queries to equal the number of rows in the input DataFrame."


def test_summarise_performance_metrics_stores_ave_time_per_query(
    sayt_metrics_df,
):
    """ave_time_per_query_ms should store the value passed in without modification."""
    result = summarise_performance_metrics(
        sayt_metrics_df,
        suggestions_col="suggestions",
        code_digit_match_length=5,
        k_values=[1],
        ave_time_per_query=42.7,
    )

    assert result.ave_time_per_query_ms == pytest.approx(
        42.7
    ), "Expected ave_time_per_query_ms to store the supplied value unchanged."


def test_summarise_performance_metrics_counts_rows_with_zero_correct_code_rank(
    sayt_metrics_df,
):
    """unmatched_query_count should equal the number of rows where correct_code_rank is None."""
    result = summarise_performance_metrics(
        sayt_metrics_df,
        suggestions_col="suggestions",
        code_digit_match_length=5,
        k_values=[1],
        ave_time_per_query=0.0,
    )

    assert (
        result.unmatched_query_count == 1
    ), "Expected unmatched_query_count to count rows where correct_code_rank is None."


def test_summarise_performance_metrics_computes_mean_reciprocal_rank(
    sayt_metrics_df,
):
    """Mrr should be the mean of the per-row mrr column."""
    result = summarise_performance_metrics(
        sayt_metrics_df,
        suggestions_col="suggestions",
        code_digit_match_length=5,
        k_values=[1],
        ave_time_per_query=0.0,
    )

    assert result.mrr == pytest.approx(
        (1.0 + 0.5 + 0.0) / 3
    ), "Expected mrr to equal the row-wise mean of the reciprocal_rank column."


def test_summarise_performance_metrics_computes_mean_rank(sayt_metrics_df):
    """mean_rank should be the mean of the per-row correct_code_rank column,
    ignoring None values.
    """
    result = summarise_performance_metrics(
        sayt_metrics_df,
        suggestions_col="suggestions",
        code_digit_match_length=5,
        k_values=[1],
        ave_time_per_query=0.0,
    )

    assert result.mean_rank == pytest.approx(
        (1.0 + 2.0) / 2
    ), "Expected mean_rank to equal the row-wise mean of the correct_code_rank column."


def test_summarise_performance_metrics_builds_precision_at_k_dict(sayt_metrics_df):
    """precision_at_k should map each k to the mean of the corresponding column."""
    result = summarise_performance_metrics(
        sayt_metrics_df,
        suggestions_col="suggestions",
        code_digit_match_length=5,
        k_values=[1, 3],
        ave_time_per_query=0.0,
    )

    assert result.precision_at_k == {
        1: pytest.approx((1.0 + 0.0 + 0.0) / 3),
        3: pytest.approx((1 / 3 + 1 / 3 + 0.0) / 3),
    }, "Expected precision_at_k to map each k to the mean precision across all rows."


def test_summarise_performance_metrics_builds_recall_at_k_dict(sayt_metrics_df):
    """recall_at_k should map each k to the mean of the corresponding column."""
    result = summarise_performance_metrics(
        sayt_metrics_df,
        suggestions_col="suggestions",
        code_digit_match_length=5,
        k_values=[1, 3],
        ave_time_per_query=0.0,
    )

    assert result.recall_at_k == {
        1: pytest.approx((1.0 + 0.0 + 0.0) / 3),
        3: pytest.approx((1.0 + 1.0 + 0.0) / 3),
    }, "Expected recall_at_k to map each k to the mean recall across all rows."


def test_summarise_performance_metrics_all_matched():
    """unmatched_query_count should be zero when every row has a non-zero rank."""
    df = pd.DataFrame(
        {
            "reciprocal_rank": [1.0, 0.5],
            "correct_code_rank": [1.0, 2.0],
            "precision_at_1": [1.0, 0.0],
            "recall_at_1": [1.0, 1.0],
        }
    )

    result = summarise_performance_metrics(
        df,
        suggestions_col="suggestions",
        code_digit_match_length=5,
        k_values=[1],
        ave_time_per_query=0.0,
    )

    assert (
        result.unmatched_query_count == 0
    ), "Expected unmatched_query_count to be 0 when all rows have a non-zero rank."


def test_summarise_performance_metrics_all_unmatched():
    """Mrr should be 0.0 and mean_rank NaN when no query returns the correct code."""
    df = pd.DataFrame(
        {
            "reciprocal_rank": [0.0, 0.0],
            "correct_code_rank": [None, None],
            "precision_at_1": [0.0, 0.0],
            "recall_at_1": [0.0, 0.0],
        }
    )

    result = summarise_performance_metrics(
        df,
        suggestions_col="suggestions",
        code_digit_match_length=5,
        k_values=[1],
        ave_time_per_query=0.0,
    )

    assert (
        result.unmatched_query_count == 2
    ), "Expected unmatched_query_count to equal total_queries when no code is found."
    assert result.mrr == pytest.approx(
        0.0
    ), "Expected mrr to be 0.0 when no query returns the correct code."
    assert math.isnan(
        result.mean_rank
    ), "Expected mean_rank to be NaN when no query has a rank (all correct_code_rank are None)."


def test_summarise_performance_metrics_single_row():
    """Metrics should be computed correctly for a single-row DataFrame."""
    df = pd.DataFrame(
        {
            "reciprocal_rank": [0.5],
            "correct_code_rank": [2.0],
            "precision_at_2": [0.5],
            "recall_at_2": [1.0],
        }
    )

    result = summarise_performance_metrics(
        df,
        suggestions_col="suggestions",
        code_digit_match_length=5,
        k_values=[2],
        ave_time_per_query=5.0,
    )

    assert (
        result.total_queries == 1
    ), "Expected total_queries to be 1 for a single-row DataFrame."
    assert (
        result.unmatched_query_count == 0
    ), "Expected unmatched_query_count to be 0 when the single row has a non-zero rank."
    assert result.mrr == pytest.approx(
        0.5
    ), "Expected mrr to equal the single row's reciprocal rank."
    assert result.mean_rank == pytest.approx(
        2.0
    ), "Expected mean_rank to equal the single row's rank value."


def test_summarise_performance_metrics_stores_suggestions_col(sayt_metrics_df):
    """suggestions_col and code_digit_match_length should be stored unchanged."""
    result = summarise_performance_metrics(
        sayt_metrics_df,
        suggestions_col="my_col",
        code_digit_match_length=7,
        k_values=[1],
        ave_time_per_query=0.0,
    )

    assert (
        result.suggestions_col == "my_col"
    ), "Expected suggestions_col to be stored as provided."
    assert (
        result.code_digit_match_length == 7
    ), "Expected code_digit_match_length to be stored as provided."


def test_summarise_performance_metrics_with_prefix_reads_prefixed_columns():
    """When prefix is provided, metrics should be read from the prefixed columns."""
    df = pd.DataFrame(
        {
            "pfx_reciprocal_rank": [1.0, 0.0],
            "pfx_correct_code_rank": [1.0, None],
            "pfx_precision_at_1": [1.0, 0.0],
            "pfx_recall_at_1": [1.0, 0.0],
        }
    )

    result = summarise_performance_metrics(
        df,
        suggestions_col="suggestions",
        code_digit_match_length=5,
        k_values=[1],
        ave_time_per_query=0.0,
        prefix="pfx_",
    )

    assert (
        result.total_queries == 2
    ), "Expected total_queries to equal the number of rows."
    assert (
        result.unmatched_query_count == 1
    ), "Expected one unmatched query from the prefixed correct_code_rank column."
    assert result.mrr == pytest.approx(
        0.5
    ), "Expected MRR computed from the prefixed reciprocal_rank column."


# ============================================================================
# Test build_sayt_metrics_comparison_table function
# ============================================================================


@pytest.fixture
def sayt_comparison_df():
    """DataFrame with two suggestion columns for comparison table tests.

    Codes are 4 characters; suggestions embed the code as the last 4 characters.
    Row 0: model_a returns correct code first; model_b misses.
    Row 1: both models miss.
    """
    return pd.DataFrame(
        {
            "correct_code": ["1111", "2222"],
            "suggestions_model_a": [
                ["label 1111", "label 3333"],
                ["label 3333", "label 4444"],
            ],
            "suggestions_model_b": [
                ["label 3333", "label 4444"],
                ["label 3333", "label 4444"],
            ],
        }
    )


def test_build_sayt_metrics_comparison_table_returns_dataframe(sayt_comparison_df):
    """The function should return a pandas DataFrame."""
    result = build_sayt_metrics_comparison_table(
        sayt_comparison_df,
        suggestions_cols_to_compare=["suggestions_model_a", "suggestions_model_b"],
        correct_codes_col="correct_code",
        k_values=[1],
        ave_time_per_query_dict={
            "suggestions_model_a": 10.0,
            "suggestions_model_b": 20.0,
        },
    )

    assert isinstance(
        result, pd.DataFrame
    ), "Expected build_sayt_metrics_comparison_table to return a pandas DataFrame."


def test_build_sayt_metrics_comparison_table_has_one_row_per_suggestions_column(
    sayt_comparison_df,
):
    """The result should contain one row for each suggestions column compared."""
    result = build_sayt_metrics_comparison_table(
        sayt_comparison_df,
        suggestions_cols_to_compare=["suggestions_model_a", "suggestions_model_b"],
        correct_codes_col="correct_code",
        k_values=[1],
        ave_time_per_query_dict={
            "suggestions_model_a": 10.0,
            "suggestions_model_b": 20.0,
        },
    )

    assert (
        len(result) == 2
    ), "Expected one row per suggestions column in the comparison table."


def test_build_sayt_metrics_comparison_table_keeps_suggestions_col_name(
    sayt_comparison_df,
):
    """Rows should retain the original suggestion column name in suggestions_col."""
    result = build_sayt_metrics_comparison_table(
        sayt_comparison_df,
        suggestions_cols_to_compare=["suggestions_model_a", "suggestions_model_b"],
        correct_codes_col="correct_code",
        k_values=[1],
        ave_time_per_query_dict={
            "suggestions_model_a": 10.0,
            "suggestions_model_b": 20.0,
        },
    )

    assert result["suggestions_col"].tolist() == [
        "suggestions_model_a",
        "suggestions_model_b",
    ], "Expected suggestions_col to match each compared suggestions column name."


def test_build_sayt_metrics_comparison_table_assigns_correct_ave_time_per_query(
    sayt_comparison_df,
):
    """Each row should use ave_time_per_query keyed by the suggestions column name."""
    result = build_sayt_metrics_comparison_table(
        sayt_comparison_df,
        suggestions_cols_to_compare=["suggestions_model_a", "suggestions_model_b"],
        correct_codes_col="correct_code",
        k_values=[1],
        ave_time_per_query_dict={
            "suggestions_model_a": 10.0,
            "suggestions_model_b": 20.0,
        },
    )

    assert result["ave_time_per_query_ms"].tolist() == pytest.approx([10.0, 20.0]), (
        "Expected ave_time_per_query_ms to be taken from ave_time_per_query_dict "
        "for each suggestions column."
    )


def test_build_sayt_metrics_comparison_table_computes_metrics_per_column(
    sayt_comparison_df,
):
    """Each row should reflect the metrics for the corresponding suggestions column."""
    result = build_sayt_metrics_comparison_table(
        sayt_comparison_df,
        suggestions_cols_to_compare=["suggestions_model_a", "suggestions_model_b"],
        correct_codes_col="correct_code",
        k_values=[1],
        ave_time_per_query_dict={
            "suggestions_model_a": 10.0,
            "suggestions_model_b": 20.0,
        },
    )

    assert result.loc[result["suggestions_col"] == "suggestions_model_a", "mrr"].iloc[
        0
    ] == pytest.approx(
        0.5
    ), "Expected MRR of 0.5 for model_a where only the first row matches."
    assert result.loc[result["suggestions_col"] == "suggestions_model_b", "mrr"].iloc[
        0
    ] == pytest.approx(0.0), "Expected MRR of 0.0 for model_b where no row matches."


def test_build_sayt_metrics_comparison_table_single_column(sayt_comparison_df):
    """The function should work correctly when only one suggestions column is provided."""
    result = build_sayt_metrics_comparison_table(
        sayt_comparison_df,
        suggestions_cols_to_compare=["suggestions_model_a"],
        correct_codes_col="correct_code",
        k_values=[1],
        ave_time_per_query_dict={"suggestions_model_a": 15.0},
    )

    assert (
        len(result) == 1
    ), "Expected exactly one row when a single suggestions column is provided."
    assert (
        result["suggestions_col"].iloc[0] == "suggestions_model_a"
    ), "Expected suggestions_col to match the single compared column."


def test_build_sayt_metrics_comparison_table_does_not_mutate_input(sayt_comparison_df):
    """The function should not modify the input DataFrame."""
    original_df = sayt_comparison_df.copy(deep=True)

    build_sayt_metrics_comparison_table(
        sayt_comparison_df,
        suggestions_cols_to_compare=["suggestions_model_a", "suggestions_model_b"],
        correct_codes_col="correct_code",
        k_values=[1],
        ave_time_per_query_dict={
            "suggestions_model_a": 10.0,
            "suggestions_model_b": 20.0,
        },
    )

    assert sayt_comparison_df.equals(
        original_df
    ), "Expected build_sayt_metrics_comparison_table to leave the input DataFrame unchanged."


# ============================================================================
# Test SAYTPerformanceMetrics class
# ============================================================================


def test_sayt_performance_metrics_instantiation_with_valid_data():
    """SAYTPerformanceMetrics should accept valid field values."""
    metrics = SAYTPerformanceMetrics(
        code_digit_match_length=5,
        suggestions_col="suggestions",
        total_queries=100,
        ave_time_per_query_ms=15.5,
        unmatched_query_count=5,
        mrr=0.85,
        mean_rank=2.3,
        precision_at_k={1: 0.9, 3: 0.8, 5: 0.7},
        recall_at_k={1: 0.7, 3: 0.85, 5: 0.9},
    )

    assert (
        metrics.total_queries == 100
    ), "Expected total_queries to be stored as provided."
    assert metrics.ave_time_per_query_ms == pytest.approx(
        15.5
    ), "Expected ave_time_per_query_ms to be stored as provided."
    assert (
        metrics.unmatched_query_count == 5
    ), "Expected unmatched_query_count to be stored as provided."
    assert metrics.mrr == pytest.approx(0.85), "Expected mrr to be stored as provided."
    assert metrics.mean_rank == pytest.approx(
        2.3
    ), "Expected mean_rank to be stored as provided."
    assert metrics.precision_at_k == {
        1: 0.9,
        3: 0.8,
        5: 0.7,
    }, "Expected precision_at_k to be stored as provided."
    assert metrics.recall_at_k == {
        1: 0.7,
        3: 0.85,
        5: 0.9,
    }, "Expected recall_at_k to be stored as provided."


def test_sayt_performance_metrics_instantiation_with_empty_k_dicts():
    """SAYTPerformanceMetrics should accept empty precision_at_k and recall_at_k."""
    metrics = SAYTPerformanceMetrics(
        code_digit_match_length=5,
        suggestions_col="suggestions",
        total_queries=50,
        ave_time_per_query_ms=10.0,
        unmatched_query_count=0,
        mrr=1.0,
        mean_rank=1.0,
        precision_at_k={},
        recall_at_k={},
    )

    assert (
        metrics.precision_at_k == {}
    ), "Expected precision_at_k to accept an empty dict."
    assert metrics.recall_at_k == {}, "Expected recall_at_k to accept an empty dict."


def test_sayt_performance_metrics_instantiation_with_zero_values():
    """SAYTPerformanceMetrics should accept zero values for numeric fields."""
    metrics = SAYTPerformanceMetrics(
        code_digit_match_length=5,
        suggestions_col="suggestions",
        total_queries=0,
        ave_time_per_query_ms=0.0,
        unmatched_query_count=0,
        mrr=0.0,
        mean_rank=0.0,
        precision_at_k={1: 0.0},
        recall_at_k={1: 0.0},
    )

    assert metrics.total_queries == 0, "Expected zero total_queries to be accepted."
    assert metrics.ave_time_per_query_ms == pytest.approx(
        0.0
    ), "Expected zero ave_time_per_query_ms to be accepted."


def test_sayt_performance_metrics_report_metrics_includes_all_fields():
    """report_metrics should include all performance metrics in the output."""
    metrics = SAYTPerformanceMetrics(
        code_digit_match_length=5,
        suggestions_col="test_suggestions",
        total_queries=100,
        ave_time_per_query_ms=15.5,
        unmatched_query_count=5,
        mrr=0.85,
        mean_rank=2.3,
        precision_at_k={1: 0.9, 3: 0.8},
        recall_at_k={1: 0.7, 3: 0.85},
    )
    report = metrics.report_metrics()

    assert "100" in report, "Expected total_queries value in report."
    assert (
        "Code digit match length: 5" in report
    ), "Expected code_digit_match_length in report."
    assert "15.50" in report, "Expected ave_time_per_query_ms value in report."
    assert "5" in report, "Expected unmatched_query_count in report."
    assert "0.8500" in report, "Expected mrr value in report."
    assert "2.30" in report, "Expected mean_rank value in report."
    assert "test_suggestions" in report, "Expected suggestions_col name in report."
    assert "Precision@1" in report, "Expected Precision@1 in report."
    assert "Precision@3" in report, "Expected Precision@3 in report."
    assert "Recall@1" in report, "Expected Recall@1 in report."
    assert "Recall@3" in report, "Expected Recall@3 in report."


def test_sayt_performance_metrics_report_metrics_returns_string():
    """report_metrics should return a string."""
    metrics = SAYTPerformanceMetrics(
        code_digit_match_length=5,
        suggestions_col="suggestions",
        total_queries=50,
        ave_time_per_query_ms=10.0,
        unmatched_query_count=0,
        mrr=0.5,
        mean_rank=2.0,
        precision_at_k={1: 0.8},
        recall_at_k={1: 0.6},
    )
    report = metrics.report_metrics()

    assert isinstance(report, str), "Expected report_metrics to return a string."


def test_sayt_performance_metrics_report_metrics_starts_with_header():
    """report_metrics should begin with a header line."""
    metrics = SAYTPerformanceMetrics(
        code_digit_match_length=5,
        suggestions_col="my_col",
        total_queries=10,
        ave_time_per_query_ms=5.0,
        unmatched_query_count=1,
        mrr=0.9,
        mean_rank=1.5,
        precision_at_k={},
        recall_at_k={},
    )
    report = metrics.report_metrics()

    assert report.startswith(
        "\nSAYT Performance Metrics for column my_col:"
    ), "Expected report to start with header including the suggestions column name."


def test_sayt_performance_metrics_report_metrics_contains_formatted_numbers():
    """report_metrics should format numbers with appropriate precision."""
    metrics = SAYTPerformanceMetrics(
        code_digit_match_length=5,
        suggestions_col="suggestions",
        total_queries=100,
        ave_time_per_query_ms=12.3456,
        unmatched_query_count=8,
        mrr=0.123456,
        mean_rank=3.6789,
        precision_at_k={1: 0.789123},
        recall_at_k={1: 0.456789},
    )
    report = metrics.report_metrics()

    assert (
        "12.35" in report
    ), "Expected ave_time_per_query_ms formatted to 2 decimal places."
    assert "0.1235" in report, "Expected mrr formatted to 4 decimal places."
    assert "3.68" in report, "Expected mean_rank formatted to 2 decimal places."
    assert (
        "0.7891" in report
    ), "Expected Precision@k values formatted to 4 decimal places."
    assert "0.4568" in report, "Expected Recall@k values formatted to 4 decimal places."


def test_sayt_performance_metrics_report_metrics_with_multiple_k_values():
    """report_metrics should report all k values in precision_at_k and recall_at_k."""
    metrics = SAYTPerformanceMetrics(
        code_digit_match_length=5,
        suggestions_col="suggestions",
        total_queries=100,
        ave_time_per_query_ms=10.0,
        unmatched_query_count=0,
        mrr=0.8,
        mean_rank=2.0,
        precision_at_k={1: 0.9, 3: 0.7, 5: 0.6, 10: 0.4},
        recall_at_k={1: 0.5, 3: 0.7, 5: 0.8, 10: 0.9},
    )
    report = metrics.report_metrics()

    assert (
        "Precision@1" in report and "0.9000" in report
    ), "Expected Precision@1 with value in report."
    assert (
        "Precision@3" in report and "0.7000" in report
    ), "Expected Precision@3 with value in report."
    assert (
        "Precision@5" in report and "0.6000" in report
    ), "Expected Precision@5 with value in report."
    assert (
        "Precision@10" in report and "0.4000" in report
    ), "Expected Precision@10 with value in report."
    assert (
        "Recall@1" in report and "0.5000" in report
    ), "Expected Recall@1 with value in report."
    assert (
        "Recall@10" in report and "0.9000" in report
    ), "Expected Recall@10 with value in report."


def test_sayt_performance_metrics_report_metrics_with_empty_k_dicts():
    """report_metrics should handle empty precision_at_k and recall_at_k gracefully."""
    metrics = SAYTPerformanceMetrics(
        code_digit_match_length=5,
        suggestions_col="suggestions",
        total_queries=50,
        ave_time_per_query_ms=8.0,
        unmatched_query_count=2,
        mrr=0.6,
        mean_rank=3.0,
        precision_at_k={},
        recall_at_k={},
    )
    report = metrics.report_metrics()

    assert isinstance(
        report, str
    ), "Expected report_metrics to return a string even with empty k dicts."
    assert (
        "SAYT Performance Metrics for column" in report
    ), "Expected header even with empty k dicts."
    assert (
        "Total queries: 50" in report
    ), "Expected basic metrics even with empty k dicts."


def test_sayt_performance_metrics_validates_field_types():
    """SAYTPerformanceMetrics should validate field types via Pydantic."""
    with pytest.raises(ValidationError):
        SAYTPerformanceMetrics(
            code_digit_match_length=5,
            suggestions_col="suggestions",
            total_queries="not_an_int",
            ave_time_per_query_ms=10.0,
            unmatched_query_count=0,
            mrr=0.8,
            mean_rank=2.0,
            precision_at_k={},
            recall_at_k={},
        )


def test_sayt_performance_metrics_validates_required_fields():
    """SAYTPerformanceMetrics should require all fields."""
    with pytest.raises(ValidationError):
        SAYTPerformanceMetrics(
            code_digit_match_length=5,
            suggestions_col="suggestions",
            total_queries=100,
            ave_time_per_query_ms=10.0,
            unmatched_query_count=0,
            mrr=0.8,
            # missing mean_rank
            precision_at_k={},
            recall_at_k={},
        )


def test_sayt_performance_metrics_report_metrics_sorts_k_values():
    """report_metrics should print k values in sorted order."""
    metrics = SAYTPerformanceMetrics(
        code_digit_match_length=5,
        suggestions_col="suggestions",
        total_queries=100,
        ave_time_per_query_ms=10.0,
        unmatched_query_count=0,
        mrr=0.8,
        mean_rank=2.0,
        precision_at_k={5: 0.6, 1: 0.9, 3: 0.7},
        recall_at_k={5: 0.8, 1: 0.5, 3: 0.7},
    )
    report = metrics.report_metrics()
    lines = report.split("\n")

    # Find indices of k values in the report
    precision_lines = [
        i for i, line in enumerate(lines) if line.startswith(" Precision@")
    ]
    recall_lines = [i for i, line in enumerate(lines) if line.startswith(" Recall@")]

    assert precision_lines == sorted(
        precision_lines
    ), "Expected precision lines to appear in sorted order by k value."
    assert recall_lines == sorted(
        recall_lines
    ), "Expected recall lines to appear in sorted order by k value."
    # Verify the order is 1, 3, 5
    assert (
        "Precision@1" in lines[precision_lines[0]]
    ), "Expected Precision@1 to appear first."
    assert (
        "Precision@3" in lines[precision_lines[1]]
    ), "Expected Precision@3 to appear second."
    assert (
        "Precision@5" in lines[precision_lines[2]]
    ), "Expected Precision@5 to appear third."


# ============================================================================
# Test compute_performance_metrics_from_suggestions function
# ============================================================================


@pytest.fixture
def suggestions_df():
    """DataFrame with raw suggestion strings for compute_performance_metrics tests.

    Codes are the last 4 characters of each suggestion string.
    Row 0: correct code '1111' appears first in suggestions -> rank 1.
    Row 1: correct code '2222' appears second -> rank 2.
    Row 2: correct code '3333' absent from suggestions -> unmatched.
    """
    return pd.DataFrame(
        {
            "correct_code": ["1111", "2222", "3333"],
            "suggestions": [
                ["alpha 1111", "beta 4444"],
                ["alpha 4444", "beta 2222"],
                ["alpha 4444", "beta 5555"],
            ],
        }
    )


def test_compute_performance_metrics_from_suggestions_returns_sayt_performance_metrics_instance(
    suggestions_df,
):
    """The function should return a SAYTPerformanceMetrics instance."""
    result = compute_performance_metrics_from_suggestions(
        suggestions_df,
        correct_codes_col="correct_code",
        suggestions_col="suggestions",
        code_length=4,
        k_values=[1],
        ave_time_per_query=10.0,
    )

    assert isinstance(result, SAYTPerformanceMetrics), (
        "Expected compute_performance_metrics_from_suggestions to return a "
        "SAYTPerformanceMetrics instance."
    )


def test_compute_performance_metrics_from_suggestions_does_not_mutate_input(
    suggestions_df,
):
    """The function should leave the input DataFrame unchanged."""
    original_df = suggestions_df.copy(deep=True)

    compute_performance_metrics_from_suggestions(
        suggestions_df,
        correct_codes_col="correct_code",
        suggestions_col="suggestions",
        code_length=4,
        k_values=[1],
        ave_time_per_query=5.0,
    )

    assert suggestions_df.equals(original_df), (
        "Expected compute_performance_metrics_from_suggestions to leave the input "
        "DataFrame unchanged."
    )


def test_compute_performance_metrics_from_suggestions_extracts_codes_by_code_length(
    suggestions_df,
):
    """code_length controls how many trailing characters are used as the code."""
    result = compute_performance_metrics_from_suggestions(
        suggestions_df,
        correct_codes_col="correct_code",
        suggestions_col="suggestions",
        code_length=4,
        k_values=[1],
        ave_time_per_query=0.0,
    )

    # Row 0: '1111' found at rank 1, row 1: '2222' found at rank 2, row 2: unmatched
    assert (
        result.unmatched_query_count == 1
    ), "Expected one unmatched query when code_length correctly extracts 4-char codes."
    assert result.mrr == pytest.approx(
        (1.0 + 0.5 + 0.0) / 3
    ), "Expected MRR computed from correctly extracted codes."


def test_compute_performance_metrics_from_suggestions_total_queries_equals_row_count(
    suggestions_df,
):
    """total_queries in the result should equal the number of rows in the input."""
    result = compute_performance_metrics_from_suggestions(
        suggestions_df,
        correct_codes_col="correct_code",
        suggestions_col="suggestions",
        code_length=4,
        k_values=[1],
        ave_time_per_query=0.0,
    )

    assert result.total_queries == len(
        suggestions_df
    ), "Expected total_queries to equal the number of rows in the input DataFrame."


def test_compute_performance_metrics_from_suggestions_stores_ave_time_per_query(
    suggestions_df,
):
    """ave_time_per_query_ms should be stored from the argument without modification."""
    result = compute_performance_metrics_from_suggestions(
        suggestions_df,
        correct_codes_col="correct_code",
        suggestions_col="suggestions",
        code_length=4,
        k_values=[1],
        ave_time_per_query=99.9,
    )

    assert result.ave_time_per_query_ms == pytest.approx(
        99.9
    ), "Expected ave_time_per_query_ms to equal the supplied argument."


def test_compute_performance_metrics_from_suggestions_all_unmatched():
    """All queries unmatched should yield MRR of 0.0 and unmatched_query_count equal to total."""
    df = pd.DataFrame(
        {
            "correct_code": ["9999", "8888"],
            "suggestions": [
                ["label 1111", "label 2222"],
                ["label 3333", "label 4444"],
            ],
        }
    )

    result = compute_performance_metrics_from_suggestions(
        df,
        correct_codes_col="correct_code",
        suggestions_col="suggestions",
        code_length=4,
        k_values=[1],
        ave_time_per_query=0.0,
    )

    assert (
        result.unmatched_query_count == 2
    ), "Expected unmatched_query_count to equal total_queries when no suggestion matches."
    assert result.mrr == pytest.approx(
        0.0
    ), "Expected MRR of 0.0 when no suggestions contain the correct code."


def test_compute_performance_metrics_from_suggestions_all_matched_at_rank_1():
    """All queries matched at rank 1 should yield MRR of 1.0."""
    df = pd.DataFrame(
        {
            "correct_code": ["1111", "2222"],
            "suggestions": [
                ["label 1111", "label 9999"],
                ["label 2222", "label 9999"],
            ],
        }
    )

    result = compute_performance_metrics_from_suggestions(
        df,
        correct_codes_col="correct_code",
        suggestions_col="suggestions",
        code_length=4,
        k_values=[1],
        ave_time_per_query=0.0,
    )

    assert (
        result.unmatched_query_count == 0
    ), "Expected no unmatched queries when all correct codes appear first."
    assert result.mrr == pytest.approx(
        1.0
    ), "Expected MRR of 1.0 when all correct codes appear at rank 1."


def test_compute_performance_metrics_from_suggestions_computes_precision_and_recall_at_k(
    suggestions_df,
):
    """precision_at_k and recall_at_k dicts should contain the requested k values."""
    result = compute_performance_metrics_from_suggestions(
        suggestions_df,
        correct_codes_col="correct_code",
        suggestions_col="suggestions",
        code_length=4,
        k_values=[1, 2],
        ave_time_per_query=0.0,
    )

    assert set(result.precision_at_k.keys()) == {
        1,
        2,
    }, "Expected precision_at_k to contain keys for each requested k value."
    assert set(result.recall_at_k.keys()) == {
        1,
        2,
    }, "Expected recall_at_k to contain keys for each requested k value."


def test_compute_performance_metrics_from_suggestions_stores_suggestions_col(
    suggestions_df,
):
    """The result should store the suggestions column name."""
    result = compute_performance_metrics_from_suggestions(
        suggestions_df,
        correct_codes_col="correct_code",
        suggestions_col="suggestions",
        code_length=4,
        k_values=[1],
        ave_time_per_query=0.0,
    )

    assert (
        result.suggestions_col == "suggestions"
    ), "Expected suggestions_col to be stored in the result."


def test_compute_performance_metrics_from_suggestions_applies_code_digit_match_length():
    """code_digit_match_length should truncate both correct and retrieved codes before scoring."""
    df = pd.DataFrame(
        {
            "correct_code": ["1234"],
            "suggestions": [["alpha 1239", "beta 9999"]],
        }
    )

    # Without truncation: no exact match between 1234 and 1239.
    without_truncation = compute_performance_metrics_from_suggestions(
        df,
        correct_codes_col="correct_code",
        suggestions_col="suggestions",
        code_length=4,
        k_values=[1],
        ave_time_per_query=0.0,
    )

    # With truncation to 3 digits: 1234 -> 123 and 1239 -> 123, so rank becomes 1.
    with_truncation = compute_performance_metrics_from_suggestions(
        df,
        correct_codes_col="correct_code",
        suggestions_col="suggestions",
        code_length=4,
        k_values=[1],
        ave_time_per_query=0.0,
        code_digit_match_length=3,
    )

    assert (
        without_truncation.unmatched_query_count == 1
    ), "Expected query to be unmatched when full 4-digit codes are compared."
    assert without_truncation.mrr == pytest.approx(
        0.0
    ), "Expected MRR to be 0.0 when full 4-digit codes do not match."

    assert (
        with_truncation.unmatched_query_count == 0
    ), "Expected truncation to make the query matched."
    assert with_truncation.mrr == pytest.approx(
        1.0
    ), "Expected MRR to be 1.0 when truncated codes match at rank 1."
    assert (
        with_truncation.code_digit_match_length == 3
    ), "Expected result to report the requested truncated match length."
