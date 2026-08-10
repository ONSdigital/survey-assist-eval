"""Tests for SAYT performance metric helper functions."""

# pylint: disable=redefined-outer-name

import pandas as pd
import pytest

from survey_assist_eval.evaluation.sayt.performance_metrics_functions import (
    add_sayt_metrics_columns,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    get_rank_of_correct_code,
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
        "Expected Recall@3 to be 1.0 when the correct code is present even if fewer "
        "than k results are returned."
    )


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


# ============================================================================
# Test get_rank_of_correct_code function
# ============================================================================


def test_get_rank_of_correct_code_returns_first_matching_rank():
    """Rank should report the first position containing the correct code."""
    rank = get_rank_of_correct_code(["1111", "2222", "1111"], "1111")

    assert rank == pytest.approx(1.0), (
        "Expected rank to report the first matching position when duplicates appear "
        "later in the list."
    )


def test_get_rank_of_correct_code_returns_zero_when_code_not_found():
    """Rank should be zero when the correct code is absent."""
    rank = get_rank_of_correct_code(["1111", "2222", "3333"], "4444")

    assert rank == pytest.approx(0.0), (
        "Expected rank to be 0.0 when the correct code is absent from the retrieved "
        "list."
    )


def test_get_rank_of_correct_code_returns_rank_beyond_first_position():
    """Rank should reflect the first matching position when it is not first."""
    rank = get_rank_of_correct_code(["1111", "2222", "3333"], "3333")

    assert rank == pytest.approx(3.0), (
        "Expected rank to equal 3.0 when the correct code is first found in the "
        "third position."
    )


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
        correct_code_col="correct_code",
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
        correct_code_col="correct_code",
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
        correct_code_col="correct_code",
        k_values=[1],
    )

    assert result_df["mrr"].tolist() == pytest.approx([1.0, 0.0, 0.0]), (
        "Expected MRR values to equal the reciprocal rank of the first matching code "
        "or 0.0 when there is no match."
    )
    assert result_df["mean_rank"].tolist() == pytest.approx([1.0, 0.0, 0.0]), (
        "Expected mean_rank values to store the first matching rank or 0.0 when the "
        "correct code is absent."
    )


def test_add_sayt_metrics_columns_handles_empty_k_values(sayt_metrics_input_df):
    """The helper should still add rank-based metrics when no k metrics are requested."""
    result_df = add_sayt_metrics_columns(
        sayt_metrics_input_df,
        retrieved_codes_col="retrieved_codes",
        correct_code_col="correct_code",
        k_values=[],
    )

    assert (
        "mrr" in result_df.columns
    ), "Expected mrr to be added even when no precision/recall cutoffs are provided."
    assert "mean_rank" in result_df.columns, (
        "Expected mean_rank to be added even when no precision/recall cutoffs are "
        "provided."
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
            correct_code_col="correct_code",
            k_values=[k],
        )
