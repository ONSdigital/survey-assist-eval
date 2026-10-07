"""Tests for additional performance metrics functions."""

# pylint: disable=redefined-outer-name

import math

import pytest

from survey_assist_eval.evaluation.sayt.performance_metrics_functions import (
    compute_normalized_discounted_cumulative_gain_at_k,
    compute_reciprocal_rank_of_final_correct_code,
)

EXAMPLE_CASES = [
    pytest.param(
        {
            "ranks_by_code_dict": {"1111": [1], "2222": [2]},
            "k": 5,
            "total_relevant_ranks": 2,
            "rr_final_code": 1 / 2,
            "ndcg_at_k": 1.0,
            "ndcg_all_at_k": 1.0,
        },
        id="perfect_ranking",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {"1111": [3]},
            "k": 5,
            "total_relevant_ranks": 1,
            "rr_final_code": 1 / 3,
            "ndcg_at_k": 0.5,
            "ndcg_all_at_k": 0.5,
        },
        id="single-code-at-later-rank",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {"1111": [2], "2222": [4]},
            "k": 5,
            "total_relevant_ranks": 2,
            "rr_final_code": 1 / 4,
            "ndcg_at_k": (1 / math.log2(3) + 1 / math.log2(5))
            / sum(1 / math.log2(i + 1) for i in range(1, 3)),
            "ndcg_all_at_k": (1 / math.log2(3) + 1 / math.log2(5))
            / sum(1 / math.log2(i + 1) for i in range(1, 3)),
        },
        id="multiple-distinct-codes",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {"1111": [4, 1], "2222": [3]},
            "k": 5,
            "total_relevant_ranks": 3,
            "rr_final_code": 1 / 3,
            "ndcg_at_k": (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)),
            "ndcg_all_at_k": (1 + 1 / math.log2(4) + 1 / math.log2(5))
            / (1 + 1 / math.log2(3) + 1 / math.log2(4)),
        },
        id="unsorted-ranks-use-earliest-hit-per-code",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {"1111": [1, 3], "2222": [4]},
            "k": 5,
            "total_relevant_ranks": 3,
            "rr_final_code": 1 / 4,
            "ndcg_at_k": (1 + 1 / math.log2(5))
            / sum(1 / math.log2(i + 1) for i in range(1, 3)),
            "ndcg_all_at_k": (1 + 1 / math.log2(4) + 1 / math.log2(5))
            / sum(1 / math.log2(i + 1) for i in range(1, 4)),
        },
        id="duplicate-ranks-for-one-code",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {"1111": [1], "2222": [5]},
            "k": 5,
            "total_relevant_ranks": 2,
            "rr_final_code": 1 / 5,
            "ndcg_at_k": (1 + 1 / math.log2(6))
            / sum(1 / math.log2(i + 1) for i in range(1, 3)),
            "ndcg_all_at_k": (1 + 1 / math.log2(6))
            / sum(1 / math.log2(i + 1) for i in range(1, 3)),
        },
        id="final_code_at_rank_five",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {"1111": [1, 2, 3]},
            "k": 5,
            "total_relevant_ranks": 3,
            "rr_final_code": 1.0,
            "ndcg_at_k": 1.0,
            "ndcg_all_at_k": 1.0,
        },
        id="single_code_multiple_retrievals",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {"1111": [1, 2, 3]},
            "k": 5,
            "total_relevant_ranks": 4,
            "rr_final_code": 1.0,
            "ndcg_at_k": 1.0,
            "ndcg_all_at_k": (1 + 1 / math.log2(3) + 1 / math.log2(4))
            / sum(1 / math.log2(i + 1) for i in range(1, 5)),
        },
        id="missing_one_relevant_rank",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {
                "1111": [1],
                "2222": [2],
                "3333": [3],
            },
            "k": 9,
            "total_relevant_ranks": 3,
            "rr_final_code": 1 / 3,
            "ndcg_at_k": 1.0,
            "ndcg_all_at_k": 1.0,
        },
        id="ideal-ranking-three-distinct-codes",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {
                "1111": [1, 2],
                "2222": [3],
                "3333": [4],
            },
            "k": 9,
            "total_relevant_ranks": 4,
            "rr_final_code": 1 / 4,
            "ndcg_at_k": (1 + 1 / math.log2(4) + 1 / math.log2(5))
            / (1 + 1 / math.log2(3) + 1 / math.log2(4)),
            "ndcg_all_at_k": 1.0,
        },
        id="duplicate-suggestion-with-all-codes-found",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {
                "1111": [1],
                "2222": [8],
                "3333": [],
            },
            "k": 9,
            "total_relevant_ranks": 3,
            "rr_final_code": 0.0,
            "ndcg_at_k": (1 + 1 / math.log2(9))
            / (1 + 1 / math.log2(3) + 1 / math.log2(4)),
            "ndcg_all_at_k": (1 + 1 / math.log2(9))
            / (1 + 1 / math.log2(3) + 1 / math.log2(4)),
        },
        id="late-hit-with-one-correct-code-missing",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {
                "1111": [2],
                "2222": [5],
                "3333": [7],
            },
            "k": 9,
            "total_relevant_ranks": 3,
            "rr_final_code": 1 / 7,
            "ndcg_at_k": (1 / math.log2(3) + 1 / math.log2(6) + 1 / math.log2(8))
            / (1 + 1 / math.log2(3) + 1 / math.log2(4)),
            "ndcg_all_at_k": (1 / math.log2(3) + 1 / math.log2(6) + 1 / math.log2(8))
            / (1 + 1 / math.log2(3) + 1 / math.log2(4)),
        },
        id="three-correct-codes-at-spaced-ranks",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {"1111": [1, 6], "2222": [3]},
            "k": 5,
            "total_relevant_ranks": 7,
            "rr_final_code": 1 / 3,
            "ndcg_at_k": (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)),
            "ndcg_all_at_k": (1 + 1 / math.log2(4))
            / sum(1 / math.log2(i + 1) for i in range(1, 6)),
        },
        id="total-relevant-ranks-exceed-k",
    ),
]


@pytest.fixture(params=EXAMPLE_CASES)
def example_case(request):
    """Provide one non-empty performance-metric example."""
    return request.param


EDGE_CASES = [
    pytest.param(
        {
            "ranks_by_code_dict": {},
            "k": 5,
            "total_relevant_ranks": 5,
            # metrics
            "rr_final_code": 0.0,
            "ndcg_at_k": 0.0,
            "ndcg_all_at_k": 0.0,
        },
        id="empty_dict",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {"1111": []},
            "k": 5,
            "total_relevant_ranks": 5,
            # metrics
            "rr_final_code": 0.0,
            "ndcg_at_k": 0.0,
            "ndcg_all_at_k": 0.0,
        },
        id="single_code_no_ranks",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {"1111": [], "2222": []},
            "k": 5,
            "total_relevant_ranks": 5,
            # metrics
            "rr_final_code": 0.0,
            "ndcg_at_k": 0.0,
            "ndcg_all_at_k": 0.0,
        },
        id="multiple_codes_no_ranks",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {"1111": [], "2222": [1, 3]},
            "k": 5,
            "total_relevant_ranks": 5,
            # metrics
            "rr_final_code": 0.0,
            "ndcg_at_k": (1 / math.log2(2))
            / sum(1 / math.log2(i + 1) for i in range(1, 3)),
            "ndcg_all_at_k": (1 / math.log2(2) + 1 / math.log2(4))
            / sum(1 / math.log2(i + 1) for i in range(1, 6)),
        },
        id="final_code_missing_other_code_found",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {"1111": [], "2222": [6]},
            "k": 5,
            "total_relevant_ranks": 5,
            "rr_final_code": 0.0,
            "ndcg_at_k": 0.0,
            "ndcg_all_at_k": 0.0,
        },
        id="final_code_missing_other_beyond_k",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {"1111": [6]},
            "k": 5,
            "total_relevant_ranks": 5,
            "rr_final_code": 1 / 6,
            "ndcg_at_k": 0.0,
            "ndcg_all_at_k": 0.0,
        },
        id="single_relevant_rank_beyond_k",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {"1111": [1, 6]},
            "k": 5,
            "total_relevant_ranks": 5,
            "rr_final_code": 1.0,
            "ndcg_at_k": 1.0,
            "ndcg_all_at_k": (1 / math.log2(2))
            / sum(1 / math.log2(i + 1) for i in range(1, 6)),
        },
        id="only_in_cutoff_rank_contributes",
    ),
    pytest.param(
        {
            "ranks_by_code_dict": {"1111": [6], "2222": [7]},
            "k": 5,
            "total_relevant_ranks": 5,
            "rr_final_code": 1 / 7,
            "ndcg_at_k": 0.0,
            "ndcg_all_at_k": 0.0,
        },
        id="all_relevant_ranks_beyond_k",
    ),
]


@pytest.fixture(params=EDGE_CASES)
def edge_case(request):
    """Provide one empty-input or cutoff edge case."""
    return request.param


# ============================================================================
# Test compute_reciprocal_rank_of_final_correct_code function
# ============================================================================
def test_compute_reciprocal_rank_of_final_correct_code_with_examples(example_case):
    """Check final-code reciprocal rank for representative examples."""
    reciprocal_rank = compute_reciprocal_rank_of_final_correct_code(
        example_case["ranks_by_code_dict"]
    )

    assert reciprocal_rank == pytest.approx(example_case["rr_final_code"])


def test_compute_reciprocal_rank_of_final_correct_code_with_edge_cases(
    edge_case,
):
    """Check final-code reciprocal rank for empty and boundary cases."""
    reciprocal_rank = compute_reciprocal_rank_of_final_correct_code(
        edge_case["ranks_by_code_dict"]
    )

    assert reciprocal_rank == pytest.approx(edge_case["rr_final_code"])


# ============================================================================
# Test compute_normalized_discounted_cumulative_gain_at_k function
# ============================================================================
def test_compute_normalized_discounted_cumulative_gain_at_k_with_examples(
    example_case,
):
    """Check single-hit NDCG@k for representative examples."""
    ranks_by_code = example_case["ranks_by_code_dict"]
    ndcg_at_k = compute_normalized_discounted_cumulative_gain_at_k(
        ranks_by_code, example_case["k"]
    )

    assert ndcg_at_k == pytest.approx(example_case["ndcg_at_k"])


def test_compute_normalized_discounted_cumulative_gain_at_k_all_with_examples(
    example_case,
):
    """Check all-ranks NDCG@k for representative examples."""
    ranks_by_code = example_case["ranks_by_code_dict"]
    ndcg_all_at_k = compute_normalized_discounted_cumulative_gain_at_k(
        ranks_by_code,
        example_case["k"],
        include_all_relevant_ranks=True,
        total_relevant_ranks=example_case["total_relevant_ranks"],
    )

    assert ndcg_all_at_k == pytest.approx(example_case["ndcg_all_at_k"])


def test_compute_normalized_discounted_cumulative_gain_at_k_with_edge_cases(
    edge_case,
):
    """Check single-hit NDCG@k for empty and boundary cases."""
    ndcg_at_k = compute_normalized_discounted_cumulative_gain_at_k(
        edge_case["ranks_by_code_dict"], edge_case["k"]
    )

    assert ndcg_at_k == pytest.approx(edge_case["ndcg_at_k"])


def test_compute_normalized_discounted_cumulative_gain_at_k_all_with_edge_cases(
    edge_case,
):
    """Check all-ranks NDCG@k for empty and boundary cases."""
    ndcg_at_k = compute_normalized_discounted_cumulative_gain_at_k(
        edge_case["ranks_by_code_dict"],
        edge_case["k"],
        include_all_relevant_ranks=True,
        total_relevant_ranks=edge_case["total_relevant_ranks"],
    )

    assert ndcg_at_k == pytest.approx(edge_case["ndcg_all_at_k"])
