"""Evaluate SAYT retriever behaviour for SIC suggestion ties.

This notebook compares several lookup suggesters (ngram, prefix, and semantic)
for the UK SIC 2007 knowledge base and measures their ranking quality against a
labelled evaluation dataset. It loads the evaluation dataset and lookup corpus,
requests suggestions for varying input lengths, computes performance metrics, and
summarises how often ties or hard limits affect the final result set.

The script is designed to be run as an exploratory notebook for assessing how
ranking and tie handling influence retrieval quality and latency for suggestion
generation.
"""

# pylint: disable=C0103

# %%
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from survey_assist_utils.logging import get_logger

from notebooks.sayt.sayt_utils import (
    build_lookup_suggester,
    build_sayt_corpus_from_df,
    get_suggestions_by_chars,
)
from survey_assist_eval.evaluation.sayt.performance_metrics_functions import (
    compute_performance_metrics_from_suggestions,
)

# %%
CODE_TYPE = "sic"
MAX_SUGGESTIONS = 30
CORRECT_CODE_COL = "correct_sic_code"
CHARACTERS_LIST = list(range(6, 10))
# CHARACTERS_LIST = list(range(4, 6))

NGRAM_WEIGHT = 3
PREFIX_WEIGHT = 4
SEMANTIC_WEIGHT = 3

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")


logger = get_logger(__name__)


# %%
def get_suggestions_for_limit(
    evaluation_data: pd.DataFrame,
    suggester_map: dict[str, Any],
    *,
    hard_limit: bool,
) -> tuple[pd.DataFrame, dict]:
    """Generate scored suggestions using the requested limit behaviour.

    Args:
        evaluation_data: Queries and evaluation data to generate suggestions for.
        suggester_map: Named suggesters to use for suggestion generation.
        hard_limit: Whether to enforce the suggestion limit strictly.

    Returns:
        A tuple containing the dataframe augmented with suggestions and scores,
        and a dictionary of average query times keyed by suggestion column.
    """
    return get_suggestions_by_chars(
        df=evaluation_data,
        suggesters_dict=suggester_map,
        num_chars=CHARACTERS_LIST,
        suggestions_limit=MAX_SUGGESTIONS,
        hard_suggestions_limit=hard_limit,
        with_scores=True,
    )


def plot_rank_distributions(
    evaluation_data: pd.DataFrame,
    rank_columns: list[str],
) -> None:
    """Display a histogram of ranks for the supplied result columns.

    Args:
        evaluation_data: Results dataframe containing the rank columns.
        rank_columns: Names of the rank columns to include in the histogram.

    Returns:
        None.
    """
    rank_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "rank": evaluation_data[rank_column_name].dropna(),
                    "retriever": rank_column_name,
                }
            )
            for rank_column_name in rank_columns
        ],
        ignore_index=True,
    )
    retriever_counts = rank_df["retriever"].value_counts()
    rank_df["retriever"] = rank_df["retriever"].map(
        lambda retriever: f"{retriever} (total matched: {retriever_counts[retriever]})"
    )
    fig = px.histogram(
        rank_df,
        x="rank",
        color="retriever",
        nbins=30,
        barmode="group",
        opacity=0.7,
        title="Rank distributions (with hard limit)",
        labels={"rank": "Rank", "count": "Count"},
    )
    fig.show()


def find_extended_suggestion_ties(
    suggestion_results: pd.DataFrame,
    characters: list[int],
    suggester_name: str,
    max_suggestions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find results where the extended suggestions match the rank's score.

    Args:
        suggestion_results: Results containing suggestions, scores, and ranks.
        characters: Query character counts used to select result columns.
        suggester_name: Suggester suffix used in the result column names.
        max_suggestions: Suggestion cutoff used for the score comparison.

    Returns:
        A tuple containing the results with extended suggestions and tie-check
        columns, and the subset where a rank's score matches the score at the
        suggestion cutoff.
    """
    extended_results = suggestion_results.copy()
    extended_suggestion_masks = []
    for num_chars in characters:
        suggestion_column_name = f"suggestions_{num_chars}chars_{suggester_name}"
        scores_col = f"scores_{num_chars}chars_{suggester_name}"
        rank_col = f"rank_{num_chars}chars_{suggester_name}_5digitmatch"

        extended_mask = (
            extended_results[suggestion_column_name].str.len() > max_suggestions
        ).fillna(False)
        extended_suggestion_masks.append(extended_mask)
        rank_exceeds_col = f"rank_gt_max_suggestions_{num_chars}chars"
        extended_results[rank_exceeds_col] = (
            extended_results[rank_col].gt(max_suggestions).fillna(False)
        )
        check_col = f"same_score_at_limit_and_rank_{num_chars}chars"
        extended_results[check_col] = [
            (
                False
                if not has_extended_suggestions
                or pd.isna(rank)
                or not 1 <= int(rank) <= len(scores)
                else scores[max_suggestions - 1] == scores[int(rank) - 1]
            )
            for scores, rank, has_extended_suggestions in zip(
                extended_results[scores_col],
                extended_results[rank_col],
                extended_mask,
                strict=True,
            )
        ]

    extended_results = extended_results.loc[
        pd.concat(extended_suggestion_masks, axis=1).any(axis=1)
    ].copy()
    matching_rank_cols = [
        extended_results[f"same_score_at_limit_and_rank_{num_chars}chars"]
        for num_chars in characters
    ]
    tied_results = extended_results[
        pd.concat(matching_rank_cols, axis=1).any(axis=1)
    ].reset_index(drop=True)
    return extended_results, tied_results


# %%
test_df = pd.read_parquet(
    f"gs://{bucket_name}/evaluation-pipeline/original_datasets/sic_2k/sic_2k_test_data.parquet"
)

is_self_employed = test_df["sic2007_employee"] == "-9"

test_df["full_entry"] = np.where(
    is_self_employed,
    test_df["sic2007_self_employed"],
    test_df["sic2007_employee"],
)
test_df["employment_status"] = np.where(is_self_employed, "self_employed", "employed")

test_df = test_df.rename(columns={"clerical_codes": CORRECT_CODE_COL})

# %%
sayt_df = pd.read_csv(
    f"gs://{bucket_name}/sic_knowledgebase/sic_kb_for_sayt.csv", dtype=str
)
sayt_corpus = build_sayt_corpus_from_df(
    df=sayt_df,
    search_text_col="search_text",
    display_text_col="display_text",
    code_col="code",
)[1]

# %%
# define suggesters to evaluate
suggesters = {
    "equal_weights": build_lookup_suggester(
        sayt_corpus,
        ngram_weights=NGRAM_WEIGHT,
        prefix_weights=PREFIX_WEIGHT,
        semantic_weights=SEMANTIC_WEIGHT,
    ),
}


# %%
test_df_hard_limit, avg_ms_dict_hard = get_suggestions_for_limit(
    test_df, suggesters, hard_limit=True
)
test_df_no_hard_limit, avg_ms_dict_no_hard = get_suggestions_for_limit(
    test_df, suggesters, hard_limit=False
)
# %%

ties_no_hard = test_df_no_hard_limit[
    test_df_no_hard_limit[
        f"scores_{CHARACTERS_LIST[0]}chars_{next(iter(suggesters.keys()))}"
    ].str.len()
    > MAX_SUGGESTIONS
].reset_index(drop=True)

# %%
suggestions_cols_to_compare = test_df_hard_limit.columns[
    test_df_hard_limit.columns.str.startswith("suggestions_")
].tolist()

# %%
for suggestions_col in suggestions_cols_to_compare:
    metrics_hard = compute_performance_metrics_from_suggestions(
        test_df_hard_limit,
        correct_codes_col=CORRECT_CODE_COL,
        suggestions_col=suggestions_col,
        code_type=CODE_TYPE,
        k_values=[1, 3, 5, MAX_SUGGESTIONS],
        ave_time_per_query=avg_ms_dict_hard.get(suggestions_col, 0),
    )
    metrics_no_hard = compute_performance_metrics_from_suggestions(
        test_df_no_hard_limit,
        correct_codes_col=CORRECT_CODE_COL,
        suggestions_col=suggestions_col,
        code_type=CODE_TYPE,
        k_values=[1, 3, 5, MAX_SUGGESTIONS],
        ave_time_per_query=avg_ms_dict_no_hard.get(suggestions_col, 0),
    )
    print(metrics_hard.report_metrics())
    print(metrics_no_hard.report_metrics())

# %%
ranks = test_df_no_hard_limit.columns[
    test_df_no_hard_limit.columns.str.startswith("rank_")
].tolist()

nan_counts_per_char_retriever = {}
for column in ranks:
    nan_counts_per_char_retriever[column] = int(
        test_df_no_hard_limit[column].isna().sum()
    )

plot_rank_distributions(test_df_hard_limit, ranks)


# %%
# Unmatched queries for specific characters, with and without hard limit
for metric_column_name in suggestions_cols_to_compare:
    metrics_hard = compute_performance_metrics_from_suggestions(
        test_df_hard_limit,
        correct_codes_col=CORRECT_CODE_COL,
        suggestions_col=metric_column_name,
        code_type=CODE_TYPE,
        k_values=[1, 3, 5, MAX_SUGGESTIONS],
        ave_time_per_query=avg_ms_dict_hard.get(metric_column_name, 0),
    )
    metrics_no_hard = compute_performance_metrics_from_suggestions(
        test_df_no_hard_limit,
        correct_codes_col=CORRECT_CODE_COL,
        suggestions_col=metric_column_name,
        code_type=CODE_TYPE,
        k_values=[1, 3, 5, MAX_SUGGESTIONS],
        ave_time_per_query=avg_ms_dict_no_hard.get(metric_column_name, 0),
    )
    print("UNMATCHED QUERIES WITH HARD LIMIT")
    print(metrics_hard.suggestions_col)
    print(metrics_hard.unmatched_query_count)
    print()
    print("UNMATCHED QUERIES NO HARD LIMIT")
    print(metrics_no_hard.suggestions_col)
    print(metrics_no_hard.unmatched_query_count)
    print()
print(MAX_SUGGESTIONS)

# %%
for reported_column_name in suggestions_cols_to_compare:
    print(
        reported_column_name,
        sum(test_df_no_hard_limit[reported_column_name].apply(len) > MAX_SUGGESTIONS),
    )

# %%
extended, same_rank = find_extended_suggestion_ties(
    test_df_no_hard_limit,
    CHARACTERS_LIST,
    next(iter(suggesters)),
    MAX_SUGGESTIONS,
)
print(same_rank)
# %%
