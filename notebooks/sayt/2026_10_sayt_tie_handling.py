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
from IPython.display import display

from notebooks.sayt.sayt_utils import (
    build_lookup_suggester,
    build_sayt_corpus_from_df,
    get_suggestions_by_chars,
)
from survey_assist_eval.evaluation.sayt.performance_metrics_functions import (
    build_sayt_metrics_comparison_table,
)

# %%
CODE_TYPE = "sic"
CORRECT_CODE_COL = "correct_sic_code"
MAX_SUGGESTIONS = [9, 20, 25, 30]
HIGH_CHARACTERS = True  # High characters: 6-9, low characters: 4-5

if HIGH_CHARACTERS:
    CHARACTERS_LIST = list(range(6, 10))
    NGRAM_WEIGHT = 3
    PREFIX_WEIGHT = 4
    SEMANTIC_WEIGHT = 3
else:
    CHARACTERS_LIST = list(range(4, 6))
    NGRAM_WEIGHT = 4
    PREFIX_WEIGHT = 5
    SEMANTIC_WEIGHT = 1

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")


# %%
def get_suggestions_for_limit(
    evaluation_data: pd.DataFrame,
    suggester_map: dict[str, Any],
    suggestions_limit: int,
    *,
    hard_limit: bool,
) -> tuple[pd.DataFrame, dict]:
    """Generate scored suggestions using the requested limit behaviour.

    Args:
        evaluation_data: Queries and evaluation data to generate suggestions for.
        suggester_map: Named suggesters to use for suggestion generation.
        suggestions_limit: the maximum length of the suggestions list.
        hard_limit: Whether to enforce the suggestion limit strictly.

    Returns:
        A tuple containing the dataframe augmented with suggestions and scores,
        and a dictionary of average query times keyed by suggestion column.
    """
    return get_suggestions_by_chars(
        df=evaluation_data,
        suggesters_dict=suggester_map,
        num_chars=CHARACTERS_LIST,
        suggestions_limit=suggestions_limit,
        hard_suggestions_limit=hard_limit,
        with_scores=True,
    )


def plot_rank_distributions(
    evaluation_data: pd.DataFrame,
    plot_title: str,
) -> None:
    """Display a histogram of ranks for the supplied result columns.

    Args:
        evaluation_data: Results dataframe containing the rank columns.
        plot_title: Title for the plot.

    Returns:
        None.
    """
    rank_columns = evaluation_data.columns[
        evaluation_data.columns.str.startswith("rank_")
    ].tolist()

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
        title=plot_title,
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
        suggestion_column_name = (
            f"suggestions_{num_chars}chars_{suggester_name}_no_hard"
        )
        scores_col = f"scores_{num_chars}chars_{suggester_name}_no_hard"
        rank_col = f"rank_{num_chars}chars_{suggester_name}_5digitmatch_no_hard"

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


# %% Read data
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
# Access the knowledgebase
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
# Define suggesters to evaluate
suggesters = {
    "suggester": build_lookup_suggester(
        sayt_corpus,
        ngram_weights=NGRAM_WEIGHT,
        prefix_weights=PREFIX_WEIGHT,
        semantic_weights=SEMANTIC_WEIGHT,
    ),
}

# %%
results_by_limit = {}

for max_sug in MAX_SUGGESTIONS:
    # Create two dataframes, one with hard_limit, another without.
    test_df_hard_limit, avg_ms_dict_hard = get_suggestions_for_limit(
        test_df, suggesters, hard_limit=True, suggestions_limit=max_sug
    )
    test_df_no_hard_limit, avg_ms_dict_no_hard = get_suggestions_for_limit(
        test_df, suggesters, hard_limit=False, suggestions_limit=max_sug
    )

    # Get rows where the suggestions list was extended beyond the limit.
    ties_no_hard = test_df_no_hard_limit[
        test_df_no_hard_limit[
            f"scores_{CHARACTERS_LIST[0]}chars_{next(iter(suggesters.keys()))}"
        ].str.len()
        > max_sug
    ].reset_index(drop=True)

    # Compare metrics for hard- and no-hard-limit suggestions.
    test_df_metrics = pd.merge(
        test_df_hard_limit,
        test_df_no_hard_limit,
        on=["unique_id"],
        suffixes=("_hard", "_no_hard"),
    )
    suggestions_cols_to_compare = test_df_metrics.columns[
        test_df_metrics.columns.str.startswith("suggestions_")
    ].tolist()

    avg_ms_dict_combined = {
        **{f"{key}_hard": value for key, value in avg_ms_dict_hard.items()},
        **{f"{key}_no_hard": value for key, value in avg_ms_dict_no_hard.items()},
    }
    metrics_overall = build_sayt_metrics_comparison_table(
        df=test_df_metrics,
        suggestions_cols_to_compare=suggestions_cols_to_compare,
        correct_codes_col=f"{CORRECT_CODE_COL}_hard",
        ave_time_per_query_dict=avg_ms_dict_combined,
        code_type=CODE_TYPE,
        code_digit_match_length=None,
        k_values=[1, 5, max_sug],
    ).sort_values(["suggestions_col"])
    metrics_overall["matched_query_count"] = (
        metrics_overall["queries_with_ground_truth"]
        - metrics_overall["unmatched_query_count"]
    )
    print(f"Suggestion limit: {max_sug}")
    display(
        metrics_overall[["suggestions_col", "matched_query_count"]].reset_index(
            drop=True
        )
    )

    plot_rank_distributions(test_df_hard_limit, "Rank distributions (with hard limit)")
    plot_rank_distributions(
        test_df_no_hard_limit, "Rank distributions (without hard limit)"
    )

    for reported_column_name in suggestions_cols_to_compare:
        extended_count = (
            test_df_metrics[reported_column_name].apply(len) > max_sug
        ).sum()
        print(
            f"# rows with extended list of suggestions at {reported_column_name} with {max_sug} suggestions limit: "  # pylint: disable=C0301
            f"{extended_count}"
        )

    extended, same_rank = find_extended_suggestion_ties(
        test_df_metrics,
        CHARACTERS_LIST,
        next(iter(suggesters)),
        max_sug,
    )
    same_score_columns = extended.columns[
        extended.columns.str.startswith("same_score_at_limit_and_rank_")
    ].tolist()

    rank_greater_than_max_sug_columns = extended.columns[
        extended.columns.str.startswith("rank_gt_max_suggestions_")
    ].tolist()
    cols_to_display = same_score_columns + rank_greater_than_max_sug_columns
    display(same_rank[cols_to_display])

    results_by_limit[max_sug] = {
        "test_df_hard_limit": test_df_hard_limit,
        "test_df_no_hard_limit": test_df_no_hard_limit,
        "ties_no_hard": ties_no_hard,
        "test_df_metrics": test_df_metrics,
        "metrics_overall": metrics_overall,
        "extended": extended,
        "same_rank": same_rank,
    }
# %%
