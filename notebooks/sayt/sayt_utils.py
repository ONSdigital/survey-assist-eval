"""Utility functions for SAYT evaluation."""

import time
from typing import Any

import pandas as pd
import plotly.express as px
from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SAYTSuggester,
    SemanticRetrieverSpec,
)
from survey_assist_utils.logging import get_logger

from survey_assist_eval.data_cleaning.code_standard import get_clean_n_digit_codes

logger = get_logger(__name__)


def build_lookup_suggester(
    corpus: list[tuple[str, str]],
    *,
    retrievers: list | None = None,
    semantic_weight: float | None = None,
) -> SAYTSuggester:
    """Build a lookup suggester using the explicit retriever-spec API.

    Args:
        corpus: Search corpus as (search_text, display_text) tuples.
        semantic_weight: Weight for semantic retrieval. If None, semantic retrieval
            is not included.
        retrievers: list of retrievers to be used.

    Returns:
        SAYTSuggester: Configured suggester instance.
    """
    if retrievers is None:
        retrievers = [PrefixRetrieverSpec(), NgramRetrieverSpec()]

    if semantic_weight is not None:
        retrievers.append(SemanticRetrieverSpec(weight=semantic_weight))
    return SAYTSuggester(corpus, retrievers=retrievers)


def validate_one_code(code: str, code_length=5) -> bool:
    """Validate one SIC code and log malformed values.

    Args:
        code: SIC code value to validate.
        logger: Logger used for warning messages.
        code_length: Expected SIC code length.

    Returns:
        bool: True when the code is valid and unchanged after cleaning, else False.
    """
    if pd.isna(code):
        logger.warning("Code is NaN")
        return False
    clean_codes = get_clean_n_digit_codes(code, n=code_length, code_type="SIC")
    if len(clean_codes[1]) != 0:
        logger.warning(f"Malformed code: {code}")
        return False
    if len(clean_codes[0]) != 1 or next(iter(clean_codes[0])) != code:
        logger.warning(f"Code {code} cleaned to different code: {clean_codes[0]}")
        return False
    return True


def get_suggestions_for_row(
    row: dict[str, Any],
    suggester: Any,
    num_chars: int,
    max_suggestions: int,
) -> list[str]:
    """Return suggester output for a single input row.

    Args:
        row: Input row containing a `full_entry` text field.
        suggester: Suggester object exposing a `suggest` method.
        num_chars: Number of leading characters from `full_entry` to use as input.
        max_suggestions: Maximum number of suggestions to request.

    Returns:
        list[str]: Ordered suggestion strings returned by the suggester.
    """
    return suggester.suggest(
        row["full_entry"][:num_chars],
        num_suggestions=max_suggestions,
    )


def rank_of_correct_code_in_suggestions(
    row: dict[str, Any],
    num_chars: int,
    suggester_label: str,
    code_length: int = 5,
    correct_code_col: str = "correct_sic_code",
) -> int | None:
    """Return the rank of the correct code in generated suggestions.

    Args:
        row: Input row containing suggestion outputs and the correct code.
        num_chars: Prefix length used to generate suggestions.
        suggester_label: Label used in the suggestion column name.
        code_length: Number of trailing characters to compare as code.
        correct_code_col: Column name holding the correct SIC code.

    Returns:
        int | None: 1-based rank of the correct code, or None if not found.
    """
    correct_code = row[correct_code_col]
    suggestions = row[f"suggestions_{num_chars}chars_{suggester_label}"]
    for rank, suggest in enumerate(suggestions):
        if suggest[-code_length:] == correct_code:
            return rank + 1
    return None


def get_suggestions_for_collection(
    df: pd.DataFrame,
    suggesters_dict: dict[str, Any],
    characters: list | None = None,
    suggestions_limit: int | None = 9,
) -> pd.DataFrame:
    """Gatheres suggestions for specified number of chatracters using suggesters.

    Args:
        df: dataframe containing melted suggestions.
        characters: number of characters to be tested.
        suggesters_dict: a dictionary with initialised suggester models.
        suggestions_limit: the maximum rank of suggestions considered as valid.

    """
    if characters is None:
        characters = [4, 5, 7, 10]
    for prefix_chars in characters:  # 150]:
        for suggester_name, suggester_obj in suggesters_dict.items():
            logger.info(
                "Starting SAYT suggesting - one loop",
                num_chars=prefix_chars,
                suggester_label=suggester_name,
            )

            t_start = time.perf_counter()
            df[f"suggestions_{prefix_chars}chars_{suggester_name}"] = df.apply(
                get_suggestions_for_row,
                suggester=suggester_obj,
                max_suggestions=suggestions_limit,
                num_chars=prefix_chars,
                axis=1,
            )
            elapsed = time.perf_counter() - t_start
            logger.info(
                "  -> suggestions done",
                elapsed_sec=elapsed,
                elapsed_per_row_ms=elapsed / len(df) * 1000,
            )
            df[f"rank_{prefix_chars}chars_{suggester_name}"] = df.apply(
                rank_of_correct_code_in_suggestions,
                correct_code_col="correct_sic_code",
                suggester_label=suggester_name,
                num_chars=prefix_chars,
                axis=1,
            )
    return df


def melt_results_for_analysis(
    df: pd.DataFrame, suggestions_limit: int | None = 9
) -> pd.DataFrame:
    """Melt results by suggester and num_chars for easier analysis.

    Args:
        df: dataframe containing columns 'correct_sic_code' and 'full_entry' with rank columns.
        suggestions_limit: the maximum rank of suggestions considered as valid.

    """
    results_df = df.melt(
        id_vars=["correct_sic_code", "full_entry"],
        value_vars=[col for col in df.columns if col.startswith("rank_")],
        var_name="suggester_numchars",
        value_name="rank",
    )
    results_df["num_chars"] = results_df["suggester_numchars"].apply(
        lambda x: int(x.split("_")[1].replace("chars", ""))
    )
    results_df["suggester"] = results_df["suggester_numchars"].apply(
        lambda x: " ".join(x.split("_")[2:])
    )
    results_df.loc[results_df["rank"] > suggestions_limit, "rank"] = None
    results_df["rank"] = results_df["rank"].fillna(
        suggestions_limit + 2
    )  # Treat 'not found' as worst rank
    results_df["rank"] = results_df["rank"].astype(int)

    results_df = results_df.sort_values(
        by=["num_chars", "suggester", "rank"]
    ).reset_index(drop=True)

    return results_df


def create_figure(
    df: pd.DataFrame,
    output_dir: str,
    suggestions_limit: int | None = 9,
):
    """Compare rank histograms for the two suggesters at different num_chars.

    Args:
        df: dataframe containing.
        output_dir: path to file location to be saved.
        suggestions_limit: the maximum rank of suggestions considered as valid.
    """
    fig = px.histogram(
        df,
        x="rank",
        color="suggester",
        facet_col="num_chars",
        category_orders={
            "rank": list(range(0, suggestions_limit + 2)),
            "suggester": sorted(df["suggester"].unique().tolist()),
        },
        barmode="group",
        title=(
            "Distribution of Ranks of Correct Code in Suggestions by Number of Characters"
            + " (on SAYT test data of 100 examples)"
        ),
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=[*range(1, suggestions_limit + 1, 1), suggestions_limit + 2],
        ticktext=[str(i) for i in range(1, suggestions_limit + 1, 1)] + ["NA"],
    )

    fig.update_layout(
        bargap=0.1,
        legend={
            "title": "Suggester method",
            "orientation": "h",
            "yanchor": "top",
            "y": -0.2,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    fig.show()

    fig.write_html(f"{output_dir}/sayt_eval_100sample_rank_histograms.html")
