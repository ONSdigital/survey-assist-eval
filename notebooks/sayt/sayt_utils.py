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
from survey_assist_eval.evaluation.sayt.suggestion_ranking_functions import (
    get_codes_from_suggestions,
    get_rank_of_first_matching_code,
    is_correct_codes_empty,
    truncate_codes_columns,
)

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


def pad_code_with_leading_zero(code: str, expected_length: int = 5) -> str:
    """Pad a code string with leading zero if needed to match expected length.

    Args:
        code: Code string to normalize.
        expected_length: Expected length of the code after padding.

    Returns:
        str: Code string padded with leading zero, or original code if already correct length
            or if padding is not possible.
    """
    if pd.isna(code):
        return code

    code_str = str(code)
    if len(code_str) == expected_length:
        return code_str
    if len(code_str) == expected_length - 1:
        return f"0{code_str}"
    if len(code_str) < expected_length - 1:
        logger.warning(
            "SIC code shorter than expected_length - 1; leaving unchanged",
            code=code_str,
            expected_length=expected_length,
            observed_length=len(code_str),
        )
    return code_str


def add_display_text_with_code(
    df: pd.DataFrame,
    text_col: str,
    code_col: str,
    output_col: str = "display_text_with_code",
    separator: str = ": ",
) -> pd.DataFrame:
    """Return a copy of df with a combined display text + code column.

    Args:
        df: Input DataFrame containing text and code columns.
        text_col: Column containing the human-readable text portion.
        code_col: Column containing the code portion.
        output_col: Name of the output combined column.
        separator: Separator string placed between text and code.

    Returns:
        pd.DataFrame: Copy of df with `output_col` added.
    """
    if text_col not in df.columns:
        raise KeyError(f"Missing required text column: {text_col}")
    if code_col not in df.columns:
        raise KeyError(f"Missing required code column: {code_col}")

    output_df = df.copy()
    output_df[output_col] = output_df[text_col] + separator + output_df[code_col]
    return output_df


def build_sayt_corpus_from_df(  # noqa: PLR0913, pylint: disable=R0917,R0913
    df: pd.DataFrame,
    search_text_col: str,
    display_text_col: str,
    code_col: str = "code",
    expected_code_length: int = 5,
    incl_code_in_display: bool = True,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Build a SAYT corpus from a DataFrame.

    Normalises codes using `pad_code_with_leading_zero`, optionally appends
    codes to display text, and returns both the updated DataFrame and the
    resulting SAYT corpus.

    Args:
        df: Input DataFrame containing the required columns.
        search_text_col: Column containing searchable text.
        display_text_col: Column containing display text.
        code_col: Column containing codes to normalise.
        expected_code_length: Expected code length used for normalisation.
        incl_code_in_display: Whether to append codes to display text.

    Returns:
        tuple[pd.DataFrame, list[tuple[str, str]]]:
            Updated DataFrame and corpus as `(search_text, display_text)` tuples.
    """
    output_df = df.copy()

    output_df[code_col] = output_df[code_col].apply(
        pad_code_with_leading_zero, expected_length=expected_code_length
    )

    final_display_col = display_text_col

    if incl_code_in_display:
        final_display_col = f"{display_text_col}_with_code"
        output_df = add_display_text_with_code(
            output_df,
            text_col=display_text_col,
            code_col=code_col,
            output_col=final_display_col,
            separator=": ",
        )

    corpus = list(
        zip(output_df[search_text_col], output_df[final_display_col], strict=False)
    )

    return output_df, corpus


def get_suggestions_for_row(  # noqa: PLR0913 pylint: disable=R0917,R0913
    row: pd.Series,
    suggester: Any,
    num_chars: int,
    max_suggestions: int,
    query_col: str = "full_entry",
    hard_suggestions_limit: bool = False,
    with_scores: bool = False,
) -> list[str] | tuple[list[str], list[float]]:
    """Return suggester output for a single input row.

    Args:
        row: Input row containing a text field specified by `query_col`.
        suggester: Suggester object exposing a `suggest` method.
        num_chars: Number of leading characters from `query_col` to use as input.
        max_suggestions: Maximum number of suggestions to request.
        query_col: Column name containing the text to be used for suggestions.
        hard_suggestions_limit: If True, limit the number of suggestions to the
            specified max_suggestions, otherwise allow more suggestions to be
             returned based on score.
        with_scores: If True, return suggestions with scores instead of just strings.

    Returns:
        list[str] | tuple[list[str], list[float]]: Ordered suggestion strings,
            or suggestions and scores when with_scores is True.
    """
    query_text = row[query_col][:num_chars]

    if with_scores:
        suggestions_with_scores = suggester.suggest_with_scores(
            query_text,
            num_suggestions=max_suggestions,
        )
        if hard_suggestions_limit:
            suggestions_with_scores = suggestions_with_scores[:max_suggestions]

        display_text = [s.display_text for s in suggestions_with_scores]
        scores = [s.score for s in suggestions_with_scores]
        return display_text, scores

    suggestions = suggester.suggest(
        query_text,
        num_suggestions=max_suggestions,
    )
    if hard_suggestions_limit:
        suggestions = suggestions[:max_suggestions]

    return suggestions


def get_suggestions_by_chars(  # noqa: PLR0913 pylint: disable=R0917,R0913,R0914
    df: pd.DataFrame,
    suggesters_dict: dict[str, Any],
    correct_codes_col: str = "correct_sic_code",
    code_length: int = 5,
    code_digit_match_length: int | None = None,
    num_chars: list | None = None,
    suggestions_limit: int = 9,
    hard_suggestions_limit: bool = False,
    with_scores: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Gathers suggestions for specified number of characters using suggesters.

    Args:
        df: dataframe containing melted suggestions.
        correct_codes_col: name of the column containing correct codes.
        code_length: expected SIC/SOC code length.
        code_digit_match_length: Length of the code digit match to consider (default is None).
        num_chars: number of characters to be tested.
        suggesters_dict: a dictionary with initialised suggester models.
        suggestions_limit: the maximum rank of suggestions considered as valid.
        hard_suggestions_limit: if True, limit the number of suggestions to the
            specified suggestions_limit, otherwise allow more suggestions to be returned.
        with_scores: if True, return suggestions with scores instead of just strings.

    Returns:
        tuple[pd.DataFrame, dict]: Suggestions for specified characters typed;
            average milliseconds per row for each suggester.
    """
    df = df.copy()

    avg_ms_dict = {}
    if num_chars is None:
        num_chars = [4, 5, 7, 10]
    for prefix_chars in num_chars:
        for suggester_name, suggester_obj in suggesters_dict.items():
            retrieved_codes_col = "_retrieved_codes"
            correct_codes_col_ = correct_codes_col

            logger.info(
                "Starting SAYT suggesting - one loop",
                num_chars=prefix_chars,
                suggester_label=suggester_name,
            )

            suggestions_col = f"suggestions_{prefix_chars}chars_{suggester_name}"
            scores_col = suggestions_col.replace("suggestions_", "scores_")

            suggestions_result, avg_ms = timed_apply(
                df,
                get_suggestions_for_row,
                suggester=suggester_obj,
                max_suggestions=suggestions_limit,
                num_chars=prefix_chars,
                hard_suggestions_limit=hard_suggestions_limit,
                with_scores=with_scores,
                axis=1,
            )

            if with_scores:
                df[[suggestions_col, scores_col]] = pd.DataFrame(
                    suggestions_result.tolist(),
                    index=df.index,
                )
            else:
                df[suggestions_col] = suggestions_result

            logger.info("  -> suggestions done", elapsed_sec=avg_ms)

            df[retrieved_codes_col] = df.apply(
                get_codes_from_suggestions,
                code_length=code_length,
                suggestions_col=suggestions_col,
                axis=1,
            )

            rank_col_name = f"rank_{prefix_chars}chars_{suggester_name}"

            if code_digit_match_length is not None:
                df = truncate_codes_columns(
                    df,
                    code_digit_match_length,
                    correct_codes_col=correct_codes_col,
                    retrieved_codes_col=retrieved_codes_col,
                )

                retrieved_codes_col = f"{retrieved_codes_col}_truncated"
                correct_codes_col_ = f"{correct_codes_col_}_truncated"
                rank_col_name = f"{rank_col_name}_{code_digit_match_length}digitmatch"

                logger.info(
                    f"Computing rank for {code_digit_match_length}-digit match",
                    code_digit_match_length=code_digit_match_length,
                    rank_col_name=rank_col_name,
                )

            df[rank_col_name] = df.apply(
                lambda row, rc=retrieved_codes_col, cc=correct_codes_col_: (
                    get_rank_of_first_matching_code(row[rc], row[cc])
                ),
                axis=1,
            )

            avg_ms_dict.update({suggestions_col: avg_ms})

    return df, avg_ms_dict


def melt_results_for_analysis(
    df: pd.DataFrame,
    suggestions_limit: int = 9,
    correct_code_column: str = "correct_sic_code",
) -> pd.DataFrame:
    """Melt results by suggester and num_chars for easier analysis.

    Args:
        df: dataframe containing columns 'correct_sic_code' and 'full_entry' with rank columns.
        suggestions_limit: the maximum rank of suggestions considered as valid.
        correct_code_column: a column name with correct codes.

    Returns:
        pd.DataFrame: dataframe with results form suggesters, split by the type of suggester
            and number of characters.
    """
    no_ground_truth = df[correct_code_column].apply(is_correct_codes_empty)

    df = df.loc[~no_ground_truth].copy()

    results_df = df.melt(
        id_vars=[correct_code_column, "full_entry"],
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
    suggestions_limit: int = 9,
):
    """Compare rank histograms for suggesters at different num_chars.

    Args:
        df: dataframe containing results created by suggesters split by number of characters.
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

    fig.write_html(f"{output_dir}/sayt_eval_100sample_rank_histograms.html")

    return fig


def timed_apply(df: pd.DataFrame, func, **kwargs) -> tuple[pd.Series, float]:
    """Run df.apply and return the results alongside the average time per row.

    Args:
        df: DataFrame to apply the function to.
        func: Callable to apply row-wise.
        **kwargs: Additional keyword arguments passed to df.apply.

    Returns:
        tuple[pd.Series, float]: Apply results and average milliseconds per row.
    """
    t_start = time.perf_counter()
    results = df.apply(func, **kwargs)
    avg_ms = (time.perf_counter() - t_start) / len(df) * 1000
    return results, avg_ms
