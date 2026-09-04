"""Evaluation tools for suggesters."""

import os

import pandas as pd
from survey_assist_utils.logging import get_logger

from notebooks.sayt.sayt_utils import (
    create_figure,
    get_suggestions_by_chars,
    melt_results_for_analysis,
)
from survey_assist_eval.evaluation.sayt.performance_metrics_functions import (
    build_sayt_metrics_comparison_table,
)
from survey_assist_eval.evaluation.sayt.suggestion_ranking_functions import (
    is_correct_codes_empty,
)

logger = get_logger(__name__)


def run_eval_for_suggesters(  # noqa: PLR0913 pylint: disable=R0913, R0914
    *,
    df: pd.DataFrame,
    correct_codes_col: str,
    suggesters_dict: dict,
    num_chars: list[int],
    output_dir: str,
    code_length: int = 5,
    code_digit_match_length: int | None = None,
    suggestions_list: list | None = None,
    suggestions_limit: int = 9,
    hard_suggestions_limit: bool = False,
    only_unambiguous_correct_codes: bool = False,
):
    """Use functions necessary to create a dataframe that allows for grouping by
        rank and suggester type. Create plots.

    Args:
        df (pd.DataFrame): dataframe to be tested.
        correct_codes_col (str): name of the column containing correct codes.
        suggesters_dict (dict): a dictionary with suggester models.
        num_chars (list): number of characters to be tested.
        suggestions_limit: the maximum rank of suggestions considered as valid.
        output_dir (str): path to file location to be saved.
        code_length (int): expected SIC/SOC code length.
        suggestions_list (list): optional variable; list of suggestions to be checked.
        hard_suggestions_limit (bool): whether to enforce a hard limit on the number of suggestions.
        only_unambiguous_correct_codes (bool): whether to only consider rows
            with unambiguous correct codes.
        code_digit_match_length (int | None): Length of the code digit match
            to consider (default is None).

    Return:
        pd.DataFrame: dataframe containing suggestions.
        figure: plot showing distribution of ranks by suggestions and number of characters.
        pd.DataFrame: dataframe with performance metrics.
    """
    logger.info(
        "Runnning evaluation of SAYT Suggesters",
        num_chars=list(num_chars),
        suggestions_limit=suggestions_limit,
        correct_codes_col=correct_codes_col,
        hard_suggestions_limit=hard_suggestions_limit,
        code_digit_match_length=code_digit_match_length,
        only_unambiguous_correct_codes=only_unambiguous_correct_codes,
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if suggestions_list is None:
        suggestions_list = range(1, suggestions_limit)

    df_copy = df.copy()

    if only_unambiguous_correct_codes:
        rows_before = len(df_copy)

        no_ground_truth = df_copy[correct_codes_col].apply(is_correct_codes_empty)
        df_copy = df_copy.loc[~no_ground_truth].copy()
        dropped_no_ground_truth = int(no_ground_truth.sum())

        only_unambiguous = df_copy[correct_codes_col].apply(
            lambda x: isinstance(x, str) or len(x) == 1
        )
        df_copy = df_copy.loc[only_unambiguous].copy()
        dropped_ambiguous = int((~only_unambiguous).sum())

        logger.info(
            "Filtered only unambiguous correct codes",
            correct_codes_col=correct_codes_col,
            rows_before=rows_before,
            rows_after=len(df_copy),
            dropped_no_ground_truth=dropped_no_ground_truth,
            dropped_ambiguous=dropped_ambiguous,
            dropped_total=rows_before - len(df_copy),
        )

    suggestions_df, avg_ms_dict = get_suggestions_by_chars(
        df_copy,
        suggesters_dict=suggesters_dict,
        correct_codes_col=correct_codes_col,
        code_length=code_length,
        code_digit_match_length=code_digit_match_length,
        num_chars=num_chars,
        suggestions_limit=suggestions_limit,
        hard_suggestions_limit=hard_suggestions_limit,
    )

    melt = melt_results_for_analysis(df=suggestions_df)

    fig = create_figure(melt, output_dir=output_dir)

    suggestions_cols_to_compare = suggestions_df.columns[
        suggestions_df.columns.str.startswith("suggestions_")
    ].tolist()

    compare_performance_metrics = build_sayt_metrics_comparison_table(
        suggestions_df,
        suggestions_cols_to_compare=suggestions_cols_to_compare,
        correct_codes_col=correct_codes_col,
        code_length=code_length,
        k_values=suggestions_list,
        ave_time_per_query_dict=avg_ms_dict,
        code_digit_match_length=code_digit_match_length,
    )

    return suggestions_df, fig, compare_performance_metrics
