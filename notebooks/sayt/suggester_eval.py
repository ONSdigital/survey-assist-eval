"""Evaluation tools for suggesters."""

import os

import pandas as pd

from notebooks.sayt.sayt_utils import (
    create_figure,
    get_suggestions_by_chars,
    melt_results_for_analysis,
)
from survey_assist_eval.evaluation.sayt.performance_metrics_functions import (
    build_sayt_metrics_comparison_table,
)


def run_eval_for_suggesters(  # noqa: PLR0913 pylint: disable=R0913, R0914
    *,
    df: pd.DataFrame,
    correct_codes_col: str,
    suggesters_dict: dict,
    num_chars: list[int],
    output_dir: str,
    code_length: int = 5,
    suggestions_list: list | None = None,
    suggestions_limit: int = 9,
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

    Return:
        pd.DataFrame: dataframe containing suggestions.
        figure: plot showing distribution of ranks by suggestions and number of characters.
        pd.DataFrame: dataframe with performance metrics.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if suggestions_list is None:
        suggestions_list = range(1, suggestions_limit)

    df_copy = df.copy()

    suggestions_df, avg_ms_dict = get_suggestions_by_chars(
        df_copy,
        suggesters_dict=suggesters_dict,
        num_chars=num_chars,
        suggestions_limit=suggestions_limit,
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
    )

    return suggestions_df, fig, compare_performance_metrics
