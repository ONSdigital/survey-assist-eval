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
    compute_performance_metrics_from_suggestions,
)


def run_eval_for_suggesters(  # noqa: PLR0913 pylint: disable=R0913, R0914
    *,
    df: pd.DataFrame,
    correct_code_col: str,
    suggesters_dict: dict,
    num_chars: list[int],
    output_dir: str,
    suggestions_limit: int = 9,
    code_length: int = 5,
):
    """Use functions necessary to create a dataframe that allows for grouping by
        rank and suggester type. Create plots.

    Args:
        df (pd.DataFrame): dataframe to be tested.
        correct_code_col (str): name of the column containing correct codes.
        suggesters_dict (dict): a dictionary with suggester models.
        num_chars (list): number of characters to be tested.
        suggestions_limit: the maximum rank of suggestions considered as valid.
        code_length (int): expected SIC/SOC code length.
        output_dir (str): path to file location to be saved.

    Return:
        pd.DataFrame: dataframe with results from suggesters, split by the type of suggester
            and number of characters.
        pd.DataFrame: dataframe containing suggestions.
        dict: dictionary containing performance metrics for each suggester.
        figure: plot showing distribution of ranks by suggestions and number of characters.
        pd.DataFrame: dataframe with performance metrics.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    metrics_dict = {}

    # Performance metrics for one suggester and prefix length
    for _, suggester in enumerate(suggestions_cols_to_compare):
        metrics = compute_performance_metrics_from_suggestions(
            suggestions_df,
            correct_code_col=correct_code_col,
            suggestions_col=suggester,
            code_length=code_length,
            k_values=num_chars,
            ave_time_per_query=avg_ms_dict.get(
                suggester.removeprefix("suggestions_"), 0
            ),
        )
        metrics_dict[suggester] = metrics.report_metrics()

    ave_elapsed_per_row_list = [
        avg_ms_dict.get(col.removeprefix("suggestions_"), 0)
        for col in suggestions_cols_to_compare
    ]

    compare_performance_metrics = build_sayt_metrics_comparison_table(
        suggestions_df,
        suggestions_cols_to_compare=suggestions_cols_to_compare,
        correct_code_col=correct_code_col,
        k_values=num_chars,
        ave_time_per_query_list=ave_elapsed_per_row_list,
    )

    return suggestions_df, metrics_dict, fig, compare_performance_metrics
