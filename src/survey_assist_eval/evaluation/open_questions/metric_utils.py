"""Utility functions for adding computed metric columns to DataFrames."""

from collections.abc import Callable

import pandas as pd


def add_metrics_columns(
    df: pd.DataFrame,
    text_column: str,
    metrics_func: Callable[[str], dict],
    prefix: str | None = None,
) -> pd.DataFrame:
    """Generic wrapper to add computed metric columns to a DataFrame.

    This utility consolidates the common pattern of applying a metrics function
    to a text column and adding the resulting columns to the DataFrame.

    Args:
        df: DataFrame with text data.
        text_column: Column containing text to analyze.
        metrics_func: Callable that takes a string and returns a dict of metrics.
        prefix: Prefix for new columns. Defaults to "{text_column}_".

    Returns:
        DataFrame with added metric columns.
    """
    metrics_df = (
        df[text_column].fillna("").astype(str).apply(metrics_func).apply(pd.Series)
    )

    if prefix is None:
        prefix = f"{text_column}_"

    metrics_df = metrics_df.rename(columns=lambda col: f"{prefix}{col}")

    return df.join(metrics_df)
