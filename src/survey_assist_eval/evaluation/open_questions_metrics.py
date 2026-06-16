"""Calculation of metrics for open questions."""

import pandas as pd


def filter_nonempty_object_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return rows where the object column is not NA and not an empty string.

    Args:
        df: DataFrame containing the target column.
        column: Name of the object column to filter.

    Returns:
        A DataFrame containing only rows where the specified column is non-null
        and has length greater than zero.
    """
    mask = df[column].notna() & df[column].astype(str).str.len().gt(0)
    return df[mask].copy()


def count_chars_in_column(
    df: pd.DataFrame, column: str, result_col: str | None = None
) -> pd.Series:
    """Return per-row character counts for the given DataFrame column.

    Args:
        df: DataFrame containing the target column.
        column: Name of the column to count characters for.
        result_col: Optional column name to write the counts back into the DataFrame.

    Returns:
        A Series of integer character counts for each row in the specified column.
    """
    counts = df[column].fillna("").astype(str).str.len()
    if result_col:
        df[result_col] = counts
    return counts


def count_words_in_column(
    df: pd.DataFrame, column: str, result_col: str | None = None
) -> pd.Series:
    """Return per-row word counts for the given DataFrame column.

    Args:
        df: DataFrame containing the target column.
        column: Name of the object column to count words for.
        result_col: Optional column name to write the counts back into the DataFrame.

    Returns:
        A Series of integer word counts for each row in the specified column.
    """
    counts = df[column].fillna("").astype(str).str.split().str.len()
    if result_col:
        df[result_col] = counts
    return counts
