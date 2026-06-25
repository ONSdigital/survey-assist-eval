"""Functions for filtering and preparing open question data for evaluation."""

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
