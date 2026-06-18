"""Calculation of metrics for open questions."""

import pandas as pd
from textstat import textstat


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


def get_text_stats(text: str) -> dict[str, int | float]:
    """Return basic text stats for one string.

    Args:
        text: Text to analyze.

    Returns:
        A dict with word, sentence, syllable, character, letter and
        average stats.

    Note:
        Sentence counts follow textstat behaviour, where sentences
        with two words or fewer may not be counted.
    """
    return {
        "word_count": textstat.lexicon_count(text, removepunct=True),
        "sentence_count": textstat.sentence_count(text),
        "syllable_count": textstat.syllable_count(text),
        "character_count": textstat.char_count(text),
        "letter_count": textstat.letter_count(text),
        "mean_words_per_sentence": textstat.words_per_sentence(text),
        "mean_syllables_per_word": textstat.avg_syllables_per_word(text),
    }


def add_text_stats_columns(
    df: pd.DataFrame,
    text_column: str,
    result_prefix: str | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Add text stats columns for a DataFrame text column.

    Args:
        df: DataFrame with text data.
        text_column: Column containing text.
        result_prefix: Prefix for new columns. Defaults to "{text_column}_".
        inplace: If True, add columns in place and return the same DataFrame.

    Returns:
        DataFrame with added text stat columns.
    """
    stats_df = (
        df[text_column].fillna("").astype(str).apply(get_text_stats).apply(pd.Series)
    )

    if result_prefix is None:
        result_prefix = f"{text_column}_"

    stats_df = stats_df.rename(columns=lambda col: f"{result_prefix}{col}")

    if inplace:
        df.loc[:, stats_df.columns] = stats_df
        return df

    return df.join(stats_df)


def summarise_text_stats(  # noqa: PLR0913 pylint: disable=R0917, R0913
    df: pd.DataFrame,
    prefix: str | None = None,
    text_column: str | None = None,
    word_threshold: int = 25,
    sentence_threshold: int = 2,
    long_sentence_threshold: int = 20,
    short_word_count_threshold: int = 2,
) -> pd.Series:
    """Summarise text statistics for a DataFrame.

    Args:
        df: DataFrame with text statistic columns already added.
        prefix: Prefix for stat columns (required if stats already exist).
        text_column: If provided, stats will be computed using
            add_text_stats_columns.
        word_threshold: Threshold for "long" text (word count).
        sentence_threshold: Threshold for number of sentences.
        long_sentence_threshold: Threshold for long sentences
            (words per sentence).
        short_word_count_threshold: Threshold for "blank or too short".

    Returns:
        A Series with summary statistics.

    Notes:
        - If text_column is provided, stats are computed internally.
        - If not, prefix must be provided and columns must already exist.
    """
    if text_column is not None:
        if prefix is None:
            prefix = f"{text_column}_"
        df = add_text_stats_columns(
            df.copy(),
            text_column=text_column,
            result_prefix=prefix,
            inplace=False,
        )
    elif prefix is None:
        raise ValueError("Provide either text_column or prefix.")

    summary = {
        "n_outputs": len(df),
        "mean_word_count": df[f"{prefix}word_count"].mean(),
        "median_word_count": df[f"{prefix}word_count"].median(),
        "sd_word_count": df[f"{prefix}word_count"].std(),
        "mean_sentence_count": df[f"{prefix}sentence_count"].mean(),
        "mean_words_per_sentence": df[f"{prefix}mean_words_per_sentence"].mean(),
        "mean_syllables_per_word": df[f"{prefix}mean_syllables_per_word"].mean(),
        f"pct_over_{word_threshold}_words": (
            df[f"{prefix}word_count"] > word_threshold
        ).mean()
        * 100,
        f"pct_over_{sentence_threshold}_sentences": (
            df[f"{prefix}sentence_count"] > sentence_threshold
        ).mean()
        * 100,
        "pct_long_sentence": (
            df[f"{prefix}mean_words_per_sentence"] > long_sentence_threshold
        ).mean()
        * 100,
        "pct_blank_or_too_short": (
            df[f"{prefix}word_count"] <= short_word_count_threshold
        ).mean()
        * 100,
    }

    return pd.Series(summary)
