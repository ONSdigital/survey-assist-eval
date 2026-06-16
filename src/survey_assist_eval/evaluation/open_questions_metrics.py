"""Calculation of metrics for open questions."""

import nltk
import pandas as pd
from nltk.corpus import cmudict

nltk.download("cmudict")


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


def lookup_word(word: str, first: bool = False):
    """Quick lookup helper for the NLTK CMU Pronouncing Dictionary.

    Returns the pronunciations for `word` from the cached `cmu_dict`.
    If ``first`` is True, returns only the first pronunciation (a list of
    ARPAbet symbols). Returns ``None`` if the word is not found or input is empty.
    """
    if not word:
        return None
    key = word.lower().strip()
    prons = cmudict.dict().get(key)
    if not prons:
        return None
    return prons[0] if first else prons


def count_syllables(word: str) -> int | None:
    """Return syllable count for `word` using the CMU Pronouncing Dictionary.

    This simple implementation uses the first pronunciation entry from
    `cmu_dict[word]` (if present) and counts phonemes that include a
    stress digit (the CMU convention for vowel phonemes, e.g. 'AH0', 'AE1').

    Args:
        word: The input word to count syllables for.

    Returns:
        The estimated syllable count as an int, or ``None`` if the word is
        not found in the CMU dictionary or the input is empty.
    """
    if not word:
        return None

    prons = lookup_word(word=word, first=False)

    if not prons:
        return None

    if len(prons) == 1:  # only one pronounciation
        # use stress point digits to count syllables
        syll_count = sum(1 for phoneme in prons[0] if phoneme[-1].isdigit())
    else:
        # multiple pronunciations, get all counts
        syll_counts = [
            sum(1 for phoneme in pron if phoneme[-1].isdigit()) for pron in prons
        ]

        if all(syll_count == syll_counts[0] for syll_count in syll_counts):
            syll_count = syll_counts[0]
        else:
            # if multiple pronunciations have different syllable counts, return None
            return None

    return syll_count
