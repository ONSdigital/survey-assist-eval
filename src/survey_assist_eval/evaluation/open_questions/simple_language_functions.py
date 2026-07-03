"""Functions for checking simple language in open questions."""

import re

import pandas as pd
from pydantic import BaseModel
from textstat import textstat

from survey_assist_eval.evaluation.open_questions.metric_utils import (
    add_metrics_columns,
)


class SimpleLanguageMetrics(BaseModel):
    """Container for all simple language evaluation metrics."""

    n_count: int
    pct_with_acronyms: float
    mean_avg_syllables_per_word: float
    pct_with_word_over_syllables_threshold: float

    def report_metrics(self) -> str:
        """Pretty print the simple language evaluation metrics."""
        lines = [
            "\nSimple language metrics:",
            f" Number of follow-up questions: {self.n_count:.0f}",
            f" Percentage with acronyms: {self.pct_with_acronyms:.2f}%",
            f" Mean average syllables per word: {self.mean_avg_syllables_per_word:.2f}",
            " Percentage with words over syllables threshold: "
            f"{self.pct_with_word_over_syllables_threshold:.2f}%",
        ]
        return "\n".join(lines)


def compute_simple_language_metrics(
    df,
    *,
    text_column: str,
    prefix: str = "eval_",
    syllables_threshold: int = 3,
) -> SimpleLanguageMetrics:
    """Evaluate simple language quality for generated follow-up questions.

    Args:
        df: DataFrame containing generated questions.
        text_column: Column containing the text responses.
        prefix: Prefix for generated metric columns.
        syllables_threshold: Threshold for counting words with high syllable counts.

    Returns:
        SimpleLanguageMetrics: Structured summary of metrics.
    """
    df = add_simple_language_columns(df, text_column=text_column, prefix=prefix)

    metrics = summarise_simple_language_columns(
        df,
        prefix=prefix,
        syllables_threshold=syllables_threshold,
    )

    return SimpleLanguageMetrics(**metrics.to_dict())


def extract_acronyms(text: str, extended: bool = False) -> list[str]:
    """Extract acronyms from a text string.

    Two regex patterns are supported:

    - Simple pattern:
        Matches uppercase tokens of length >= 2, optionally followed by digits.
        Examples: "ONS", "NLP", "ISO9001", "G7"

    - Extended pattern:
        Matches:
            - Uppercase tokens (same as simple pattern)
            - Dotted acronyms (e.g. "U.S.A.", "U.K.")
            - Ampersand acronyms (e.g. "R&D")

    The pattern used is controlled by the ``extended`` flag.

    Args:
        text: Input text to search for acronyms. If the input is not a string,
            an empty list is returned.
        extended: If True, use the extended pattern to capture dotted and
            ampersand acronyms. If False, use the simpler pattern.

    Returns:
        A list of acronyms found in the input text. Returns an empty
        list if no acronyms are found or if the input is not a string.
    """
    if not isinstance(text, str):
        return []
    # simple patterns
    all_caps = r"[A-Z]{2,}[A-Z0-9]*"
    letter_number = r"[A-Z]\d+"

    # extended patterns
    initialisms = r"[A-Z]\.[A-Z]\.(?![A-Z])|(?:[A-Z]\.){2,}[A-Z]\.?"
    ampersans = r"[A-Z]+(?:&[A-Z]+)+"

    pattern_simple = rf"\b(?:{all_caps}|{letter_number})(?=\W|$)"
    pattern_extended = (
        rf"\b(?:{all_caps}|{letter_number}|{initialisms}|{ampersans})(?=\W|$)"
    )
    pattern = pattern_extended if extended else pattern_simple

    return re.findall(pattern, text)


def get_syllable_count_per_word(text: str) -> list[int]:
    """Return the number of syllables in each word of the input text.

    Syllable counts are calculated using textstat.

    Args:
        text: Input text.

    Returns:
        A list of syllable counts for each word in the input text.
    """
    if not text or not isinstance(text, str):
        return []

    return [textstat.syllable_count(word) for word in text.split()]


def get_avg_syllables_per_word(text: str) -> float:
    """Return the average number of syllables per word in the text.

    Args:
        text: Input text.

    Returns:
        float: Average syllables per word.
    """
    if not text or not isinstance(text, str):
        return 0.0

    return textstat.avg_syllables_per_word(text)


def get_simple_language_metrics(text: str) -> dict[str, int | float | list[int]]:
    """Return simple language metrics for one string.

    Args:
        text: Text to analyse.

    Returns:
        A dict containing simple language metrics.
    """
    return {
        "n_acronyms": len(extract_acronyms(text, extended=True)),
        "avg_syllables_per_word": get_avg_syllables_per_word(text),
        "syllable_counts": get_syllable_count_per_word(text),
    }


def add_simple_language_columns(
    df: pd.DataFrame,
    text_column: str,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add simple language metric columns derived from a text column.

    Args:
        df: DataFrame with text data.
        text_column: Column containing text.
        prefix: Prefix for new columns. Defaults to "{text_column}_".

    Returns:
        DataFrame with added text stat columns.
    """
    return add_metrics_columns(
        df,
        text_column=text_column,
        metrics_func=get_simple_language_metrics,
        prefix=prefix,
    )


def summarise_simple_language_columns(
    df: pd.DataFrame,
    *,
    prefix: str,
    syllables_threshold: int = 3,
) -> pd.Series:
    """Summarise precomputed simple language metric columns into a Series.

    Args:
        df: DataFrame containing precomputed simple language metric columns.
        prefix: Prefix used for the metric columns
            (e.g. "<prefix>avg_syllables_per_word").
        syllables_threshold: Threshold for counting words with high syllable counts.

    Returns:
        A Series containing summary statistics.

    """
    n_acronyms_col = df[f"{prefix}n_acronyms"]
    avg_syllables_per_word_col = df[f"{prefix}avg_syllables_per_word"]
    syllable_counts = df[f"{prefix}syllable_counts"]

    max_syllables = syllable_counts.apply(lambda counts: max(counts) if counts else 0)

    summary = {
        "n_count": len(df),
        "pct_with_acronyms": (n_acronyms_col > 0).sum() / len(df) * 100,
        "mean_avg_syllables_per_word": avg_syllables_per_word_col.mean(),
        "pct_with_word_over_syllables_threshold": (
            max_syllables > syllables_threshold
        ).sum()
        / len(df)
        * 100,
    }

    return pd.Series(summary)
