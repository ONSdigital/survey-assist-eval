"""Functions for computing text statistics for open questions."""

import re

import numpy as np
import pandas as pd
from pydantic import BaseModel
from textstat import textstat

from survey_assist_eval.evaluation.open_questions.metric_utils import (
    add_metrics_columns,
)


class OpenQuestionTextStatistics(BaseModel):
    """Container for all open question evaluation metrics."""

    n_count: int
    mean_word_count: float
    median_word_count: float
    sd_word_count: float
    mean_sentence_count: float
    mean_word_count_per_sentence: float
    word_threshold: int
    pct_over_word_count_threshold: float
    sentence_threshold: int
    pct_over_sentence_count_threshold: float
    long_sentence_word_threshold: int
    pct_with_sentence_over_long_sentence_word_threshold: float
    short_text_word_threshold: int
    pct_blank_or_too_short: float

    def report_metrics(self):
        """Pretty print the open questions evaluation metrics."""
        lines = [
            "\nText statistics:",
            f" Number of follow-up questions: {self.n_count:.0f}",
            f" Median Word Count: {self.median_word_count:.2f}",
            f" Standard Deviation of Word Count: {self.sd_word_count:.2f}",
            f" Mean Sentence Count: {self.mean_sentence_count:.2f}",
            f" Mean Word Count: {self.mean_word_count:.2f}",
            f" Mean Word Count per Sentence: {self.mean_word_count_per_sentence:.2f}",
            f" Percent with more than {self.word_threshold} words: "
            f"{self.pct_over_word_count_threshold:.2f}%",
            f" Percent with less than {self.short_text_word_threshold} words: "
            f"{self.pct_blank_or_too_short:.2f}%",
            f" Percent with more than {self.sentence_threshold} sentences: "
            f"{self.pct_over_sentence_count_threshold:.2f}%",
            f" Percent with more than {self.long_sentence_word_threshold} words in a sentence: "
            f"{self.pct_with_sentence_over_long_sentence_word_threshold:.2f}%",
        ]
        return "\n".join(lines)


def compute_text_statistics(  # noqa: PLR0913 pylint: disable = R0913, R0917
    df,
    text_column: str,
    word_threshold: int = 25,
    sentence_threshold: int = 2,
    long_sentence_word_threshold: int = 20,
    short_text_word_threshold: int = 2,
) -> OpenQuestionTextStatistics:
    """Evaluate open-ended question responses.

    Args:
        df: DataFrame containing the responses. Rows should already be filtered to
        remove null or empty values in the text column.

        text_column: Column containing the text responses.
        word_threshold: Threshold for "long" text (word count).
        sentence_threshold: Threshold for number of sentences.
        long_sentence_word_threshold: Threshold for long sentences.
        short_text_word_threshold: Threshold for "blank or too short".

    Returns:
        An OpenQuestionTextStatistics object containing summary
        statistics for the open-ended question responses.
    """
    df = add_text_stats_columns(df, text_column=text_column, prefix="eval_")

    statistics = summarise_text_stat_columns(
        df,
        prefix="eval_",
        word_threshold=word_threshold,
        sentence_threshold=sentence_threshold,
        long_sentence_word_threshold=long_sentence_word_threshold,
        short_text_word_threshold=short_text_word_threshold,
    )

    return statistics


def word_count_hyph_contract_split(text: str) -> int:
    """Return the number of words in the input text.

    Word counts are calculated using textstat, counting
    contractions and hyphenated words as separate words.

    Args:
        text: Input text.

    Returns:
        The word count as an integer.
    """
    return textstat.lexicon_count(
        text, removepunct=True, split_contractions=True, split_hyphens=True
    )


def word_counts_per_sentence(text: str) -> list[int]:
    """Return the number of words in each sentence of the input text.

    Sentences are split on ., !, and ?.
    Word counts are calculated using textstat.

    Args:
        text: Input text.

    Returns:
        A list of word counts, one per sentence.
    """
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return [word_count_hyph_contract_split(sentence) for sentence in sentences]


def get_text_stats(text: str) -> dict[str, int | float | list[int]]:
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
        "word_count": word_count_hyph_contract_split(text),
        "sentence_count": textstat.sentence_count(text),
        "character_count": textstat.char_count(text),
        "letter_count": textstat.letter_count(text),
        "words_per_sentence": word_counts_per_sentence(text),
        "mean_words_per_sentence": textstat.words_per_sentence(text),
    }


def add_text_stats_columns(
    df: pd.DataFrame,
    text_column: str,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add text stats columns for a DataFrame text column.

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
        metrics_func=get_text_stats,
        prefix=prefix,
    )


def summarise_text_stat_columns(  # noqa: PLR0913, pylint: disable=R0913
    df: pd.DataFrame,
    *,
    prefix: str,
    word_threshold: int = 25,
    sentence_threshold: int = 2,
    long_sentence_word_threshold: int = 20,
    short_text_word_threshold: int = 2,
) -> OpenQuestionTextStatistics:
    """Summarise precomputed text statistic columns into a Series.

    Args:
        df: DataFrame containing precomputed text statistic columns.
        prefix: Prefix used for the statistic columns
            (e.g. "<prefix>word_count").
        word_threshold: Threshold for "long" text (word count).
        sentence_threshold: Threshold for number of sentences.
        long_sentence_word_threshold: Threshold for long sentences
            (words per sentence).
        short_text_word_threshold: Threshold for "blank or too short".

    Returns:
        An OpenQuestionTextStatistics object containing summary
        statistics for the open-ended question responses.

    Notes:
        This function assumes all required columns already exist:
        - {prefix}word_count
        - {prefix}sentence_count
        - {prefix}words_per_sentence
    """
    word_count = df[f"{prefix}word_count"]
    sentence_count = df[f"{prefix}sentence_count"]
    words_per_sentence = df[f"{prefix}words_per_sentence"]

    summary = {
        "n_count": len(df),
        "mean_word_count": word_count.mean(),
        "median_word_count": word_count.median(),
        "sd_word_count": word_count.std(),
        "mean_sentence_count": sentence_count.mean(),
        "mean_word_count_per_sentence": np.mean(words_per_sentence.sum()),
        "word_threshold": word_threshold,
        "pct_over_word_count_threshold": (word_count > word_threshold).mean() * 100,
        "sentence_threshold": sentence_threshold,
        "pct_over_sentence_count_threshold": (
            sentence_count > sentence_threshold
        ).mean()
        * 100,
        "long_sentence_word_threshold": long_sentence_word_threshold,
        "pct_with_sentence_over_long_sentence_word_threshold": (
            words_per_sentence.apply(max) > long_sentence_word_threshold
        ).mean()
        * 100,
        "short_text_word_threshold": short_text_word_threshold,
        "pct_blank_or_too_short": (word_count < short_text_word_threshold).mean() * 100,
    }

    return OpenQuestionTextStatistics(**summary)


def compare_text_statistics(  # noqa: PLR0913, pylint: disable=R0913
    datasets: dict[str, pd.DataFrame],
    *,
    prefix: str | None = None,
    text_column: str | None = None,
    word_threshold: int = 25,
    sentence_threshold: int = 2,
    long_sentence_word_threshold: int = 20,
    short_text_word_threshold: int = 3,
) -> pd.DataFrame:
    """Compare text statistics across labelled datasets.

    Args:
        datasets: Mapping of dataset labels to DataFrames.
        prefix: Prefix for precomputed stat columns.
        text_column: Column containing raw text (used to compute stats).
        word_threshold: Threshold for "long" text (word count).
        sentence_threshold: Threshold for number of sentences.
        long_sentence_word_threshold: Threshold for long sentences (word count per sentence).
        short_text_word_threshold: Threshold for "blank or too short".

    Returns:
        A DataFrame of summary statistics, indexed by dataset label.

    Notes:
        - Provide `text_column` to compute stats from raw text.
        - Provide `prefix` if stat columns already exist.
        - Exactly one of `text_column` or `prefix` must be provided.
    """
    if text_column is None and prefix is None:
        raise ValueError("Provide at least one of 'text_column' or 'prefix'.")

    summaries = []

    for label, df in datasets.items():
        if text_column is not None:
            text_stats_summary = compute_text_statistics(
                df.copy(),
                text_column=text_column,
                word_threshold=word_threshold,
                sentence_threshold=sentence_threshold,
                long_sentence_word_threshold=long_sentence_word_threshold,
                short_text_word_threshold=short_text_word_threshold,
            )
        else:
            if prefix is None:
                raise ValueError("Prefix must be provided when text_column is None.")
            text_stats_summary = summarise_text_stat_columns(
                df.copy(),
                prefix=prefix,
                word_threshold=word_threshold,
                sentence_threshold=sentence_threshold,
                long_sentence_word_threshold=long_sentence_word_threshold,
                short_text_word_threshold=short_text_word_threshold,
            )

        text_stats_summary_series = pd.Series(text_stats_summary.__dict__, name=label)
        summaries.append(text_stats_summary_series)

    result = pd.DataFrame(summaries)
    result.index.name = "dataset"
    return result
