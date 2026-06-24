"""Calculation of metrics for open questions."""

import re

import numpy as np
import pandas as pd
from pydantic import BaseModel
from textstat import textstat


class OpenQuestionMetrics(BaseModel):
    """Container for all open question evaluation metrics."""

    n_count: int
    median_word_count: float
    sd_word_count: float
    mean_sentence_count: float
    mean_word_count_per_sentence: float
    mean_of_mean_syllables_per_word: float
    pct_over_word_count_threshold: float
    pct_over_sentence_count_threshold: float
    pct_with_long_sentence_over_word_count_threshold: float
    pct_blank_or_too_short: float

    def report_metrics(self):
        """Pretty print the open questions evaluation metrics."""
        lines = [
            "\nOpen questions metrics:",
            f" Number of open questions: {self.n_count:.0f}",
            f" Median Word Count: {self.median_word_count:.2f}",
            f" Standard Deviation of Word Count: {self.sd_word_count:.2f}",
            f" Mean Sentence Count: {self.mean_sentence_count:.2f}",
            f" Mean Word Count per Sentence: {self.mean_word_count_per_sentence:.2f}",
            f" Mean of Mean Syllables per Word: {self.mean_of_mean_syllables_per_word:.2f}",
            f" Percent Over Word Threshold Count: {self.pct_over_word_count_threshold:.2f}%",
            f" Percent Over Setence Threshold Count: {self.pct_over_sentence_count_threshold:.2f}%",
            f" Percent with Long Sentences: {
                self.pct_with_long_sentence_over_word_count_threshold:.2f}%",
            f" Percent with Blank or Too Short Sentences: {self.pct_blank_or_too_short:.2f}%",
        ]
        return "\n".join(lines)


def evaluate_open_question(  # noqa: PLR0913 pylint: disable = R0913, R0917
    df,
    text_column: str,
    word_threshold: int = 25,
    sentence_threshold: int = 2,
    long_sentence_threshold: int = 20,
    short_word_count_threshold: int = 2,
) -> OpenQuestionMetrics:
    """Evaluate open-ended question responses.

    Args:
        df: DataFrame containing the responses.
        text_column: Column containing the text responses.
        word_threshold: Threshold for "long" text (word count).
        sentence_threshold: Threshold for number of sentences.
        long_sentence_threshold: Threshold for long sentences.
        short_word_count_threshold: Threshold for "blank or too short".

    Returns:
        A Series with summary statistics for the open-ended question.
    """
    df = filter_nonempty_object_column(df, column=text_column)

    df = add_text_stats_columns(
        df, text_column=text_column, result_prefix="eval_", inplace=True
    )

    metrics = summarise_text_stats(
        df,
        prefix="eval_",
        word_threshold=word_threshold,
        sentence_threshold=sentence_threshold,
        long_sentence_threshold=long_sentence_threshold,
        short_word_count_threshold=short_word_count_threshold,
    )

    return OpenQuestionMetrics(**metrics.to_dict())


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


def word_counts_per_setence(text: str) -> list[int]:
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

    return [
        textstat.lexicon_count(sentence, removepunct=True) for sentence in sentences
    ]


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
        "word_count": textstat.lexicon_count(text, removepunct=True),
        "sentence_count": textstat.sentence_count(text),
        "syllable_count": textstat.syllable_count(text),
        "character_count": textstat.char_count(text),
        "letter_count": textstat.letter_count(text),
        "words_per_sentence": word_counts_per_setence(text),
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
    *,
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
        "n_count": len(df),
        "mean_word_count": df[f"{prefix}word_count"].mean(),
        "median_word_count": df[f"{prefix}word_count"].median(),
        "sd_word_count": df[f"{prefix}word_count"].std(),
        "mean_sentence_count": df[f"{prefix}sentence_count"].mean(),
        "mean_word_count_per_sentence": np.mean(
            df[f"{prefix}words_per_sentence"].sum()
        ),
        "mean_of_mean_syllables_per_word": df[
            f"{prefix}mean_syllables_per_word"
        ].mean(),
        "pct_over_word_count_threshold": (
            df[f"{prefix}word_count"] > word_threshold
        ).mean()
        * 100,
        "pct_over_sentence_count_threshold": (
            df[f"{prefix}sentence_count"] > sentence_threshold
        ).mean()
        * 100,
        "pct_with_long_sentence_over_word_count_threshold": (
            df[f"{prefix}words_per_sentence"].apply(max) > long_sentence_threshold
        ).mean()
        * 100,
        "pct_blank_or_too_short": (
            df[f"{prefix}word_count"] <= short_word_count_threshold
        ).mean()
        * 100,
    }

    return pd.Series(summary)


def compare_text_stats(  # noqa: PLR0913 pylint: disable=R0913
    datasets: dict[str, pd.DataFrame],
    *,
    prefix: str | None = None,
    text_column: str | None = None,
    word_threshold: int = 25,
    sentence_threshold: int = 2,
    long_sentence_threshold: int = 20,
    short_word_count_threshold: int = 2,
) -> pd.DataFrame:
    """Compare text statistics for labeled datasets.

    Args:
        datasets: Mapping of labels to DataFrames.
        prefix: Prefix for precomputed stat columns. Required if text_column is None.
        text_column: If provided, stats will be computed from this text column.
        word_threshold: Threshold for "long" text (word count).
        sentence_threshold: Threshold for number of sentences.
        long_sentence_threshold: Threshold for long sentences.
        short_word_count_threshold: Threshold for "blank or too short".

    Returns:
        A DataFrame containing summary statistics for each labeled dataset.
    """
    summaries = []
    labels = []

    for label, dataframe in datasets.items():
        summary = summarise_text_stats(
            dataframe,
            prefix=prefix,
            text_column=text_column,
            word_threshold=word_threshold,
            sentence_threshold=sentence_threshold,
            long_sentence_threshold=long_sentence_threshold,
            short_word_count_threshold=short_word_count_threshold,
        )
        summaries.append(summary)
        labels.append(label)

    result = pd.DataFrame(summaries, index=labels)
    result.index.name = "dataset"
    return result


def has_question_mark(text: str) -> bool:
    """Check if the text contains a question mark.

    Args:
        text (str): Input text.

    Returns:
        bool: True if '?' is present, else False.
    """
    return isinstance(text, str) and "?" in text


def has_interrogative_anywhere(text: str) -> bool:
    """Check if the text contains interrogative (WH) words anywhere.

    Args:
        text: Input text.

    Returns:
        bool: True if a WH-word is found, else False.
    """
    if not isinstance(text, str):
        return False

    pattern = r"\b(what|why|how|when|where|who|whom|whose|which)\b"
    return re.search(pattern, text.lower()) is not None


def has_interrogative_start(text: str) -> bool:
    """Check if the text starts with an interrogative or auxiliary verb.

    Args:
        text: Input text.

    Returns:
        bool: True if the sentence starts with a question form, else False.
    """
    if not isinstance(text, str):
        return False

    pattern = r"""^\s*
        (what|why|how|when|where|who|whom|whose|which|
        is|are|do|does|did|can|could|would|should|will|have|has|had)
        \b
    """

    return re.search(pattern, text.lower(), re.VERBOSE) is not None


def has_instruction_prompt_anywhere(text: str) -> bool:
    """Check if the text contains instruction-style prompts.

    Args:
        text: Input text.

    Returns:
        bool: True if an instruction prompt is found, else False.
    """
    if not isinstance(text, str):
        return False

    patterns = [
        r"\btell me\b",
        r"\bdescribe\b",
        r"\bexplain\b",
        r"\bplease describe\b",
        r"\bplease explain\b",
        r"\bplease tell me\b",
        r"\bplease share\b",
        r"\bshare\b",
        r"\bgive details\b",
    ]

    text = text.lower()
    return any(re.search(p, text) for p in patterns)


def has_instruction_prompt_start(text: str) -> bool:
    """Check if the text starts with an instruction-style prompt.

    Args:
        text: Input text.

    Returns:
        bool: True if the text starts with an instruction prompt, else False.
    """
    if not isinstance(text, str):
        return False

    pattern = r"""^\s*
        (tell\ me|
         describe|
         explain|
         please\ describe|
         please\ explain|
         please\ tell\ me|
         please\ share|
         share|
         give)
        \b
    """

    return re.search(pattern, text.lower(), re.VERBOSE) is not None


def count_question_signals(text: str) -> int:
    """Count distinct question signals in the text.

    Signals include:
    - Question mark
    - Interrogative start
    - Interrogative anywhere
    - Instruction prompt at start
    - Instruction prompt anywhere

    Args:
        text: Input text.

    Returns:
        int: Number of detected signals.
    """
    signals = [
        has_question_mark(text),
        has_interrogative_start(text),
        has_interrogative_anywhere(text),
        has_instruction_prompt_start(text),
        has_instruction_prompt_anywhere(text),
    ]

    return sum(signals)


def is_compound_question(text: str) -> bool:
    """Check whether the text is likely a compound (double-barrelled) question.

    Criteria:
    - Contains multiple clauses joined by conjunctions (e.g. 'and', 'or')
    - May include multiple question signals (e.g. multiple interrogatives)

    Args:
        text: Input text.

    Returns:
        bool: True if the text is likely compound, else False.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    text = text.strip().lower()

    # Rule 1: multiple question marks
    if text.count("?") > 1:
        return True

    # Rule 2: multiple interrogatives (e.g. "what ... and how ...")
    interrogatives = re.findall(r"\b(what|why|how|when|where|who|which)\b", text)
    if len(interrogatives) > 1:
        return True

    # Rule 3: conjunctions linking clauses
    conjunctions = r"\b(and|or|also)\b"
    return not re.search(conjunctions, text)


def has_single_question(text: str) -> bool:
    """Check whether the text contains a single question.

    Criteria:
    - Contains at most one question mark
    - Contains at least one question signal (e.g. interrogative, instruction prompt)
    - Is not a compound (double-barrelled) question

    Args:
        text: Input text.

    Returns:
        bool: True if the text is likely a single question, else False.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    text = text.strip()

    # Rule 1: multiple '?' means multiple questions
    if text.count("?") > 1:
        return False

    # Rule 2: must have at least one question signal
    if count_question_signals(text) == 0:
        return False

    # Rule 3: reject compound questions
    return not is_compound_question(text)


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
