"""Functions for checking question structure in open questions."""

import re

import pandas as pd
from pydantic import BaseModel


class QuestionStructureMetrics(BaseModel):
    """Container for all question structure evaluation metrics."""

    n_follow_up_questions: int
    pct_is_question: float
    pct_is_single_question: float
    pct_with_instruction_prompt_start: float
    mean_instruction_prompt_count_excluding_zero: float | None
    pct_with_interrogative_start: float
    mean_interrogative_wh_count_excluding_zero: float | None
    pct_has_question_mark: float

    def report_metrics(self) -> str:
        """Pretty print the question structure evaluation metrics."""
        lines = [
            "\nQuestion structure metrics:",
            f" Number of follow-up questions: {self.n_follow_up_questions:.0f}",
            f" Percentage is_question: {self.pct_is_question:.2f}%",
            " Percentage is_single_question:" f" {self.pct_is_single_question:.2f}%",
            " Percentage with instruction_prompt_start:"
            f" {self.pct_with_instruction_prompt_start:.2f}%",
            (
                f" Mean instruction prompt count (excluding zero):"
                f" {self.mean_instruction_prompt_count_excluding_zero:.2f}"
                if pd.notna(self.mean_instruction_prompt_count_excluding_zero)
                else " Mean instruction prompt count (excluding zero): N/A"
            ),
            f" Percentage with interrogative_start: {self.pct_with_interrogative_start:.2f}%",
            (
                f" Mean interrogative WH count (excluding zero):"
                f" {self.mean_interrogative_wh_count_excluding_zero:.2f}"
                if self.mean_interrogative_wh_count_excluding_zero is not None
                else " Mean interrogative WH count (excluding zero): N/A"
            ),
            f" Percentage with question mark: {self.pct_has_question_mark:.2f}%",
        ]
        return "\n".join(lines)


def compute_question_structure_metrics(
    df,
    *,
    text_column: str,
    prefix: str = "eval_",
) -> QuestionStructureMetrics:
    """Evaluate question structure quality for generated follow-up questions.

    Args:
        df: DataFrame containing generated questions.
        text_column: Column containing the text responses.
        prefix: Prefix for generated metric columns.

    Returns:
        QuestionStructureMetrics: Structured summary of metrics.
    """
    df = add_question_structure_columns(df, text_column=text_column, prefix=prefix)

    metrics = summarise_question_structure_columns(
        df,
        prefix=prefix,
    )

    return QuestionStructureMetrics(**metrics.to_dict())


def has_question_mark(text: str) -> bool:
    """Check if the text contains a question mark.

    Args:
        text (str): Input text.

    Returns:
        bool: True if '?' is present, else False.
    """
    return isinstance(text, str) and "?" in text


def count_wh_interrogatives(text: str) -> int:
    """Count WH-interrogative words in the input text.

    WH-interrogatives include:
        what, why, how, when, where, who, whom, whose, which

    Args:
        text: Input text to analyse.

    Returns:
        int: Number of WH-interrogative word matches found in the text.
             Returns 0 if input is not a string.
    """
    if not isinstance(text, str):
        return 0

    pattern = r"\b(what|why|how|when|where|who|whom|whose|which)\b"
    return len(re.findall(pattern, text.lower()))


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


def has_interrogative_not_at_start(text: str) -> bool:
    """Check if the text contains interrogative (WH) words but does not start
    with an interrogative or auxiliary verb.

    Args:
        text (str): Input text.

    Returns:
        bool: True if a WH-word is found and the text does not start with a
        question form, else False.
    """
    return count_wh_interrogatives(text) > 0 and not has_interrogative_start(text)


def count_instruction_prompts(text: str) -> int:
    """Count instruction-style prompt phrases in the input text.

    Instruction prompts include directive or request-based language such as:
        "tell me", "describe", "explain", "share", "give details",
        including polite variants (e.g. "please explain", "please describe").

    Args:
        text: Input text to analyse.

    Returns:
        int: Number of instruction prompt matches found in the text.
             Returns 0 if input is not a string.
    """
    if not isinstance(text, str):
        return 0

    pattern = re.compile(
        r"""\b(please (describe|explain|tell me|share)
        |tell me|describe|explain|share|give details)\b""",
        re.IGNORECASE,
    )

    return len(pattern.findall(text))


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


def has_instruction_prompt_not_at_start(text: str) -> bool:
    """Check if the text contains instruction-style prompts but does not start
    with one.

    Args:
        text (str): Input text.

    Returns:
        bool: True if an instruction prompt is found not at the start of the
        text, else False.
    """
    return count_instruction_prompts(text) > 0 and not has_instruction_prompt_start(
        text
    )


def is_question(text: str) -> bool:
    """Determine whether a text is a question based on question signals.

    A text is considered a question if any question signal is present.

    Args:
        text: Input text.

    Returns:
        True if the text contains at least one question signal, else False.
    """
    if not isinstance(text, str):
        return False

    return any(
        [
            has_question_mark(text),
            has_interrogative_start(text),
            count_wh_interrogatives(text) > 0,
            count_instruction_prompts(text) > 0,
        ]
    )


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
    if count_wh_interrogatives(text) > 1:
        return True

    if count_instruction_prompts(text) > 1:
        return True

    if has_interrogative_start(text) and count_wh_interrogatives(text) > 1:
        return True

    # Rule 3: conjunctions linking clauses
    conjunctions = r"\b(and|or|also)\b"
    return re.search(conjunctions, text) is not None


def is_single_question(text: str) -> bool:
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
    if not is_question(text):
        return False

    # Rule 3: reject compound questions
    return not is_compound_question(text)


def get_question_structure_metrics(text: str) -> dict[str, int | bool]:
    """Return question structure metrics for one string.

    Args:
        text: Text to analyse.

    Returns:
        A dict containing question signal counts, question structure flags,
        and single/compound question classifications.
    """
    return {
        # Question signal flags
        "has_question_mark": has_question_mark(text),
        "interrogative_start": has_interrogative_start(text),
        "interrogative_anywhere": has_interrogative_anywhere(text),
        "interrogative_not_at_start": has_interrogative_not_at_start(text),
        "instruction_prompt_start": has_instruction_prompt_start(text),
        "instruction_prompt_anywhere": has_instruction_prompt_anywhere(text),
        "instruction_prompt_not_at_start": (has_instruction_prompt_not_at_start(text)),
        # Question signal counts
        "interrogative_wh_count": count_wh_interrogatives(text),
        "instruction_prompt_count": count_instruction_prompts(text),
        # Summary signal count
        "question_signal_count": is_question(text),
        # Question structure classifications
        "is_question": is_question(text),
        "is_compound_question": is_compound_question(text),
        "is_single_question": is_single_question(text),
    }


def add_question_structure_columns(
    df: pd.DataFrame,
    text_column: str,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add question structure metric columns derived from a text column.

    Args:
        df: DataFrame with text data.
        text_column: Column containing text.
        prefix: Prefix for new columns. Defaults to "{text_column}_".

    Returns:
        DataFrame with added text stat columns.
    """
    question_metrics_df = (
        df[text_column]
        .fillna("")
        .astype(str)
        .apply(get_question_structure_metrics)
        .apply(pd.Series)
    )

    if prefix is None:
        prefix = f"{text_column}_"

    question_metrics_df = question_metrics_df.rename(
        columns=lambda col: f"{prefix}{col}"
    )

    return df.join(question_metrics_df)


def summarise_question_structure_columns(
    df: pd.DataFrame,
    *,
    prefix: str,
) -> pd.Series:
    """Summarise precomputed question structure metric columns into a Series.

    Args:
        df: DataFrame containing precomputed question structure columns.
        prefix: Prefix used for the metric columns
            (e.g. "<prefix>is_question").

    Returns:
        A Series containing summary statistics.

    Notes:
        This function assumes all required boolean columns already exist:
        - {prefix}is_question
        - {prefix}is_single_question
        - {prefix}instruction_prompt_start
        - {prefix}interrogative_start
        - {prefix}has_question_mark
    """
    is_question_col = df[f"{prefix}is_question"]
    is_single_question_col = df[f"{prefix}is_single_question"]
    instruction_prompt_start_col = df[f"{prefix}instruction_prompt_start"]
    instruction_prompt_count_col = df[f"{prefix}instruction_prompt_count"]
    interrogative_start_col = df[f"{prefix}interrogative_start"]
    interrogative_wh_count_col = df[f"{prefix}interrogative_wh_count"]
    has_question_mark_col = df[f"{prefix}has_question_mark"]

    summary = {
        # Count
        "n_follow_up_questions": len(df),
        # Percentages
        "pct_is_question": is_question_col.mean() * 100,
        "pct_is_single_question": is_single_question_col.mean() * 100,
        "pct_with_instruction_prompt_start": instruction_prompt_start_col.mean() * 100,
        "pct_with_interrogative_start": interrogative_start_col.mean() * 100,
        "pct_has_question_mark": has_question_mark_col.mean() * 100,
        "mean_instruction_prompt_count_excluding_zero": instruction_prompt_count_col[
            instruction_prompt_count_col > 0
        ].mean(),
        "mean_interrogative_wh_count_excluding_zero": interrogative_wh_count_col[
            interrogative_wh_count_col > 0
        ].mean(),
    }

    return pd.Series(summary)
