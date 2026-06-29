"""Functions for checking question structure in open questions."""

import re

import pandas as pd


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


def has_interrogative_not_at_start(text: str) -> bool:
    """Check if the text contains interrogative (WH) words but does not start
    with an interrogative or auxiliary verb.

    Args:
        text (str): Input text.

    Returns:
        bool: True if a WH-word is found and the text does not start with a
        question form, else False.
    """
    return has_interrogative_anywhere(text) and not has_interrogative_start(text)


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
    return has_instruction_prompt_anywhere(text) and not has_instruction_prompt_start(
        text
    )


def count_question_signals(text: str) -> int:
    """Count distinct question signals in the text.

    Signals include:
    - Question mark
    - Interrogative at start
    - Interrogative not at start
    - Instruction prompt at start
    - Instruction prompt not at start

    Args:
        text (str): Input text.

    Returns:
        int: Number of detected signals.
    """
    return sum(
        [
            has_question_mark(text),
            has_interrogative_start(text),
            has_interrogative_not_at_start(text),
            has_instruction_prompt_start(text),
            has_instruction_prompt_not_at_start(text),
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
    interrogatives = re.findall(r"\b(what|why|how|when|where|who|which)\b", text)
    if len(interrogatives) > 1:
        return True

    # Rule 3: conjunctions linking clauses
    conjunctions = r"\b(and|or|also)\b"
    return re.search(conjunctions, text) is not None


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
        "has_interrogative_start": has_interrogative_start(text),
        "has_interrogative_anywhere": has_interrogative_anywhere(text),
        "has_interrogative_not_at_start": has_interrogative_not_at_start(text),
        "has_instruction_prompt_start": has_instruction_prompt_start(text),
        "has_instruction_prompt_anywhere": has_instruction_prompt_anywhere(text),
        "has_instruction_prompt_not_at_start": (
            has_instruction_prompt_not_at_start(text)
        ),
        # Summary signal count
        "question_signal_count": count_question_signals(text),
        # Question structure classifications
        "is_compound_question": is_compound_question(text),
        "has_single_question": has_single_question(text),
    }


def add_question_structure_columns(
    df: pd.DataFrame,
    text_column: str,
    prefix: str | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Add question structure metric columns derived from a text column.

    Args:
        df: DataFrame with text data.
        text_column: Column containing text.
        prefix: Prefix for new columns. Defaults to "{text_column}_".
        inplace: If True, add columns in place and return the same DataFrame.

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

    if inplace:
        df.loc[:, question_metrics_df.columns] = question_metrics_df

    return df
