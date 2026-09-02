"""Functions for extracting and ranking codes within SAYT suggestions."""

import pandas as pd


def get_codes_from_suggestions(
    row: pd.Series,
    suggestions_col: str,
    code_length: int = 5,
) -> list[str]:
    """Extract code suffixes from suggestion strings for a single input row.

    Args:
        row: Input row containing a suggestions column.
        suggestions_col: Column name containing suggestion strings.
        code_length: Number of trailing characters to extract as a code.

    Returns:
        list[str]: Extracted codes in suggestion order.
    """
    return [suggestion[-code_length:] for suggestion in row[suggestions_col]]


def get_rank_of_first_matching_code(
    retrieved_codes: list[str], correct_codes: str | list[str]
) -> int | None:
    """Get the rank of the first retrieved code matching correct code(s).

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_codes: A single correct code or list of correct codes to match against.

    Returns:
        int: Rank of the first matching code, or None if no match found.
    """
    if isinstance(correct_codes, str):
        correct_codes = [correct_codes]

    for rank, item in enumerate(retrieved_codes, start=1):
        if item in correct_codes:
            return int(rank)
    return None


def is_correct_codes_empty(codes: str | list[str] | None) -> bool:
    """Check whether a correct-codes value represents missing ground truth.

    Args:
        codes: A single correct code, list of correct codes, or a missing value
            (None or NaN).

    Returns:
        bool: True if codes is None, NaN, an empty string, or an empty list.
    """
    if isinstance(codes, str):
        return pd.isna(codes) or codes == ""
    if codes is None:
        return True
    if isinstance(codes, float) and pd.isna(codes):
        return True
    return len(codes) == 0


def rank_of_correct_code_in_suggestions(
    row: pd.Series,
    num_chars: int,
    suggester_label: str,
    code_length: int = 5,
    correct_codes_col: str = "correct_sic_code",
) -> int | None:
    """Return the rank of the correct code in generated suggestions.

    Args:
        row: Input row containing suggestion outputs and the correct code.
        num_chars: Prefix length used to generate suggestions.
        suggester_label: Label used in the suggestion column name.
        code_length: Number of trailing characters to compare as code.
        correct_codes_col: Column name holding the correct SIC code(s).

    Returns:
        int | None: 1-based rank of the correct code, or None if not found.
    """
    correct_codes = row[correct_codes_col]

    suggested_codes = get_codes_from_suggestions(
        row,
        suggestions_col=f"suggestions_{num_chars}chars_{suggester_label}",
        code_length=code_length,
    )

    return get_rank_of_first_matching_code(suggested_codes, correct_codes)
