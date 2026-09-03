"""Functions for extracting and ranking codes within SAYT suggestions."""

import pandas as pd

from survey_assist_eval.data_cleaning.code_standard import SIC_EXPECTED_CODE_LENGTH
from survey_assist_eval.data_cleaning.prep_data import (
    get_clean_n_digit_codes,
)


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
    retrieved_codes: list[str], correct_codes: str | set[str]
) -> int | None:
    """Get the rank of the first retrieved code matching correct code(s).

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_codes: A single correct code or set of correct codes to match against.

    Returns:
        int: Rank of the first matching code, or None if no match found.
    """
    if isinstance(correct_codes, str):
        correct_codes = {correct_codes}

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


def _get_valid_codes(codes: str | list[str] | None, n: int, code_type: str) -> set[str]:
    """Return the set of valid, cleaned codes for a single codes value.

    Args:
        codes: A single code, list of codes, or a missing value (None or NaN).
        n: Number of leading characters to keep.
        code_type: Type of code ('sic' or 'soc').

    Returns:
        set[str]: Set of valid, cleaned codes, or an empty set if codes is missing.
    """
    if codes is None or is_correct_codes_empty(codes):
        return set()
    return get_clean_n_digit_codes(
        codes if isinstance(codes, str) else list(codes), n=n, code_type=code_type
    )[0]


def clean_codes_columns(
    df: pd.DataFrame,
    code_digit_match_length: int,
    code_length: int = 5,
    correct_codes_col: str | None = None,
    retrieved_codes_col: str | None = None,
) -> pd.DataFrame:
    """Clean and truncate correct codes, and optionally retrieved codes, to a
    fixed digit length.

    Args:
        df: DataFrame containing the correct-codes column and, optionally, the
            retrieved-codes column.
        code_digit_match_length: Number of leading characters to keep.
        code_length: Full code length used to infer code_type ('sic' if 5, else 'soc').
        correct_codes_col: Column name containing correct code(s) (string or list).
        retrieved_codes_col: Optional column name containing lists of retrieved
            codes to clean and truncate as well.

    Returns:
        pd.DataFrame: Copy of df with the following columns added:
            - "{correct_codes_col}_clean": set of valid, cleaned correct codes
              (only added if correct_codes_col is given).
            - "{retrieved_codes_col}_valid": set of valid, cleaned retrieved codes
              (only added if retrieved_codes_col is given).
            - "{retrieved_codes_col}_clean": list of retrieved codes truncated to
              code_digit_match_length, in the original order with duplicates
              preserved, with any code not in "_valid" replaced by "-9"
              (only added if retrieved_codes_col is given).

    Raises:
        ValueError: If correct_codes_col and retrieved_codes_col are the same
            column.
    """
    df = df.copy()

    code_type = "sic" if code_length == SIC_EXPECTED_CODE_LENGTH else "soc"

    if correct_codes_col is not None and correct_codes_col == retrieved_codes_col:
        raise ValueError(
            "correct_codes_col and retrieved_codes_col must not be the same column"
        )

    for col in [correct_codes_col, retrieved_codes_col]:
        if col is None:
            continue

        valid_col = f"{col}_valid"
        df[valid_col] = df[col].apply(
            _get_valid_codes, n=code_digit_match_length, code_type=code_type
        )

        if col == correct_codes_col:
            df[f"{col}_clean"] = df[valid_col]
        else:
            df[f"{col}_clean"] = [
                [
                    (
                        truncated
                        if (truncated := code[:code_digit_match_length]) in valid_set
                        else "-9"
                    )
                    for code in ([] if is_correct_codes_empty(codes) else list(codes))
                ]
                for codes, valid_set in zip(df[col], df[valid_col], strict=False)
            ]

        df = df.drop(columns=[f"{col}_valid"], errors="ignore")

    return df


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
