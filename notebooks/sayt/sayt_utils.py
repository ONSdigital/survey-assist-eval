"""Utility functions for SAYT evaluation."""

import logging
import time
from typing import Any

import pandas as pd
from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SAYTSuggester,
    SemanticRetrieverSpec,
)

from survey_assist_eval.data_cleaning.code_standard import get_clean_n_digit_codes


def build_lookup_suggester(
    corpus: list[tuple[str, str]], *, semantic_weight: float | None
) -> SAYTSuggester:
    """Build a lookup suggester using the explicit retriever-spec API.

    Args:
        corpus: Search corpus as (search_text, display_text) tuples.
        semantic_weight: Weight for semantic retrieval. If None, semantic retrieval
            is not included.

    Returns:
        SAYTSuggester: Configured suggester instance.
    """
    retrievers = [PrefixRetrieverSpec(), NgramRetrieverSpec()]
    if semantic_weight is not None:
        retrievers.append(SemanticRetrieverSpec(weight=semantic_weight))
    return SAYTSuggester(corpus, retrievers=retrievers)


def validate_one_code(code: str, logger: logging.Logger, code_length=5) -> bool:
    """Validate one SIC code and log malformed values.

    Args:
        code: SIC code value to validate.
        logger: Logger used for warning messages.
        code_length: Expected SIC code length.

    Returns:
        bool: True when the code is valid and unchanged after cleaning, else False.
    """
    if pd.isna(code):
        logger.warning("Code is NaN")
        return False
    clean_codes = get_clean_n_digit_codes(code, n=code_length, code_type="SIC")
    if len(clean_codes[1]) != 0:
        logger.warning(f"Malformed code: {code}")
        return False
    if len(clean_codes[0]) != 1 or next(iter(clean_codes[0])) != code:
        logger.warning(f"Code {code} cleaned to different code: {clean_codes[0]}")
        return False
    return True


def get_suggestions_for_row(
    row: dict[str, Any],
    suggester: Any,
    num_chars: int,
    max_suggestions: int,
) -> list[str]:
    """Return suggester output for a single input row.

    Args:
        row: Input row containing a `full_entry` text field.
        suggester: Suggester object exposing a `suggest` method.
        num_chars: Number of leading characters from `full_entry` to use as input.
        max_suggestions: Maximum number of suggestions to request.

    Returns:
        list[str]: Ordered suggestion strings returned by the suggester.
    """
    return suggester.suggest(
        row["full_entry"][:num_chars],
        num_suggestions=max_suggestions,
    )


def rank_of_correct_code_in_suggestions(
    row: dict[str, Any],
    num_chars: int,
    suggester_label: str,
    code_length: int = 5,
    correct_code_col: str = "correct_sic_code",
) -> int | None:
    """Return the rank of the correct code in generated suggestions.

    Args:
        row: Input row containing suggestion outputs and the correct code.
        num_chars: Prefix length used to generate suggestions.
        suggester_label: Label used in the suggestion column name.
        code_length: Number of trailing characters to compare as code.
        correct_code_col: Column name holding the correct SIC code.

    Returns:
        int | None: 1-based rank of the correct code, or None if not found.
    """
    correct_code = row[correct_code_col]
    suggested_codes = get_codes_from_suggestions(
        row,
        suggestions_col=f"suggestions_{num_chars}chars_{suggester_label}",
        code_length=code_length,
    )

    for rank, suggest in enumerate(suggested_codes):
        if suggest == correct_code:
            return rank + 1
    return None


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


def timed_apply(df: pd.DataFrame, func, **kwargs) -> tuple[pd.Series, float]:
    """Run df.apply and return the results alongside the average time per row.

    Args:
        df: DataFrame to apply the function to.
        func: Callable to apply row-wise.
        **kwargs: Additional keyword arguments passed to df.apply.

    Returns:
        tuple[pd.Series, float]: Apply results and average milliseconds per row.
    """
    t_start = time.perf_counter()
    results = df.apply(func, **kwargs)
    avg_ms = (time.perf_counter() - t_start) / len(df) * 1000
    return results, avg_ms
