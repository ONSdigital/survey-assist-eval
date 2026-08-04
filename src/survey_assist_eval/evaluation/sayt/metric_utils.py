"""Utility functions for SAYT evaluation."""

from typing import Any


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


def get_codes_from_suggestions(
    row: dict[str, Any],
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
