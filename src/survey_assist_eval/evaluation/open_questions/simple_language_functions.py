"""Functions for checking simple language in open questions."""

import re


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
