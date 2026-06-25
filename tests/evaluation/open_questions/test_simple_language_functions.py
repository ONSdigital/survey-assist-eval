"""Tests for simple language functions."""

from survey_assist_eval.evaluation.open_questions.simple_language_functions import (
    extract_acronyms,
)


def test_simple_acronyms_basic_and_digits():
    """Extracts uppercase acronyms and those with digits using the simple pattern."""
    text = "ONS, NLP and ISO9001 with G7 and DD10."
    assert extract_acronyms(text) == ["ONS", "NLP", "ISO9001", "G7", "DD10"]


def test_simple_ignores_invalid_patterns():
    """Ignores lowercase, mixed case and single-letter tokens."""
    text = "A b Ab OnS nlp are not valid acronyms."
    assert extract_acronyms(text) == []


def test_extended_includes_dotted_and_ampersand():
    """Extended mode captures dotted and ampersand acronyms."""
    text = (
        "A ONS works with U.S.A or U.S. on R&D and M&S&Z projects "
        "and check simple G7."
    )
    result = extract_acronyms(text, extended=True)
    assert set(result) == {"ONS", "U.S.A", "U.S.", "R&D", "M&S&Z", "G7"}


def test_extended_excludes_invalid_variants():
    """Extended mode does not match malformed dotted or spaced ampersand patterns."""
    text = "U.S is incomplete and R & D has spaces and US.A is only US."
    result = extract_acronyms(text, extended=True)
    assert "U.S" not in result
    assert "R & D" not in result
    assert "US.A" not in result
    assert "US" in result


def test_extended_ignores_invalid_patterns():
    """Ignores lowercase, mixed case and single-letter tokens."""
    text = "A b Ab OnS nlp 9 are not valid acronyms."
    assert extract_acronyms(text) == []


def test_boundary_and_punctuation_handling():
    """Correctly identifies acronyms at boundaries and next to punctuation."""
    text = "(ONS), NLP; ISO9001."
    assert extract_acronyms(text) == ["ONS", "NLP", "ISO9001"]


def test_duplicates_and_order_preserved():
    """Preserves duplicate acronyms and maintains original order."""
    text = "ONS works with ONS and NLP"
    assert extract_acronyms(text) == ["ONS", "ONS", "NLP"]


def test_non_string_and_empty_inputs():
    """Returns empty list for non-string, empty or whitespace-only input."""
    assert extract_acronyms(None) == []
    assert extract_acronyms("") == []
    assert extract_acronyms("   ") == []


def test_long_token_not_split():
    """Does not split long uppercase tokens into partial matches."""
    text = "ABCDEF, A&B&C&D&E, A.B.C.D.E"
    result = extract_acronyms(text, extended=True)
    assert set(result) == {"ABCDEF", "A&B&C&D&E", "A.B.C.D.E"}
