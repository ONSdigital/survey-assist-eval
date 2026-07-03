"""Tests for simple language functions."""

# pylint: disable=redefined-outer-name

import pandas as pd
import pytest
from textstat import textstat

from survey_assist_eval.evaluation.open_questions.simple_language_functions import (
    add_simple_language_columns,
    extract_acronyms,
    get_avg_syllables_per_word,
    get_syllable_count_per_word,
)


@pytest.fixture
def simple_language_input_df():
    """Return test data for simple language wrapper function tests.

    Covers diverse text patterns for wrapper testing.
    """
    return pd.DataFrame(
        {
            "follow_up_question": [
                "Would you like to participate?",
                "ONS is a UK government agency.",
                "Does the NHS provide good service?",
                "",
                None,
            ],
            "respondent_id": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def expected_simple_language_df():
    """Return expected output after adding simple language columns."""
    return pd.DataFrame(
        {
            "follow_up_question": [
                "Would you like to participate?",
                "ONS is a UK government agency.",
                "Does the NHS provide good service?",
                "",
                None,
            ],
            "respondent_id": [1, 2, 3, 4, 5],
            "follow_up_question_n_acronyms": [0, 2, 1, 0, 0],
            "follow_up_question_avg_syllables_per_word": [
                1.6,
                1.66667,
                1.33333,
                0.0,
                0.0,
            ],
            "follow_up_question_syllable_counts": [
                [1, 1, 1, 1, 4],
                [1, 1, 1, 1, 3, 3],
                [1, 1, 1, 2, 1, 2],
                [],
                [],
            ],
        }
    )


# ============================================================================
# Test extract_acronyms function
# ============================================================================


def test_extract_acronyms_simple_acronyms_basic_and_digits():
    """Extracts uppercase acronyms and those with digits using the simple pattern."""
    text = "ONS, NLP and ISO9001 with G7 and DD10."
    assert extract_acronyms(text) == [
        "ONS",
        "NLP",
        "ISO9001",
        "G7",
        "DD10",
    ], "Expected basic acronyms and digit-containing tokens to be extracted."


def test_extract_acronyms_simple_ignores_invalid_patterns():
    """Ignores lowercase, mixed case and single-letter tokens."""
    text = "A b Ab OnS nlp are not valid acronyms."
    assert (
        extract_acronyms(text) == []
    ), "Expected invalid acronym patterns to be ignored."


def test_extract_acronyms_extended_includes_dotted_and_ampersand():
    """Extended mode captures dotted and ampersand acronyms."""
    text = (
        "A ONS works with U.S.A or U.S. on R&D and M&S&Z projects "
        "and check simple G7."
    )
    result = extract_acronyms(text, extended=True)
    assert set(result) == {
        "ONS",
        "U.S.A",
        "U.S.",
        "R&D",
        "M&S&Z",
        "G7",
    }, "Expected extended acronyms with punctuation and ampersands to be detected."


def test_extract_acronyms_extended_excludes_invalid_variants():
    """Extended mode does not match malformed dotted or spaced ampersand patterns."""
    text = "U.S is incomplete and R & D has spaces and US.A is only US."
    result = extract_acronyms(text, extended=True)
    assert "U.S" not in result, "Expected incomplete dotted acronyms to be excluded."
    assert "R & D" not in result, "Expected spaced acronyms to be excluded."
    assert "US.A" not in result, "Expected malformed dotted initials to be excluded."
    assert "US" in result, "Expected a valid uppercase token to remain in the result."


def test_extract_acronyms_extended_ignores_invalid_patterns():
    """Ignores lowercase, mixed case and single-letter tokens."""
    text = "A b Ab OnS nlp 9 are not valid acronyms."
    assert (
        extract_acronyms(text) == []
    ), "Expected extended mode to ignore invalid acronym patterns."


def test_extract_acronyms_boundary_and_punctuation_handling():
    """Correctly identifies acronyms at boundaries and next to punctuation."""
    text = "(ONS), NLP; ISO9001."
    assert extract_acronyms(text) == [
        "ONS",
        "NLP",
        "ISO9001",
    ], "Expected acronyms near punctuation and boundaries to be matched."


def test_extract_acronyms_duplicates_and_order_preserved():
    """Preserves duplicate acronyms and maintains original order."""
    text = "ONS works with ONS and NLP"
    assert extract_acronyms(text) == [
        "ONS",
        "ONS",
        "NLP",
    ], "Expected duplicate acronyms and original order to be preserved."


def test_extract_acronyms_non_string_and_empty_inputs():
    """Returns empty list for non-string, empty or whitespace-only input."""
    assert extract_acronyms(None) == [], "Expected None input to return no acronyms."
    assert extract_acronyms("") == [], "Expected empty input to return no acronyms."
    assert (
        extract_acronyms("   ") == []
    ), "Expected whitespace-only input to return no acronyms."


def test_extract_acronyms_long_token_not_split():
    """Does not split long uppercase tokens into partial matches."""
    text = "ABCDEF, A&B&C&D&E, A.B.C.D.E"
    result = extract_acronyms(text, extended=True)
    assert set(result) == {
        "ABCDEF",
        "A&B&C&D&E",
        "A.B.C.D.E",
    }, "Expected long uppercase tokens to remain intact rather than be split."


# ============================================================================
# Test get_syllable_count_per_word function
# ============================================================================


def test_get_syllable_count_per_word_single_word():
    """Verify syllable count for a single word."""
    text = "hello"
    result = get_syllable_count_per_word(text)

    assert result == [2], f"Expected [2] for {text!r}, got {result}"


def test_get_syllable_count_per_word_multiple_words():
    """Validate syllable counts across multiple words."""
    text = "hello world"
    result = get_syllable_count_per_word(text)

    assert result == [2, 1], f"Expected [2, 1] for {text!r}, got {result}"


def test_get_syllable_count_per_word_multi_syllable_words():
    """Ensure words with multiple syllables are counted correctly."""
    text = "question analysis whole"
    result = get_syllable_count_per_word(text)

    assert result == [2, 4, 1], f"Expected [2, 4, 1] for {text!r}, got {result}"


def test_get_syllable_count_per_word_ignores_extra_whitespace():
    """Confirm that extra whitespace between words is handled correctly."""
    text = "hello   world"
    result = get_syllable_count_per_word(text)

    assert result == [2, 1], f"Expected [2, 1] for {text!r}, got {result}"


def test_get_syllable_count_per_word_punctuation():
    """Verify syllable counts are returned for words with punctuation."""
    text = "Hello, world!"
    result = get_syllable_count_per_word(text)

    assert result == [2, 1], f"Expected [2, 1] for {text!r}, got {result}"


def test_get_syllable_count_per_word_empty_string():
    """Check that an empty string returns an empty list."""
    text = ""
    result = get_syllable_count_per_word(text)

    assert result == [], f"Expected [] for empty string, got {result}"


# ============================================================================
# Test get_avg_syllables_per_word function
# ============================================================================


def test_get_avg_syllables_per_word_returns_expected_average():
    """Check the average syllables per word calculation is as expected."""
    texts = [
        "Simple sentence",
        "This is a simple sentence",
        "More complex wording with variability",
        "A",
    ]

    expected = [2, 1.4, 2.4, 1]

    for i, text in enumerate(texts):
        assert (
            get_avg_syllables_per_word(text) == expected[i]
        ), f"Expected syllable average {expected[i]} for {text!r}."


def test_get_avg_syllables_per_word_matches_textstat_exactly():
    """Matches textstat's average syllables per word calculation."""
    texts = ["Simple sentence", "More complex wording with variability", "A"]

    for text in texts:
        assert get_avg_syllables_per_word(text) == textstat.avg_syllables_per_word(
            text
        ), f"Expected result to match textstat for {text!r}."


def test_get_avg_syllables_per_word_returns_float_type():
    """Returns a float value for valid text input."""
    text = "Short text"
    result = get_avg_syllables_per_word(text)

    assert isinstance(result, float), "Expected a float result for valid input text."


def test_get_avg_syllables_per_word_empty_string_returns_zero():
    """Returns 0.0 when input text is empty."""
    assert get_avg_syllables_per_word("") == 0.0, "Expected empty input to return zero."


def test_get_avg_syllables_per_word_none_input_returns_zero():
    """Returns 0.0 when input is None."""
    assert (
        get_avg_syllables_per_word(None) == 0.0
    ), "Expected None input to return zero."


def test_get_avg_syllables_per_word_non_string_input_returns_zero():
    """Returns 0.0 when input is not a string."""
    assert (
        get_avg_syllables_per_word(123) == 0.0
    ), "Expected non-string input to return zero."


# ============================================================================
# Test add_simple_language_columns function
# ============================================================================


def test_add_simple_language_columns_returns_expected_dataframe(
    simple_language_input_df,
    expected_simple_language_df,
):
    """Adds expected simple language metric columns and values."""
    result = add_simple_language_columns(
        simple_language_input_df,
        text_column="follow_up_question",
    )

    pd.testing.assert_frame_equal(result, expected_simple_language_df)
