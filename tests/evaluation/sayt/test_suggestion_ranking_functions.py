"""Tests for suggestion ranking functions in the survey_assist_eval package."""

import pandas as pd
import pytest

from survey_assist_eval.evaluation.sayt.suggestion_ranking_functions import (
    clean_codes_columns,
    get_codes_from_suggestions,
    get_rank_of_first_matching_code,
    is_correct_codes_empty,
    rank_of_correct_code_in_suggestions,
)

# ============================================================================
# Test is_correct_codes_empty function
# ============================================================================


def test_is_correct_codes_empty_returns_false_for_non_empty_string():
    """A non-empty string should not be considered empty."""
    assert (
        is_correct_codes_empty("1234") is False
    ), "Expected a non-empty string to not be considered empty."


def test_is_correct_codes_empty_returns_true_for_empty_string():
    """An empty string should be considered empty."""
    assert (
        is_correct_codes_empty("") is True
    ), "Expected an empty string to be considered empty."


def test_is_correct_codes_empty_returns_true_for_none():
    """None should be considered empty."""
    assert is_correct_codes_empty(None) is True, "Expected None to be considered empty."


def test_is_correct_codes_empty_returns_true_for_nan_float():
    """A NaN float scalar should be considered empty."""
    assert (
        is_correct_codes_empty(float("nan")) is True
    ), "Expected a NaN float scalar to be considered empty."


def test_is_correct_codes_empty_returns_false_for_non_empty_list():
    """A non-empty list of codes should not be considered empty."""
    assert (
        is_correct_codes_empty(["1234", "5678"]) is False
    ), "Expected a non-empty list to not be considered empty."


def test_is_correct_codes_empty_returns_true_for_empty_list():
    """An empty list should be considered empty."""
    assert (
        is_correct_codes_empty([]) is True
    ), "Expected an empty list to be considered empty."


def test_is_correct_codes_empty_does_not_error_on_nan_mixed_with_lists():
    """A NaN scalar should be handled safely even when other values are lists."""
    values = [["1234"], float("nan"), []]

    results = [is_correct_codes_empty(value) for value in values]

    assert results == [False, True, True], (
        "Expected NaN and empty list to be treated as empty, and non-empty list "
        "as not empty, without raising an error."
    )


# ============================================================================
# Test get_rank_of_first_matching_code function
# ============================================================================


def test_get_rank_of_first_matching_code_returns_first_matching_rank():
    """Rank should report the first position containing the correct code."""
    rank = get_rank_of_first_matching_code(["1111", "2222", "1111"], "1111")

    assert rank == pytest.approx(1.0), (
        "Expected rank to report the first matching position when duplicates appear "
        "later in the list."
    )


def test_get_rank_of_first_matching_code_returns_zero_when_code_not_found():
    """Rank should be None when the correct code is absent."""
    rank = get_rank_of_first_matching_code(["1111", "2222", "3333"], "4444")

    assert rank is None, (
        "Expected rank to be None when the correct code is absent from the retrieved "
        "list."
    )


def test_get_rank_of_first_matching_code_returns_rank_beyond_first_position():
    """Rank should reflect the first matching position when it is not first."""
    rank = get_rank_of_first_matching_code(["1111", "2222", "3333"], "3333")

    assert rank == pytest.approx(3.0), (
        "Expected rank to equal 3.0 when the correct code is first found in the "
        "third position."
    )


def test_get_rank_of_first_matching_code_works_with_list_of_correct_codes():
    """Rank should find first match when given a list of correct codes."""
    rank = get_rank_of_first_matching_code(["1111", "2222", "3333"], ["3333", "4444"])

    assert rank == 3, (
        "Expected rank to equal 3 when the first code in the list matches at "
        "position 3."
    )


def test_get_rank_of_first_matching_code_finds_first_match_in_list():
    """Rank should return the earliest matching position from a list of codes."""
    rank = get_rank_of_first_matching_code(
        ["1111", "2222", "3333", "4444"], ["3333", "2222"]
    )

    assert rank == 2, (
        "Expected rank to equal 2 when the earliest match from the list is at "
        "position 2."
    )


def test_get_rank_of_first_matching_code_returns_none_when_list_has_no_match():
    """Rank should be None when no codes in the list are found."""
    rank = get_rank_of_first_matching_code(["1111", "2222", "3333"], ["4444", "5555"])

    assert rank is None, (
        "Expected rank to be None when none of the codes in the list match the "
        "retrieved codes."
    )


def test_get_rank_of_first_matching_code_handles_empty_list_of_correct_codes():
    """Rank should be None when given an empty list of correct codes."""
    rank = get_rank_of_first_matching_code(["1111", "2222", "3333"], [])

    assert (
        rank is None
    ), "Expected rank to be None when the list of correct codes is empty."


def test_get_rank_of_first_matching_code_single_code_in_list():
    """Rank should work with a single-item list equivalently to a string."""
    rank_from_list = get_rank_of_first_matching_code(["1111", "2222", "3333"], ["2222"])
    rank_from_string = get_rank_of_first_matching_code(["1111", "2222", "3333"], "2222")

    assert (
        rank_from_list == rank_from_string == 2
    ), "Expected single-item list and string to produce the same rank."


def test_get_rank_of_first_matching_code_works_with_set_of_correct_codes():
    """Rank should find first match when given a set of correct codes."""
    rank = get_rank_of_first_matching_code(["1111", "2222", "3333"], {"3333", "4444"})

    assert rank == 3, (
        "Expected rank to equal 3 when the first code in the set matches at "
        "position 3."
    )


def test_get_rank_of_first_matching_code_handles_empty_set_of_correct_codes():
    """Rank should be None when given an empty set of correct codes."""
    rank = get_rank_of_first_matching_code(["1111", "2222", "3333"], set())

    assert (
        rank is None
    ), "Expected rank to be None when the set of correct codes is empty."


# ============================================================================
# Test get_codes_from_suggestions function
# ============================================================================


def test_get_codes_from_suggestions_extracts_trailing_code_from_each_suggestion():
    """Codes should be the trailing code_length characters of each suggestion."""
    row = pd.Series({"suggestions": ["alpha 1234", "beta 5678"]})

    codes = get_codes_from_suggestions(
        row, suggestions_col="suggestions", code_length=4
    )

    assert codes == ["1234", "5678"], (
        "Expected the trailing 4 characters of each suggestion string to be "
        "extracted as the code."
    )


def test_get_codes_from_suggestions_preserves_suggestion_order():
    """Extracted codes should be returned in the same order as the suggestions."""
    row = pd.Series({"suggestions": ["third 3333", "first 1111", "second 2222"]})

    codes = get_codes_from_suggestions(
        row, suggestions_col="suggestions", code_length=4
    )

    assert codes == [
        "3333",
        "1111",
        "2222",
    ], "Expected extracted codes to preserve the original suggestion order."


def test_get_codes_from_suggestions_returns_empty_list_for_no_suggestions():
    """An empty suggestions list should yield an empty list of codes."""
    row = pd.Series({"suggestions": []})

    codes = get_codes_from_suggestions(
        row, suggestions_col="suggestions", code_length=5
    )

    assert (
        codes == []
    ), "Expected no codes to be extracted from an empty suggestions list."


def test_get_codes_from_suggestions_uses_default_code_length():
    """The default code_length of 5 should be used when not specified."""
    row = pd.Series({"suggestions": ["some entry 12345"]})

    codes = get_codes_from_suggestions(row, suggestions_col="suggestions")

    assert codes == [
        "12345"
    ], "Expected the default code_length of 5 to extract the trailing 5 characters."


# ============================================================================
# Test rank_of_correct_code_in_suggestions function
# ============================================================================


def test_rank_of_correct_code_in_suggestions_returns_rank_for_single_correct_code():
    """Rank should reflect the position of the correct code in the suggestion column."""
    row = pd.Series(
        {
            "suggestions_4chars_prefix": ["alpha 1111", "beta 2222", "gamma 3333"],
            "correct_sic_code": "2222",
        }
    )

    rank = rank_of_correct_code_in_suggestions(
        row, num_chars=4, suggester_label="prefix", code_length=4
    )

    assert (
        rank == 2
    ), "Expected rank to be 2 when the correct code matches the second suggestion."


def test_rank_of_correct_code_in_suggestions_returns_none_when_not_found():
    """Rank should be None when the correct code is not among the suggestions."""
    row = pd.Series(
        {
            "suggestions_4chars_prefix": ["alpha 1111", "beta 2222"],
            "correct_sic_code": "9999",
        }
    )

    rank = rank_of_correct_code_in_suggestions(
        row, num_chars=4, suggester_label="prefix", code_length=4
    )

    assert (
        rank is None
    ), "Expected rank to be None when the correct code is absent from the suggestions."


def test_rank_of_correct_code_in_suggestions_works_with_list_of_correct_codes():
    """Rank should match against any code in a list of correct codes."""
    row = pd.Series(
        {
            "suggestions_4chars_prefix": ["alpha 1111", "beta 2222", "gamma 3333"],
            "correct_sic_code": ["3333", "4444"],
        }
    )

    rank = rank_of_correct_code_in_suggestions(
        row, num_chars=4, suggester_label="prefix", code_length=4
    )

    assert rank == 3, (
        "Expected rank to be 3 when the first matching code from the list of "
        "correct codes appears at the third suggestion."
    )


def test_rank_of_correct_code_in_suggestions_uses_custom_correct_codes_col():
    """A custom correct_codes_col name should be used to look up the correct code."""
    row = pd.Series(
        {
            "suggestions_5chars_semantic": ["alpha 11111"],
            "my_correct_code": "11111",
        }
    )

    rank = rank_of_correct_code_in_suggestions(
        row,
        num_chars=5,
        suggester_label="semantic",
        code_length=5,
        correct_codes_col="my_correct_code",
    )

    assert rank == 1, (
        "Expected rank to be 1 when using a custom correct_codes_col that matches "
        "the only suggestion."
    )


# ============================================================================
# Test clean_codes_columns function
# ============================================================================


def test_clean_codes_columns_adds_clean_correct_codes_column():
    """A new correct_code_clean column should hold the set of cleaned correct codes."""
    df = pd.DataFrame({"correct_code": ["1111", "1231"]})

    result = clean_codes_columns(
        df, code_digit_match_length=3, code_length=4, correct_codes_col="correct_code"
    )

    assert result["correct_code_clean"].tolist() == [
        {"111"},
        {"123"},
    ], "Expected correct_code_clean to hold each code truncated to 3 characters as a set."
    assert result["correct_code"].tolist() == [
        "1111",
        "1231",
    ], "Expected the original correct_code column to remain unchanged."


def test_clean_codes_columns_correct_codes_empty_returns_empty_set():
    """Missing correct-codes values should produce an empty clean set."""
    df = pd.DataFrame({"correct_code": [None, ""]})

    result = clean_codes_columns(
        df, code_digit_match_length=3, code_length=4, correct_codes_col="correct_code"
    )

    assert result["correct_code_clean"].tolist() == [
        set(),
        set(),
    ], "Expected missing correct codes to produce an empty clean set."


def test_clean_codes_columns_adds_clean_retrieved_codes_column():
    """retrieved_codes_col should get a _clean list column, with no _valid column left."""
    df = pd.DataFrame({"retrieved": [["1111", "1231"]]})

    result = clean_codes_columns(
        df, code_digit_match_length=3, code_length=4, retrieved_codes_col="retrieved"
    )

    assert result["retrieved_clean"].tolist() == [
        ["111", "123"]
    ], "Expected retrieved_clean to hold each code truncated to 3 characters, in order."
    assert (
        "retrieved_valid" not in result.columns
    ), "Expected the intermediate retrieved_valid column to be dropped from the result."


def test_clean_codes_columns_keeps_duplicates_and_order_in_retrieved_clean():
    """retrieved_clean should preserve original order and keep duplicates."""
    df = pd.DataFrame({"retrieved": [["1231", "1111", "1231"]]})

    result = clean_codes_columns(
        df, code_digit_match_length=3, code_length=4, retrieved_codes_col="retrieved"
    )

    assert result["retrieved_clean"].tolist() == [["123", "111", "123"]], (
        "Expected retrieved_clean to preserve retrieval order and keep duplicate "
        "codes produced by truncation."
    )


def test_clean_codes_columns_replaces_invalid_retrieved_codes_with_sentinel():
    """Retrieved codes not present in the valid set should be replaced with None."""
    df = pd.DataFrame({"retrieved": [["1111", "9999"]]})

    result = clean_codes_columns(
        df, code_digit_match_length=3, code_length=4, retrieved_codes_col="retrieved"
    )

    assert result["retrieved_clean"].tolist() == [["111", None]], (
        "Expected invalid retrieved codes to be replaced with the None sentinel "
        "while valid codes are truncated normally."
    )


def test_clean_codes_columns_handles_missing_retrieved_codes():
    """A missing retrieved-codes value should produce an empty clean list."""
    df = pd.DataFrame({"retrieved": [None]})

    result = clean_codes_columns(
        df, code_digit_match_length=3, code_length=4, retrieved_codes_col="retrieved"
    )

    assert result["retrieved_clean"].tolist() == [
        []
    ], "Expected a missing retrieved-codes value to produce an empty clean list."


def test_clean_codes_columns_handles_both_columns_together():
    """Both correct and retrieved codes columns should be cleaned when provided."""
    df = pd.DataFrame(
        {
            "correct_code": ["1231"],
            "retrieved": [["1231", "9999"]],
        }
    )

    result = clean_codes_columns(
        df,
        code_digit_match_length=3,
        code_length=4,
        correct_codes_col="correct_code",
        retrieved_codes_col="retrieved",
    )

    assert result["correct_code_clean"].iloc[0] == {
        "123"
    }, "Expected correct_code_clean to hold the cleaned correct code as a set."
    assert result["retrieved_clean"].iloc[0] == [
        "123",
        None,
    ], "Expected retrieved_clean to hold each retrieved code truncated to 3 characters."


def test_clean_codes_columns_handles_both_columns_together_with_block_section_retrieved():
    """Both columns should be cleaned together using SIC block-section logic and sentinels."""
    df = pd.DataFrame(
        {
            "correct_code": ["10310"],
            "retrieved": [["10310", "01110", "01110", "00000"]],
        }
    )

    result = clean_codes_columns(
        df,
        code_digit_match_length=0,
        code_length=5,
        code_type="sic",
        correct_codes_col="correct_code",
        retrieved_codes_col="retrieved",
    )

    assert result["correct_code_clean"].iloc[0] == {
        "C"
    }, "Expected correct_code_clean to hold the cleaned SIC block section as a set."
    assert result["retrieved_clean"].iloc[0] == [
        "C",
        "A",
        "A",
        None,
    ], "Expected retrieved_clean to preserve order and replace invalid SIC codes with None."


def test_clean_codes_columns_skips_columns_not_requested():
    """Columns should only be added when explicitly requested."""
    df = pd.DataFrame({"retrieved": [["1111"]]})

    result = clean_codes_columns(
        df, code_digit_match_length=3, code_length=4, retrieved_codes_col="retrieved"
    )

    assert "correct_code_clean" not in result.columns, (
        "Expected no cleaned correct-codes column to be added when "
        "correct_codes_col is not provided."
    )


def test_clean_codes_columns_does_not_mutate_input():
    """The function should not modify the input DataFrame."""
    df = pd.DataFrame({"correct_code": ["1111"]})
    original_df = df.copy(deep=True)

    clean_codes_columns(
        df, code_digit_match_length=3, code_length=4, correct_codes_col="correct_code"
    )

    assert df.equals(
        original_df
    ), "Expected clean_codes_columns to leave the input DataFrame unchanged."


def test_clean_codes_columns_raises_when_columns_are_the_same():
    """A ValueError should be raised when correct_codes_col equals retrieved_codes_col."""
    df = pd.DataFrame({"code": ["1111"]})

    with pytest.raises(ValueError, match="both cannot be the same value"):
        clean_codes_columns(
            df,
            code_digit_match_length=3,
            code_length=4,
            correct_codes_col="code",
            retrieved_codes_col="code",
        )


def test_clean_codes_columns_raises_when_both_columns_are_none():
    """A ValueError should be raised when both correct_codes_col and
    retrieved_codes_col are None.
    """
    df = pd.DataFrame({"correct_code": ["1111"], "retrieved": [["2222"]]})

    with pytest.raises(ValueError, match="or both None"):
        clean_codes_columns(df, code_digit_match_length=3, code_length=4)


def test_clean_codes_columns_is_safe_to_call_again_on_its_own_output():
    """Calling clean_codes_columns again on already-cleaned output should not error
    or duplicate columns (e.g. when a notebook cell is rerun on the same DataFrame).
    """
    df = pd.DataFrame(
        {
            "correct_code": ["1231"],
            "retrieved": [["1231", "9999"]],
        }
    )

    once = clean_codes_columns(
        df,
        code_digit_match_length=3,
        code_length=4,
        correct_codes_col="correct_code",
        retrieved_codes_col="retrieved",
    )
    twice = clean_codes_columns(
        once,
        code_digit_match_length=3,
        code_length=4,
        correct_codes_col="correct_code",
        retrieved_codes_col="retrieved",
    )

    assert list(twice.columns) == list(
        once.columns
    ), "Expected no duplicate columns to be created when called again on its own output."
    assert twice["correct_code_clean"].tolist() == once["correct_code_clean"].tolist()
    assert twice["retrieved_clean"].tolist() == once["retrieved_clean"].tolist()
