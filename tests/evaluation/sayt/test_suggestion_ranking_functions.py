"""Tests for suggestion ranking functions in the survey_assist_eval package."""

import pandas as pd
import pytest

from survey_assist_eval.evaluation.sayt.suggestion_ranking_functions import (
    get_codes_from_suggestions,
    get_rank_of_first_matching_code,
    is_correct_codes_empty,
    rank_of_correct_code_in_suggestions,
    truncate_codes_columns,
    truncate_correct_codes,
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
# Test truncate_correct_codes function
# ============================================================================


def test_truncate_correct_codes_truncates_a_single_string():
    """A single correct code string should be truncated to the requested length."""
    result = truncate_correct_codes("12345", code_digit_match_length=3)

    assert result == "123", "Expected the string to be truncated to 3 characters."


def test_truncate_correct_codes_truncates_each_code_in_a_list():
    """Each code in a list should be truncated to the requested length."""
    result = truncate_correct_codes(["12345", "67890"], code_digit_match_length=3)

    assert set(result) == {
        "123",
        "678",
    }, "Expected each code in the list to be truncated to 3 characters."


def test_truncate_correct_codes_dedupes_list_after_truncation():
    """Truncating a list of codes should remove duplicates created by the truncation."""
    result = truncate_correct_codes(["1231", "1239"], code_digit_match_length=3)

    assert result == [
        "123"
    ], "Expected duplicate truncated codes to be de-duplicated into a single entry."


def test_truncate_correct_codes_handles_empty_list():
    """Truncating an empty list should return an empty list."""
    result = truncate_correct_codes([], code_digit_match_length=3)

    assert isinstance(result, list), "Expected a list to be returned."
    assert not result, "Expected an empty list to remain an empty list."


# ============================================================================
# Test truncate_codes_columns function
# ============================================================================


def test_truncate_codes_columns_adds_truncated_correct_codes_column():
    """A new correct_codes_col_truncated column should hold truncated correct codes."""
    df = pd.DataFrame({"correct_code": ["12345", "67890"]})

    result = truncate_codes_columns(
        df, code_digit_match_length=3, correct_codes_col="correct_code"
    )

    assert result["correct_code_truncated"].tolist() == [
        "123",
        "678",
    ], "Expected correct_code_truncated to hold each code truncated to 3 characters."
    assert result["correct_code"].tolist() == [
        "12345",
        "67890",
    ], "Expected the original correct_code column to remain unchanged."


def test_truncate_codes_columns_adds_truncated_retrieved_codes_column():
    """A new retrieved_codes_col_truncated column should hold truncated retrieved codes."""
    df = pd.DataFrame({"retrieved": [["12345", "67890"]]})

    result = truncate_codes_columns(
        df, code_digit_match_length=3, retrieved_codes_col="retrieved"
    )

    assert result["retrieved_truncated"].tolist() == [
        ["123", "678"]
    ], "Expected retrieved_truncated to hold each code truncated to 3 characters."


def test_truncate_codes_columns_keeps_duplicates_in_retrieved_codes():
    """Truncating retrieved codes should keep duplicates, unlike correct codes."""
    df = pd.DataFrame({"retrieved": [["1234", "1235"]]})

    result = truncate_codes_columns(
        df, code_digit_match_length=3, retrieved_codes_col="retrieved"
    )

    assert result["retrieved_truncated"].tolist() == [["123", "123"]], (
        "Expected retrieved_truncated to keep duplicate codes produced by "
        "truncation, preserving retrieval order and rank."
    )


def test_truncate_codes_columns_handles_both_columns_together():
    """Both correct and retrieved codes columns should be truncated when provided."""
    df = pd.DataFrame(
        {
            "correct_code": ["12345"],
            "retrieved": [["12345", "99999"]],
        }
    )

    result = truncate_codes_columns(
        df,
        code_digit_match_length=3,
        correct_codes_col="correct_code",
        retrieved_codes_col="retrieved",
    )

    assert (
        result["correct_code_truncated"].iloc[0] == "123"
    ), "Expected correct_code_truncated to be truncated to 3 characters."
    assert result["retrieved_truncated"].iloc[0] == [
        "123",
        "999",
    ], "Expected retrieved_truncated to be truncated to 3 characters."


def test_truncate_codes_columns_skips_columns_not_requested():
    """Columns should only be added when explicitly requested."""
    df = pd.DataFrame({"correct_code": ["12345"]})

    result = truncate_codes_columns(df, code_digit_match_length=3)

    assert "correct_code_truncated" not in result.columns, (
        "Expected no truncated correct-codes column to be added when "
        "correct_codes_col is not provided."
    )


def test_truncate_codes_columns_does_not_mutate_input():
    """The function should not modify the input DataFrame."""
    df = pd.DataFrame({"correct_code": ["12345"]})
    original_df = df.copy(deep=True)

    truncate_codes_columns(
        df, code_digit_match_length=3, correct_codes_col="correct_code"
    )

    assert df.equals(
        original_df
    ), "Expected truncate_codes_columns to leave the input DataFrame unchanged."
