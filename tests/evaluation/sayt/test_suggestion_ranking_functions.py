"""Tests for suggestion ranking functions in the survey_assist_eval package."""

import pandas as pd
import pytest

from survey_assist_eval.evaluation.sayt.suggestion_ranking_functions import (
    clean_codes_columns,
    get_code_length_from_type,
    get_codes_from_suggestions,
    get_rank_of_first_matching_code,
    is_correct_codes_empty,
    rank_of_correct_code_in_suggestions,
)

# ============================================================================
# Test get_code_length_from_type function
# ============================================================================


@pytest.mark.parametrize(
    "code_type,expected_length",
    [
        (None, 5),
        ("sic", 5),
        ("SIC", 5),
        ("Sic", 5),
        ("soc", 4),
        ("SOC", 4),
        ("Soc", 4),
    ],
    ids=[
        "default_sic",
        "lowercase_sic",
        "uppercase_sic",
        "mixedcase_sic",
        "lowercase_soc",
        "uppercase_soc",
        "mixedcase_soc",
    ],
)
def test_get_code_length_from_type_returns_correct_length(code_type, expected_length):
    """The function should return the correct code length for all code type variations."""
    length = (
        get_code_length_from_type()
        if code_type is None
        else get_code_length_from_type(code_type=code_type)
    )

    assert (
        length == expected_length
    ), f"Expected code_type '{code_type}' to return length {expected_length}."


@pytest.mark.parametrize(
    "invalid_code_type",
    [
        "invalid",
        "xyz",
        "",
        "soc_code",
        "sic_code",
        "SIC_CODE",
    ],
)
def test_get_code_length_from_type_raises_for_invalid_string_types(invalid_code_type):
    """Unsupported string code_type values should raise a ValueError."""
    with pytest.raises(ValueError, match="Unsupported code_type"):
        get_code_length_from_type(code_type=invalid_code_type)


@pytest.mark.parametrize(
    "non_string_code_type",
    [
        None,
        123,
        45.67,
        [],
        {},
        object(),
    ],
)
def test_get_code_length_from_type_raises_for_non_string_types(non_string_code_type):
    """Non-string code_type inputs should raise an error (AttributeError or TypeError)."""
    with pytest.raises((AttributeError, TypeError)):
        get_code_length_from_type(code_type=non_string_code_type)


def test_get_code_length_from_type_error_message_shows_valid_types():
    """The error message should list the valid code types."""
    with pytest.raises(ValueError) as exc_info:
        get_code_length_from_type(code_type="xyz")

    error_msg = str(exc_info.value)
    assert (
        "sic" in error_msg and "soc" in error_msg
    ), "Expected error message to include the list of valid code types."


# ============================================================================
# Test is_correct_codes_empty function
# ============================================================================


@pytest.mark.parametrize(
    "input_value,expected_empty",
    [
        ("1234", False),
        (["1234", "5678"], False),
    ],
    ids=[
        "non_empty_string",
        "non_empty_list",
    ],
)
def test_is_correct_codes_empty_with_non_empty_inputs(input_value, expected_empty):
    """Non-empty strings and lists should not be considered empty."""
    result = is_correct_codes_empty(input_value)

    assert (
        result is expected_empty
    ), f"Expected is_correct_codes_empty({input_value!r}) to be {expected_empty}."


@pytest.mark.parametrize(
    "input_value,expected_empty",
    [
        ("", True),
        (None, True),
        (float("nan"), True),
        ([], True),
    ],
    ids=[
        "empty_string",
        "none",
        "nan_float",
        "empty_list",
    ],
)
def test_is_correct_codes_empty_with_empty_or_none_inputs(input_value, expected_empty):
    """Empty strings, None, NaN, and empty lists should be considered empty."""
    result = is_correct_codes_empty(input_value)

    assert (
        result is expected_empty
    ), f"Expected is_correct_codes_empty({input_value!r}) to be {expected_empty}."


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


@pytest.mark.parametrize(
    "retrieved_codes,correct_codes,expected_rank",
    [
        (["1111", "2222", "1111"], "1111", 1.0),
        (["1111", "2222", "3333"], "3333", 3.0),
        (["1111", "2222", "3333"], ["3333", "4444"], 3),
        (["1111", "2222", "3333", "4444"], ["3333", "2222"], 2),
        (["1111", "2222", "3333"], {"3333", "4444"}, 3),
        (["1111", "2222", "3333"], ["2222"], 2),
    ],
    ids=[
        "first_position_with_duplicates",
        "rank_beyond_first_position",
        "list_of_correct_codes_first_match",
        "list_of_correct_codes_earliest_match",
        "set_of_correct_codes",
        "single_item_list_equivalence",
    ],
)
def test_get_rank_of_first_matching_code_normal_scenarios(
    retrieved_codes, correct_codes, expected_rank
):
    """Rank should find first matching position for various correct_codes types."""
    rank = get_rank_of_first_matching_code(retrieved_codes, correct_codes)

    assert (
        rank == pytest.approx(expected_rank)
        if isinstance(expected_rank, float)
        else rank == expected_rank
    ), f"Expected rank {expected_rank} but got {rank}."


def test_get_rank_of_first_matching_code_returns_none_when_code_not_found():
    """Rank should be None when the correct code is absent."""
    rank = get_rank_of_first_matching_code(["1111", "2222", "3333"], "4444")

    assert rank is None, (
        "Expected rank to be None when the correct code is absent from the retrieved "
        "list."
    )


def test_get_rank_of_first_matching_code_returns_none_when_list_has_no_match():
    """Rank should be None when no codes in the list are found."""
    rank = get_rank_of_first_matching_code(["1111", "2222", "3333"], ["4444", "5555"])

    assert rank is None, (
        "Expected rank to be None when none of the codes in the list match the "
        "retrieved codes."
    )


@pytest.mark.parametrize(
    "retrieved_codes,correct_codes,expected_rank",
    [
        ([None, "1111", "2222"], "1111", 2),
        (["1111", None, "2222"], "2222", 3),
        ([None, None, "3333"], "3333", 3),
    ],
    ids=[
        "none_at_start",
        "none_in_middle",
        "all_none_except_correct_code",
    ],
)
def test_get_rank_of_first_matching_code_with_none_in_retrieved(
    retrieved_codes, correct_codes, expected_rank
):
    """None values in retrieved codes list should be handled correctly."""
    rank = get_rank_of_first_matching_code(retrieved_codes, correct_codes)

    assert rank == expected_rank, (
        f"Expected rank {expected_rank} for retrieved_codes={retrieved_codes} "
        f"and correct_codes={correct_codes}, but got {rank}."
    )


@pytest.mark.parametrize(
    "retrieved_codes,correct_codes,expected_rank",
    [
        ([], "1111", None),
        ([], ["1111", "2222"], None),
        ([], set(), None),
        (["1111"], "1111", 1),
        (["2222"], "1111", None),
        ([None], "1111", None),
    ],
    ids=[
        "empty_with_string",
        "empty_with_list",
        "empty_with_set",
        "single_match",
        "single_no_match",
        "single_none",
    ],
)
def test_get_rank_of_first_matching_code_with_empty_or_single_element(
    retrieved_codes, correct_codes, expected_rank
):
    """Empty or single element retrieved codes should be handled correctly."""
    rank = get_rank_of_first_matching_code(retrieved_codes, correct_codes)

    assert rank == expected_rank, (
        f"Expected rank {expected_rank} for retrieved_codes={retrieved_codes} "
        f"and correct_codes={correct_codes}, but got {rank}."
    )


@pytest.mark.parametrize(
    "retrieved_codes,correct_codes,expected_rank",
    [
        (["1111", "2222"], None, None),
        (["1111", "2222", "3333"], [], None),
        (["1111", "2222", "3333"], set(), None),
        (["1111", "2222", "3333"], "", None),
    ],
    ids=[
        "none_as_correct_codes",
        "empty_list_as_correct_codes",
        "empty_set_as_correct_codes",
        "empty_string_as_correct_codes",
    ],
)
def test_get_rank_of_first_matching_code_with_none_or_empty_correct_codes(
    retrieved_codes, correct_codes, expected_rank
):
    """None or empty correct_codes should result in rank of None."""
    rank = get_rank_of_first_matching_code(retrieved_codes, correct_codes)

    assert rank == expected_rank, (
        f"Expected rank {expected_rank} for retrieved_codes={retrieved_codes} "
        f"and correct_codes={correct_codes}, but got {rank}."
    )


# ============================================================================
# Test get_codes_from_suggestions function
# ============================================================================


@pytest.mark.parametrize(
    "suggestions,code_type,expected_codes",
    [
        (["alpha 1234", "beta 5678"], "soc", ["1234", "5678"]),
        (["third 3333", "first 1111", "second 2222"], "soc", ["3333", "1111", "2222"]),
        (["some entry 12345"], None, ["12345"]),
    ],
    ids=[
        "extract_trailing_codes",
        "preserve_suggestion_order",
        "uses_default_code_type",
    ],
)
def test_get_codes_from_suggestions_normal_scenarios(
    suggestions, code_type, expected_codes
):
    """Codes should be extracted and order preserved from suggestion strings."""
    row = pd.Series({"suggestions": suggestions})

    if code_type is None:
        codes = get_codes_from_suggestions(row, suggestions_col="suggestions")
    else:
        codes = get_codes_from_suggestions(
            row, suggestions_col="suggestions", code_type=code_type
        )

    assert (
        codes == expected_codes
    ), f"Expected {expected_codes} but got {codes} for suggestions={suggestions}."


def test_get_codes_from_suggestions_returns_empty_list_for_no_suggestions():
    """An empty suggestions list should yield an empty list of codes."""
    row = pd.Series({"suggestions": []})

    codes = get_codes_from_suggestions(
        row, suggestions_col="suggestions", code_type="sic"
    )

    assert (
        codes == []
    ), "Expected no codes to be extracted from an empty suggestions list."


# ============================================================================
# Test rank_of_correct_code_in_suggestions function
# ============================================================================


@pytest.mark.parametrize(
    "suggestions,correct_code,expected_rank",
    [
        (["alpha 1111", "beta 2222", "gamma 3333"], "2222", 2),
        (["alpha 1111", "beta 2222", "gamma 3333"], ["3333", "4444"], 3),
    ],
    ids=[
        "single_correct_code_found",
        "list_of_correct_codes_first_match",
    ],
)
def test_rank_of_correct_code_in_suggestions_normal_scenarios(
    suggestions, correct_code, expected_rank
):
    """Rank should find the correct code position in suggestions."""
    row = pd.Series(
        {
            "suggestions_4chars_prefix": suggestions,
            "correct_sic_code": correct_code,
        }
    )

    rank = rank_of_correct_code_in_suggestions(
        row, num_chars=4, suggester_label="prefix", code_type="soc"
    )

    assert (
        rank == expected_rank
    ), f"Expected rank {expected_rank} but got {rank} for correct_code={correct_code}."


def test_rank_of_correct_code_in_suggestions_returns_none_when_not_found():
    """Rank should be None when the correct code is not among the suggestions."""
    row = pd.Series(
        {
            "suggestions_4chars_prefix": ["alpha 1111", "beta 2222"],
            "correct_sic_code": "9999",
        }
    )

    rank = rank_of_correct_code_in_suggestions(
        row, num_chars=4, suggester_label="prefix", code_type="soc"
    )

    assert (
        rank is None
    ), "Expected rank to be None when the correct code is absent from the suggestions."


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
        code_type="sic",
        correct_codes_col="my_correct_code",
    )

    assert rank == 1, (
        "Expected rank to be 1 when using a custom correct_codes_col that matches "
        "the only suggestion."
    )


# ============================================================================
# Test clean_codes_columns function
# ============================================================================


@pytest.mark.parametrize(
    "correct_code_values,expected_clean",
    [
        (["1111", "1231"], [{"111"}, {"123"}]),
        ([None, None], [set(), set()]),
    ],
    ids=[
        "truncate_to_match_length",
        "none_values_return_empty_set",
    ],
)
def test_clean_codes_columns_adds_clean_correct_codes_column(
    correct_code_values, expected_clean
):
    """A new correct_code_clean column should hold the set of cleaned correct codes."""
    df = pd.DataFrame({"correct_code": correct_code_values})

    result = clean_codes_columns(
        df, code_digit_match_length=3, code_type="soc", correct_codes_col="correct_code"
    )

    assert (
        result["correct_code_clean"].tolist() == expected_clean
    ), f"Expected correct_code_clean to be {expected_clean}."
    if correct_code_values != [None, None]:
        assert (
            result["correct_code"].tolist() == correct_code_values
        ), "Expected the original correct_code column to remain unchanged."


def test_clean_codes_columns_adds_clean_retrieved_codes_column():
    """retrieved_codes_col should get a _clean list column, with no _valid column left."""
    df = pd.DataFrame({"retrieved": [["1111", "1231"]]})

    result = clean_codes_columns(
        df, code_digit_match_length=3, code_type="soc", retrieved_codes_col="retrieved"
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
        df, code_digit_match_length=3, code_type="soc", retrieved_codes_col="retrieved"
    )

    assert result["retrieved_clean"].tolist() == [["123", "111", "123"]], (
        "Expected retrieved_clean to preserve retrieval order and keep duplicate "
        "codes produced by truncation."
    )


def test_clean_codes_columns_replaces_invalid_retrieved_codes_with_sentinel():
    """Retrieved codes not present in the valid set should be replaced with None."""
    df = pd.DataFrame({"retrieved": [["1111", "9999"]]})

    result = clean_codes_columns(
        df, code_digit_match_length=3, code_type="soc", retrieved_codes_col="retrieved"
    )

    assert result["retrieved_clean"].tolist() == [["111", None]], (
        "Expected invalid retrieved codes to be replaced with the None sentinel "
        "while valid codes are truncated normally."
    )


def test_clean_codes_columns_handles_missing_retrieved_codes():
    """A missing retrieved-codes value should produce an empty clean list."""
    df = pd.DataFrame({"retrieved": [None]})

    result = clean_codes_columns(
        df, code_digit_match_length=3, code_type="soc", retrieved_codes_col="retrieved"
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
        code_type="soc",
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
        df, code_digit_match_length=3, code_type="soc", retrieved_codes_col="retrieved"
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
        df, code_digit_match_length=3, code_type="soc", correct_codes_col="correct_code"
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
            code_type="soc",
            correct_codes_col="code",
            retrieved_codes_col="code",
        )


def test_clean_codes_columns_raises_when_both_columns_are_none():
    """A ValueError should be raised when both correct_codes_col and
    retrieved_codes_col are None.
    """
    df = pd.DataFrame({"correct_code": ["1111"], "retrieved": [["2222"]]})

    with pytest.raises(ValueError, match="or both None"):
        clean_codes_columns(df, code_digit_match_length=3, code_type="soc")


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
        code_type="soc",
        correct_codes_col="correct_code",
        retrieved_codes_col="retrieved",
    )
    twice = clean_codes_columns(
        once,
        code_digit_match_length=3,
        code_type="soc",
        correct_codes_col="correct_code",
        retrieved_codes_col="retrieved",
    )

    assert list(twice.columns) == list(
        once.columns
    ), "Expected no duplicate columns to be created when called again on its own output."
    assert twice["correct_code_clean"].tolist() == once["correct_code_clean"].tolist()
    assert twice["retrieved_clean"].tolist() == once["retrieved_clean"].tolist()
