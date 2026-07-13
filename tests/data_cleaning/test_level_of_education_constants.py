"""Tests for expanding level of education."""

import pandas as pd

from survey_assist_eval.data_cleaning.level_of_education_constants import (
    expand_level_of_education,
)


def test_expand_education_int():
    """Test cases, where category provided as an integer is in the dictionary."""
    valid_cases_int = pd.DataFrame({"level_of_education": [-9, -8, 0, 1, 7]})

    result = valid_cases_int.apply(expand_level_of_education, axis=1)
    expected = [
        "The respondent did not provide information about their education.",
        "The survey did not ask the respondent question about their education.",
        "Unclassified / Don't know",
        "No Qualifications",
        "Higher Degree",
    ]

    for r in enumerate(result):
        assert r[1] == expected[r[0]]


def test_expand_education_str():
    """Test cases, where category provided as a string is in the dictionary."""
    valid_cases_str = pd.DataFrame({"level_of_education": ["-9", "-8", "0", "1", "7"]})

    result = valid_cases_str.apply(expand_level_of_education, axis=1)
    expected = [
        "The respondent did not provide information about their education.",
        "The survey did not ask the respondent question about their education.",
        "Unclassified / Don't know",
        "No Qualifications",
        "Higher Degree",
    ]

    for r in enumerate(result):
        assert r[1] == expected[r[0]]


def test_fail_not_valid_category_int():
    """Test case, where category provided as an integer is not in the dictionary."""
    not_valid_cases_int = pd.DataFrame({"level_of_education": [11, 12, 13]})

    result = not_valid_cases_int.apply(expand_level_of_education, axis=1)

    expected = ["11", "12", "13"]

    for r in enumerate(result):
        assert r[1] == expected[r[0]]


def test_fail_not_valid_category_str():
    """Test case, where category provided as a string is not in the dictionary."""
    not_valid_cases_str = pd.DataFrame(
        {"level_of_education": ["11", "12", "Higher education"]}
    )

    result = not_valid_cases_str.apply(expand_level_of_education, axis=1)

    expected = ["11", "12", "Higher education"]

    for r in enumerate(result):
        assert r[1] == expected[r[0]]
