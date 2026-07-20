"""Tests for creating a respondent data dictionary."""

import pandas as pd

from survey_assist_eval.data_cleaning.prep_respondent_data import (
    respondent_data_to_dict,
)


def test_all_fields():
    """Test case, where all fields are provided.
    Expecting creating two dictionaries with four keys each.
    """
    df = pd.DataFrame(
        {
            "level_of_education": [-9, "5"],
            "soc2020_job_description": ["jd1", "jd2"],
            "merged_industry_desc": ["ind1", "ind2"],
            "soc2020_job_title": ["jt1", "jt2"],
        }
    )

    result = df.apply(respondent_data_to_dict, axis=1)

    expected = pd.Series(
        [
            {
                "Job title": "jt1",
                "Job description": "jd1",
                "Company main activity": "ind1",
                "Level of education": -9,
            },
            {
                "Job title": "jt2",
                "Job description": "jd2",
                "Company main activity": "ind2",
                "Level of education": "5",
            },
        ],
    )

    assert expected.equals(result)


def test_answer_not_provided():
    """Test case with no answers provided (marked as None, "unknown", "", "-8", or "-9").
    Expecting returning empty dictionaries.
    """
    df = pd.DataFrame(
        {
            "level_of_education": [None, "unknown", "", "-8", "-9"],
            "soc2020_job_description": [None, "unknown", "", "-8", "-9"],
            "merged_industry_desc": [None, "unknown", "", "-8", "-9"],
            "soc2020_job_title": [None, "unknown", "", "-8", "-9"],
        }
    )

    result = df.apply(respondent_data_to_dict, axis=1)

    expected = pd.Series([{}, {}, {}, {}, {}])

    assert expected.equals(result)


def test_some_answers_provided():
    """Test cases, where some responses are provided.
    Expecting dictionaries with only fields that were provided.
    """
    df = pd.DataFrame(
        {
            "soc2020_job_title": ["jt1", "-9", "-8", "unknown"],
            "level_of_education": ["-9", "edu2", "-8", "unknown"],
            "soc2020_job_description": ["-9", "-8", "jd3", "unknown"],
            "merged_industry_desc": ["-9", "-8", "unknown", "ind4"],
        }
    )

    result = df.apply(respondent_data_to_dict, axis=1)

    expected = pd.Series(
        [
            {"Job title": "jt1"},
            {"Level of education": "edu2"},
            {"Job description": "jd3"},
            {"Company main activity": "ind4"},
        ],
    )

    assert expected.equals(result)
