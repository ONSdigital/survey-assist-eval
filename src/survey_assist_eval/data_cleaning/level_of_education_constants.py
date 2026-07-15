# pylint: disable=C0301
"""Enables expanding descriptions for the categorical level of education."""

import pandas as pd

LEVEL_OF_EDUCATION = {
    "-9": "The respondent did not provide information about their education.",
    "-8": "The survey did not ask the respondent question about their education.",
    "0": "Unclassified / Don't know",
    "1": "No Qualifications",
    "2": "GCSE Grades D-G or Below",
    "3": "GCSE Grades A*-C or Equivalent",
    "4": "GCE A Level or Equivalent",
    "5": "Further Education",
    "6": "Undergraduate degree or equivalent",
    "7": "Higher Degree",
}


def expand_level_of_education(row: pd.Series) -> str:
    """Expands the level of education to the qualification description.

    Args:
        row (pd.Series): A row from the input DataFrame containing level of education.

    Returns:
        description (str): The expanded descriptions.

    """
    education_coded = str(row["level_of_education"])
    if education_coded in LEVEL_OF_EDUCATION:
        return LEVEL_OF_EDUCATION[education_coded]
    return education_coded
