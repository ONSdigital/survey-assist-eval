# pylint: disable=C0301

"""Contains constants used in pipeline."""

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
