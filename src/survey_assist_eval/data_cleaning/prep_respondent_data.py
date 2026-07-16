"""Prepare respondent data."""

import pandas as pd

MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
EDUCATION_COL = "level_of_education"


def respondent_data_to_dict(row: pd.Series):
    """Prepares a dictionary with collected respondent data."""
    respondent_data = {}
    if row.get(JOB_TITLE_COL, None) not in {None, "unknown", "", "-8", "-9"}:
        respondent_data["Job title"] = row[JOB_TITLE_COL]
    if row.get(JOB_DESCRIPTION_COL, None) not in {None, "unknown", "", "-8", "-9"}:
        respondent_data["Job description"] = row[JOB_DESCRIPTION_COL]
    if row.get(MERGED_INDUSTRY_DESC_COL, None) not in {None, "unknown", "", "-8", "-9"}:
        respondent_data["Company main activity"] = row[MERGED_INDUSTRY_DESC_COL]
    if row.get(EDUCATION_COL, None) not in {None, "unknown", "", "-8", "-9"}:
        respondent_data["Level of education"] = row[EDUCATION_COL]

    return respondent_data
