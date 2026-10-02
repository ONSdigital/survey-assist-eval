"""Run classification for free text responses subset of Blaise SAYT pilot survey."""

# pylint: disable=C0103,R0801

# %%
import os

import pandas as pd
from dotenv import load_dotenv
from survey_assist_utils.logging import get_logger

# %%
# Load environment variables and set up logging
DATADROP_NUM = 1

load_dotenv()
bucket_name = os.getenv("PREPROD_DATA_BUCKET_NAME")
if not bucket_name:
    raise ValueError("PREPROD_DATA_BUCKET_NAME environment variable not set")

logger = get_logger("sayt_blaise_pilot_sic_classification")

input_data_xlsx = (
    f"gs://{bucket_name}/2026-08-tlfs-sayt-free-text/TLFS_drop_1_coded_data_for_SA.xlsx",
    f"gs://{bucket_name}/2026-08-tlfs-sayt-free-text/???.xlsx",
)[DATADROP_NUM - 1]

logger.info(
    "Analysing SIC classification from SAYT Blaise pilot survey",
    input_data=input_data_xlsx,
)


# %%
# Load the dataset (two sheets from the Excel file)
df_tlfs = pd.read_excel(input_data_xlsx, sheet_name="TLFS", dtype=str)
