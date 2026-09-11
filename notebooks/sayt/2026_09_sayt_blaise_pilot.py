"""Run classification for free text responses subset of Blaise SAYT pilot survey."""

# ruff: noqa: S605
# pylint: disable=C0103

# %%
import os

import pandas as pd
from dotenv import load_dotenv
from survey_assist_utils.logging import get_logger

# %%
# Load environment variables and set up logging
load_dotenv()
bucket_name = os.getenv("PREPROD_DATA_BUCKET_NAME")
if not bucket_name:
    raise ValueError("PREPROD_DATA_BUCKET_NAME environment variable not set")

logger = get_logger("sayt_blaise_pilot_sic_classification")
work_folder = f"gs://{bucket_name}/2026-08-tlfs-sayt-free-text/tmp"
input_data_xlsx = (
    f"gs://{bucket_name}/2026-08-tlfs-sayt-free-text/SA_coding_sheet_01.xlsx"
)
logger.info(
    "Processing SIC classification for SAYT Blaise pilot survey",
    input_data=input_data_xlsx,
    work_folder=work_folder,
)


# %%
# Load the dataset (two sheets from the Excel file)
df_tlfs = pd.read_excel(input_data_xlsx, sheet_name="TLFS", dtype=str)
df_tlfs["case_person_id"] = df_tlfs["CaseID"] + "_" + df_tlfs["PersonID"]
df_tlfs["unique_id"] = (
    "tlfs"
    + "_"
    + df_tlfs["case_person_id"].astype(str)
    + "_"
    + (df_tlfs.index + 2).astype(str)
)
df_tlfs = df_tlfs[df_tlfs["case_person_id"].notna()].reset_index(drop=True)
print(df_tlfs.describe().T)

df_fus = pd.read_excel(input_data_xlsx, sheet_name="FUS", dtype=str)
df_fus["unique_id"] = (
    "fus"
    + "_"
    + df_fus["case_person_id"].astype(str)
    + "_"
    + (df_fus.index + 2).astype(str)
)
print(df_fus.describe().T)


# %%
# Check ID properties/issues
msk = df_fus.case_person_id.isin(df_tlfs.case_person_id)
logger.info(f"Number of FUS unique_ids not in TLFS: {sum(~msk)}")

msk_fus = df_fus.case_person_id.duplicated(keep=False)
if msk_fus.any():
    logger.warning(f"Duplicated FUS case_person_ids: {msk_fus.sum()}")

msk_tlfs = df_tlfs.case_person_id.duplicated(keep=False)
if msk_tlfs.any():
    logger.warning(f"Duplicated TLFS case_person_ids: {msk_tlfs.sum()}")

# %%
# Prep data for pipeline (column names)
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
INDUSTRY_DESCR_COL = "sic2007_employee"
SELF_EMPLOYED_DESC_COL = "sic2007_self_employed"
payload_cols = [
    JOB_TITLE_COL,
    JOB_DESCRIPTION_COL,
    INDUSTRY_DESCR_COL,
    SELF_EMPLOYED_DESC_COL,
]

df = pd.concat(
    [
        df_tlfs.copy().rename(
            columns={
                "soc2020_job_title_main_job": JOB_TITLE_COL,
                "soc2020_job_description_main_job": JOB_DESCRIPTION_COL,
                "sic2007_employed_main_job": INDUSTRY_DESCR_COL,
                "sic2007_self_employed_main_job": SELF_EMPLOYED_DESC_COL,
            }
        ),
        df_fus.copy().rename(
            columns={
                "Main_activity": INDUSTRY_DESCR_COL,
            }
        ),
    ],
    ignore_index=True,
)[["unique_id", *payload_cols]]
print(df.describe().T)

all_missing = pd.Series(True, index=df.index)
for col in payload_cols:
    df[col] = df[col].str.capitalize()
    # capitalise for consistency, but not needed anymore (see spellcheck issue)
    all_missing = all_missing & (df[col].isna() | df[col] == "-9")
if all_missing.any():
    logger.warning(f"Rows with all relevant columns missing: {all_missing.sum()}")
    print(df[all_missing])

input_data_file = work_folder + "/prep_input_data.parquet"
df.to_parquet(input_data_file, index=False)


# %%
# Run pipeline!
# Only STG1 and STG2 are needed, so you may want to edit the pipeline script to skip the rest
os.system(
    f"./scripts/sic_pipeline/run_full_pipeline.sh -p 2 -i {input_data_file} -o {work_folder}"
)

# %%
# Postprocessing the pipeline output
out_df = pd.read_parquet(work_folder + "/STG2.parquet").rename(
    columns={"alt_sic_candidates": "alt_codes"}
)
alt_msk = out_df["initial_code"].isna() | (out_df["initial_code"] == "")
out_df["alt_sic_candidates"] = ""
out_df.loc[alt_msk, "alt_sic_candidates"] = out_df.loc[alt_msk, "alt_codes"].apply(
    lambda x: [y["code"] for y in x]
)


# %%
# Attach model codes to TLFS and FUS dataframe
out_df_tlfs = (
    df_tlfs[df_tlfs.columns.difference(["initial_code", "alt_sic_candidates"])]
    .copy()
    .merge(
        out_df[["unique_id", "initial_code", "alt_sic_candidates"]].copy(),
        on="unique_id",
        how="left",
    )
)
tlfs_output_file = input_data_xlsx.replace(".xlsx", "_tlfs.csv")
out_df_tlfs[df_tlfs.columns].to_csv(tlfs_output_file, index=False, quoting=False)

out_df_fus = (
    df_fus[df_fus.columns.difference(["initial_code", "alt_sic_candidates"])]
    .copy()
    .merge(
        out_df[["unique_id", "initial_code", "alt_sic_candidates"]].copy(),
        on="unique_id",
        how="left",
    )
)
fus_output_file = input_data_xlsx.replace(".xlsx", "_fus.csv")
out_df_fus[df_fus.columns].to_csv(fus_output_file, index=False, quoting=False)

logger.info(
    "SIC Classification attached to TLFS and FUS dataframes.",
    tlfs_output_file=tlfs_output_file,
    fus_output_file=fus_output_file,
)

# %%
