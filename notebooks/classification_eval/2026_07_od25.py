"""Notebook to prepare data for TLFS iteration evaluation.

It loads clerical coding data, SurveyAssist outputs, and CIMS outputs from the
configured evaluation bucket.
The bucket prefix is read from the .env file, where it should be stored as
EVALUATION_BUCKET_NAME (without gs:// and trailing /).

Disabled check for too long lines (f strings) and variables names (uppercase for constants)
"""

# ruff: noqa: S605
# pylint: disable=C0301,C0103,R0801,W0104

# %%
import os

import pandas as pd
from dotenv import load_dotenv

from survey_assist_eval.data_cleaning.code_standard import (
    get_clean_n_digit_codes,
)
from survey_assist_eval.data_cleaning.prep_data import (
    prep_model_codes,
)
from survey_assist_eval.evaluation.metrics import calc_simple_metrics

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")
print(f"Using bucket for data loading: {bucket_name}")

work_dir = "data/pipeline/od25_gemini25"
os.makedirs(work_dir, exist_ok=True)
file_name = f"gs://{bucket_name}/evaluation-pipeline/original_datasets/OD25/TLFS_OD25_Uncodable_Non_Disclosive.csv"

# %%
df = pd.read_csv(file_name).rename(
    columns={
        "sic07_free_text_main_activity": "sic2007_employee",
        "soc2020_job_title_main_job": "soc2020_job_title",
        "soc2020_job_description_main_job": "soc2020_job_description",
    }
)
df["sic2007_self_employed"] = "-9"
input_file = f"{work_dir}/TLFS_OD25_Uncodable_Non_Disclosive.csv"
df.to_csv(input_file, index=False)

# %%
os.system(
    f"scripts/sic_pipeline/run_full_pipeline.sh -p 2 -i {input_file} -o {work_dir}"
)

# %%
df = pd.read_parquet(f"{work_dir}/STG7.parquet")

# %%
df["clerical_codes"] = pd.Series([set() for _ in range(len(df))])

df_m = prep_model_codes(df, code_type="SIC")
df["sa_initial_codes"] = df_m["model_codes"]

df_m = prep_model_codes(
    df,
    code_type="SIC",
    codes_col="final_code",
    alt_codes_col="alt_sic_candidates_final",
)
df["sa_final_codes"] = df_m["model_codes"]

# %%
met = calc_simple_metrics(df)
print(met.report_metrics())

# %%
df["unambiguously_coded"] = df["sa_initial_codes"].map(len) == 1
# %%
df["sic_code"] = df["sa_initial_codes"].map(
    lambda x: next(iter(x)) if len(x) == 1 else None
)
# %%
out = df[
    [
        "unique_id",
        "sic2007_employee",
        "soc2020_job_title",
        "soc2020_job_description",
        "unambiguously_coded",
        "sic_code",
        "alt_sic_candidates",
    ]
].rename(columns={"sic2007_employee": "sic07_free_text_main_activity"})
out.to_csv(
    f"{work_dir}/TLFS_OD25_Uncodable_Non_Disclosive_with_SA_codes.csv", index=False
)

# %%
kb = pd.read_csv(
    "gs://ons-survey-assist-dev-evaluation-data/sic_knowledgebase/sic_kb_for_direct_lookup.csv"
)
# %%
clean_text = kb["description"].str.lower().str.strip()
msk = out["sic07_free_text_main_activity"].str.lower().str.strip().isin(clean_text)
out[msk]


# %%
out["sic_section"] = out.sic_code.map(
    lambda x: get_clean_n_digit_codes(x, n=0, code_type="SIC")[0]
).map(lambda x: next(iter(x))[0] if len(x) == 1 else None)
out.sic_section.value_counts()

# %%
