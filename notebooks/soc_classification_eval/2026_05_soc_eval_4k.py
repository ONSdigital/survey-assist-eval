"""This notebook evaluates SOC classification on a sample of 4,000 dataset."""

# ruff: noqa: S605
# pylint: disable=C0103,R0801

# %%
import os

import pandas as pd
from dotenv import load_dotenv

from survey_assist_eval.data_cleaning.prep_data import (
    prep_clerical_codes,
    prep_model_codes,
)
from survey_assist_eval.evaluation.metrics import (
    calc_simple_metrics,
)

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")
print(f"Using bucket for data loading: {bucket_name}")

work_folder = "data/pipeline/soc_4k"
os.makedirs(work_folder, exist_ok=True)
output_folder = f"gs://{bucket_name}/evaluation-pipeline/soc_4k/two_prompt_v1"  # %%
input_data_file = (
    f"gs://{bucket_name}/evaluation-pipeline/original_datasets/soc_4k/"
    + "soc_4k_test_data.parquet"
)


# %%
# call evaluation pipeline if needed
if not os.path.exists(f"{output_folder}/STG4.parquet"):  # this doesn't work with GCS
    print("Running evaluation pipeline...")
    os.system(
        f"./scripts/soc_pipeline/run_full_pipeline.sh -p 2 -i {input_data_file} -o {work_folder}"
    )
    # move output to bucket
    os.system(f"gsutil cp {work_folder}/STG4.parquet {output_folder}/STG4.parquet")
else:
    print("Evaluation pipeline already run, loading results...")

# %%
df = pd.read_parquet(f"{output_folder}/STG4.parquet")
# df = pd.read_parquet(f"{work_folder}/STG2.parquet")
print(df.head())

# %%
df_clerical = prep_clerical_codes(
    df.rename(columns={"final_uuid": "unique_id"}),
    code_type="SOC",
    clerical_col="soc2020_code",
    digits=4,
    out_col="clerical_codes",
)
df_model = prep_model_codes(
    df.rename(columns={"final_uuid": "unique_id"}),
    code_type="SOC",
    codes_col="initial_code",
    alt_codes_col="alt_soc_candidates",
    digits=4,
    out_col="model_codes",
)
# %%
metrics_summary = calc_simple_metrics(
    df_clerical.merge(df_model),
    truth_col="clerical_codes",
    initial_model_col="model_codes",
    final_model_col=None,
)
print(metrics_summary.report_metrics())
