"""This notebook evaluates SOC classification on a sample of 4,000 dataset."""

# ruff: noqa: S605
# pylint: disable=C0103,R0801

# %%
import os

import pandas as pd
from dotenv import load_dotenv

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")
print(f"Using bucket for data loading: {bucket_name}")

work_folder = "data/pipeline/soc_4k"
os.makedirs(work_folder, exist_ok=True)
output_folder = f"gs://{bucket_name}/evaluation-pipeline/soc_4k/one_prompt_v1"  # %%
input_data_file = (
    f"gs://{bucket_name}/evaluation-pipeline/original_datasets/soc_4k/"
    + "non-disclosive_with_SOC_code.csv"
)


# %%
# call evaluation pipeline if needed
if not os.path.exists(f"{output_folder}/STG4.parquet"):
    print("Running evaluation pipeline...")
    formatted_file = f"{work_folder}/formatted_non-disclosive_with_SOC_code.csv"
    df = pd.read_csv(input_data_file)
    df = df.rename(
        columns={
            "soc2020_job_title_main_job": "soc2020_job_title",
            "soc2020_job_description_main_job": "soc2020_job_description",
            "sic2007_employed_main_job": "sic2007_employee",
            "sic2007_self_employed_main_job": "sic2007_self_employed",
            "level_of_highest_qual_dv": "level_of_education",
        }
    ).drop(columns=["Unnamed: 0"])
    df.to_csv(formatted_file, index=False)
    os.system(
        f"./scripts/soc_pipeline/run_full_pipeline.sh -p 1 -i {formatted_file} -o {work_folder}"
    )
    # move output to bucket
    os.system(f"gsutil cp {work_folder}/STG4.parquet {output_folder}/STG4.parquet")
else:
    print("Evaluation pipeline already run, loading results...")

df = pd.read_parquet(f"{output_folder}/STG4.parquet")
print(df.head())

# %%
