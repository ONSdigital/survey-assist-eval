"""This notebook evaluates SOC classification on a sample of 4k+ dataset."""

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

input_data_file = (
    f"gs://{bucket_name}/evaluation-pipeline/original_datasets/soc_4k/"
    + "soc_4k_test_data.parquet"
)

# %%
# call evaluation pipeline if needed
work_folder = "data/pipeline/soc_4k"
os.makedirs(work_folder, exist_ok=True)

if not os.path.exists(f"{work_folder}/STG4.parquet"):  # this doesn't work with GCS
    print("Running evaluation pipeline...")
    os.system(
        f"./scripts/soc_pipeline/run_full_pipeline.sh -p 2 -i {input_data_file} -o {work_folder}"
    )
else:
    print("Evaluation pipeline already run, loading results...")

# %%
df = pd.read_parquet(f"{work_folder}/STG4.parquet")
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


# %% ================================
# test new prompt that returns just top match and likelihood, and reasoning
work_folder = "data/pipeline/soc_4k_top_one"
os.makedirs(work_folder, exist_ok=True)

print("Running evaluation pipeline...")
os.system(
    f"./scripts/soc_pipeline/run_full_pipeline.sh -p 1 -i {input_data_file} -o {work_folder}"
)

# %%
df = pd.read_parquet(f"{work_folder}/STG2.parquet")

# subset when working with intermediate outputs
df_sub = df[df.reasoning.notna()]
df_sub["match"] = df_sub.soc2020_code == df_sub.initial_code

print(f"Total rows: {len(df_sub)}")
print("-" * 20)
for label, lh in [("Low", 0.6), ("Medium", 0.8), ("High", 0.9)]:
    print(f"Confidence level: {label} ({lh})")
    print(f"Codability: {(df_sub.likelihood >= lh).mean():.0%}")
    print(f"Accuracy: {(df_sub[df_sub.likelihood >= lh].match).mean():.0%}")
    print("-" * 20)

# %%
df_sub["distance"] = df_sub.apply(
    lambda row: row["semantic_search_results"][0]["distance"], axis=1
)
df_sub["sem_match"] = df_sub.apply(
    lambda row: row["semantic_search_results"][0]["code"] == row["soc2020_code"], axis=1
)
for label, dd in [
    ("Low", 0.35),
    ("Medium", 0.25),
    ("Medium-High", 0.15),
    ("High", 0.1),
    ("Very High", 0.05),
]:
    print(f"Confidence level: {label} ({dd})")
    print(f"Codability: {(df_sub.distance <= dd).mean():.0%}")
    print(f"Accuracy: {(df_sub[df_sub.distance <= dd].sem_match).mean():.0%}")
    print("-" * 20)

# %%
