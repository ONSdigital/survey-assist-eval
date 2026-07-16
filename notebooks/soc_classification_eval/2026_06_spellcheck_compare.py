"""This notebook evaluates SOC classification on a sample of 4k+ dataset."""

# pylint: disable=C0103,R0801

# %%
import os
import re

import pandas as pd
from dotenv import load_dotenv

from survey_assist_eval.data_cleaning.prep_data import (
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

# %%
sic_dirs = {
    "with": "data/pipeline/sic_2k_spellcheck",
    "without": "data/pipeline/sic_2k_spellcheck_off",
}
soc_dirs = {
    "with": "data/pipeline/soc_4k_spellcheck",
    "without": "data/pipeline/soc_4k_spellcheck_off",
}


# %%
sic_dfs = {x: pd.read_parquet(f"{y}/STG2.parquet") for x, y in sic_dirs.items()}


# take codes as sets
for lab, df in sic_dfs.items():
    print(f"Processing data {lab} spellcheck for SIC...")
    df["clerical_codes"] = df["clerical_codes"].apply(set)
    df_m = prep_model_codes(df, code_type="SIC")
    df["sa_initial_codes"] = df_m["model_codes"]
    metric = calc_simple_metrics(df)
    print(metric.report_metrics())

# %%

soc_dfs = {x: pd.read_parquet(f"{y}/STG2.parquet") for x, y in soc_dirs.items()}


def combine_alt_codes(in_df, threshold=0.75):
    """Fill alt_codes column with empty sets if not present."""
    out_col = in_df["initial_code"].apply(
        lambda x: set() if pd.isna(x) else {x} if isinstance(x, str) else set(x)
    )
    msk = in_df["initial_likelihood"] < threshold
    out_col[msk] = in_df.loc[msk, "alt_soc_candidates"].apply(
        lambda x: set(re.findall(r"\d+", x)) if pd.notna(x) else set()
    )
    return out_col


for lab, df in soc_dfs.items():
    print(f"Processing data {lab} spellcheck for SOC...")
    df["clerical_codes"] = df["clerical_codes"].apply(set)
    df["sa_initial_codes"] = combine_alt_codes(df)
    metric = calc_simple_metrics(df)
    print(metric.report_metrics())


# %%
