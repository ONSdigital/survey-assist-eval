"""In this notebook we investigate common issues with SOC classification on the ashe kb datsaset."""

# pylint: disable=C0103, R0801

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
soc_kb_classifed_file = (
    f"gs://{bucket_name}/soc_knowledgebase/wip_data/soc_kb_top_one_STG2.parquet"
)

# %%
df = pd.read_parquet(soc_kb_classifed_file)
column_names = {
    "unique_id": "unique_id",
    "soc2020_job_title": "job_title",
    "clerical_soc": "clerical_code",
    "initial_code": "model_code",
    "likelihood": "likelihood",
    "reasoning": "reasoning",
    "semantic_search_results": "semantic_search_results",
}
df = df.rename(columns=column_names)[list(column_names.values())]
df["clerical_code"] = df["clerical_code"].astype(str)
df["match"] = df["clerical_code"] == df["model_code"]
print(df.head())

# %%
print(df.groupby(["likelihood"]).match.aggregate(["count", "mean"]).reset_index())
msk = (df.likelihood > 0.75) & ~df.match  # noqa: PLR2004
sub_df = df[msk].copy().reset_index(drop=True)

# %%
pairs = (
    sub_df.groupby(["clerical_code", "model_code"])
    .agg(
        count=("unique_id", "count"),
        job_titles=("job_title", list),
    )
    .reset_index()
    .sort_values("count", ascending=False)
)
print(pairs.head(5))


# %%
# quick look at n.e.c. using last digit of SOC code
df["clerical_nec"] = df["clerical_code"].str[-1] == "9"
df["model_nec"] = df["model_code"].str[-1] == "9"

print(
    f"clerical_nec: {df.clerical_nec.mean():.2%}, model_nec (all): {df.model_nec.mean():.2%}"
)
model_nec = df.groupby("likelihood").model_nec.mean().reset_index()
model_nec["model_nec"] = model_nec["model_nec"].apply(lambda x: f"{x:.2%}")
print(model_nec)


# %%
