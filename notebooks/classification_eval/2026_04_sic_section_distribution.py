"""Notebook to compare distribution at SIC section level between CC, SA and CIMS codings.

Loads preprocessed data with clerical, SA and CIMS codings,
calculates codability levels, and visualises the distribution of codings at SIC section level.
Expects environment variable EVALUATION_BUCKET_NAME to be set.

"""

# pylint: disable=C0301,C0103,R0801

# %%
import os

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from scipy.stats import binomtest

from survey_assist_eval.data_cleaning.prep_data import get_clean_n_digit_codes

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")
print(f"Using bucket for data loading: {bucket_name}")

out_dir = "data/figures/tlfs_it11/"  # needs local folder unfortunately, set to None to skip saving
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

work_folder = f"gs://{bucket_name}/evaluation-pipeline/two_prompt_pipeline/2026_03_tlfs_it11_gemini25_europe_west9/"

# %% read data prepared in 2026_02_prep_tlfs_iteration_data.py script
combined_df = pd.read_parquet(f"{work_folder}sa_cc_cims_combined.parquet")
# ensure code columns are sets (they were saved as arrays in parquet)
for col in [col for col in combined_df.columns if "codes" in col.lower()]:
    print(f"Converting {col} to sets...")
    combined_df[col] = combined_df[col].apply(set)


# %%
# valuea counts by section (digits=0)
renaming = {
    "cc_initial_codes": "Clerical coding",
    "sa_initial_codes": "SurveyAssist",
    # "sa_without_kb_initial_codes": "SurveyAssist without KB",
    "cims_initial_codes": "CIMS",
}
combined_df = combined_df.rename(columns=renaming)
columns_to_compare = list(renaming.values())
df_section = {}
for i, col in enumerate(columns_to_compare):
    print(f"Processing section distribution for column {col}...")
    df_section[col] = (
        combined_df[col]
        .map(
            lambda x: (
                get_clean_n_digit_codes(x, n=0, code_type="SIC")[0]
                if pd.notna(x)
                else {}
            )
        )
        .map(lambda x: next(iter(x)) if len(x) == 1 else None)
        .dropna()
        .to_frame(name="sic_section")
    )
    df_section[col]["source"] = col
    df_section[col]["ind"] = i  # for ordering traces in plot
    # group B+D+E and R+S+T to align with LCF reporting
    df_section[col]["sic_section"] = df_section[col]["sic_section"].replace(
        {
            "B": "B,D,E",
            "D": "B,D,E",
            "E": "B,D,E",
            "R": "R,S,T",
            "S": "R,S,T",
            "T": "R,S,T",
        }
    )

plot_df_section = (
    pd.concat(df_section.values(), ignore_index=True)
    .dropna(subset=["sic_section"])
    .groupby(
        [
            "sic_section",
            "ind",
            "source",
        ]
    )
    .size()
    .reset_index()
    .rename(columns={0: "count"})
)

# %%
# add expected distribution from ONS LFS data
# source: https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/datasets/employmentbyindustryemp13
lfs = {
    "A": 310,
    "B,D,E": 582,
    "C": 2614,
    "F": 2054,
    "G": 3614,
    "H": 1646,
    "I": 1910,
    "J": 1679,
    "K": 1360,
    "L": 403,
    "M": 2947,
    "N": 1436,
    "O": 2917,
    "P": 3705,
    "Q": 4896,
    "R,S,T": 2001,
}
df_lfs = (
    pd.DataFrame.from_dict(lfs, orient="index", columns=["count"])
    .reset_index()
    .rename(columns={"index": "sic_section"})
)
df_lfs["source"] = "ONS LFS 2025 Q3"

plot_df_section = pd.concat([plot_df_section, df_lfs], ignore_index=True)

plot_df_section["sample_size"] = plot_df_section.groupby("source")["count"].transform(
    "sum"
)
plot_df_section["Frequency"] = plot_df_section.groupby("source")["count"].transform(
    lambda x: x / x.sum()
)


def add_proportion_confint(prop: float, nobs: int, alpha=0.05):
    """Calculate confidence interval for a proportion."""
    ci = binomtest(int(prop * nobs), nobs).proportion_ci(confidence_level=1 - alpha)
    return prop - ci.low, ci.high - prop


plot_df_section[["error_y_minus", "error_y_plus"]] = plot_df_section.apply(
    lambda row: pd.Series(
        add_proportion_confint(prop=row["Frequency"], nobs=row["sample_size"])
    ),
    axis=1,
)

# remove CI for the published LFS data
msk_lfs = plot_df_section["source"] == "ONS LFS 2025 Q3"
plot_df_section.loc[msk_lfs, ["error_y_minus", "error_y_plus"]] = None

# %%

fig = px.bar(
    plot_df_section,
    x="sic_section",
    y="Frequency",
    color="source",
    barmode="group",
    title="Distribution of unambiguously coded responses at SIC Section level",
    template="simple_white",
    error_y="error_y_plus",
    error_y_minus="error_y_minus",
    hover_data={"count": True, "sample_size": True},
)

fig.update_xaxes(
    title="SIC Section",
    categoryorder="category ascending",
    showgrid=True,
    gridcolor="lightgrey",
    ticks="outside",
    showline=True,
    mirror=True,
    zeroline=False,
    dtick=1,
    tickson="boundaries",
)

# make the ci lines thinner
fig.update_traces(error_y={"thickness": 1, "width": 2})
fig.update_yaxes(showgrid=True, gridcolor="lightgrey", tickformat=".0%")

# legend on top, no title
fig.update_layout(
    legend={
        "title_text": "",
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
    }
)

fig.update_layout(height=500, width=1400)
fig.show()

if out_dir:
    # fig.write_image(f"{out_dir}/cc_sa_sic_section_distribution.png")
    fig.write_html(f"{out_dir}/it11_cc_sa_cims_sic_section_distribution.html")

# %%
