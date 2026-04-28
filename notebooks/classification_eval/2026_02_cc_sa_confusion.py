"""Notebook to compare clerical coding vs SurveyAssist model coding performance.

Loads preprocessed clerical and SurveyAssist codings,
builds confusion matrices at multiple code depths, and extracts representative
disagreement examples.
Expects environment variable EVALUATION_BUCKET_NAME to be set.

Disabled check for too long lines (f strings) and variables names (uppercase for constants)
"""

# pylint: disable=C0301,C0103,W0104,R0801

# %%
import os

import dotenv
import pandas as pd
import plotly.express as px

from survey_assist_eval.data_cleaning.prep_data import get_clean_n_digit_codes

# %%
bucket_name = dotenv.get_key(".env", "EVALUATION_BUCKET_NAME")
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
for prefix in ["cc", "sa", "sa_without_kb", "cims"]:
    for DIGITS in [0, 2, 5]:
        combined_df[f"{prefix}_initial_codes_to_{DIGITS}digits"] = (
            combined_df[f"{prefix}_initial_codes"]
            .apply(
                get_clean_n_digit_codes,
                n=DIGITS,
            )
            .map(lambda x: x[0])
        )


# %%
# Create confusion matrices for 5-digit, 2-digit, and section-level codings.
df = combined_df.copy()  # [combined_df.batch_num.notna()].copy()
for DIGITS in [5, 2, 0]:
    col1 = f"cc_initial_codes_to_{DIGITS}digits"
    col2 = f"sa_initial_codes_to_{DIGITS}digits"
    subset = {}
    subset["Unambiguously coded cases only"] = (df[col1].map(len) == 1) & (
        df[col2].map(len) == 1
    )
    # for semi-unambiguous, keep only cases where there is small set on either side
    n = 3
    subset["Subset of ambiguous cases with only two candidates"] = (
        (df[col1].map(len) < n)
        & (df[col2].map(len) < n)
        & ~subset["Unambiguously coded cases only"]
    )

    for lab, msk in subset.items():
        df2 = df[msk].copy().explode(col1).explode(col2)
        if DIGITS > 1:
            # find the most frequent off diagonal entries in plot_df
            df3 = (
                df2[df2[col1] != df2[col2]]
                .groupby([col1, col2])
                .size()
                .sort_values(ascending=False)
            )
            if df3.empty:
                print(
                    f"Skipping confusion matrix for {DIGITS}-digit, {lab} due to no off-diagonal data."
                )
                continue
            cutoff = df3.iloc[min(10, len(df3) - 1)]
            df3 = df3[df3 > cutoff].reset_index()
            labels = sorted(set(df3[col1]).union(df3[col2]))
            plot_df = (
                df2[df2[col1].isin(labels) & df2[col2].isin(labels)]
                .groupby([col1, col2])
                .size()
                .unstack(fill_value="")
            )
        else:
            labels = sorted(df[col1].explode().dropna().unique())
            plot_df = df2.groupby([col1, col2]).size().unstack(fill_value="")

        if plot_df.shape[0] == 0 or plot_df.shape[1] == 0:
            print(
                f"Skipping confusion matrix for {DIGITS}-digit, {lab} due to no data."
            )
            continue

        # Compute min/max for color scale excluding diagonal
        non_diag_values = [
            plot_df.loc[i, c]
            for i in plot_df.index
            for c in plot_df.columns
            if plot_df.loc[i, c] not in ("", None, 0) and i != c
        ]
        color_min = min(non_diag_values)
        color_max = max(non_diag_values)

        fig1 = px.imshow(
            plot_df,
            text_auto=True,
            aspect="equal",
            color_continuous_scale="Blues",
            title=f"Confusion matrix for SIC section, Clerical vs SurveyAssist<br><b>{lab}</b>",
            template="simple_white",
            zmin=color_min,
            zmax=color_max,
        )
        # reorder x axis values
        fig1.update_xaxes(
            title="Model Initial Code",
            categoryorder="array",
            categoryarray=labels,
            showgrid=True,
            gridcolor="lightgrey",
            ticks="outside",
            showline=True,
            mirror=True,
            zeroline=False,
            dtick=1,
            tickson="boundaries",  # show grid between ticks
        )
        fig1.update_yaxes(
            title="Clerical Initial Code",
            categoryorder="array",
            categoryarray=labels,
            showgrid=True,
            gridcolor="lightgrey",
            ticks="outside",
            showline=True,
            mirror=True,
            zeroline=False,
            dtick=1,
            tickson="boundaries",
        )

        fig1.update_layout(height=700, width=770)
        fig1.show()

        if out_dir:
            fig1.write_image(
                f"{out_dir}/cc_sa_initial_codes_{lab.lower().replace('-', '_')}_confusion_matrix_{DIGITS}digits.png"
            )
            fig1.write_html(
                f"{out_dir}/cc_sa_initial_codes_{lab.lower().replace('-', '_')}_confusion_matrix_{DIGITS}digits.html"
            )


# %%
# Inspect common non-overlapping clerical vs SurveyAssist disagreements in section C.
digits = 5
tmp_df = combined_df[combined_df.sic_section == "C"].copy()
min_mistakes = 5

col_cc = f"cc_initial_codes_to_{digits}digits"
col_sa = f"sa_initial_codes_to_{digits}digits"
mask_diff = tmp_df[col_sa] != tmp_df[col_cc]
mask_excl = tmp_df.apply(
    lambda row: len(row[col_cc].intersection(row[col_sa])) == 0, axis=1
)
tmp_df = tmp_df[mask_excl].copy()
tmp_df["cc_codes_str"] = tmp_df[col_cc].apply(lambda x: ", ".join(sorted(x)))
tmp_df["sa_codes_str"] = tmp_df[col_sa].apply(lambda x: ", ".join(sorted(x)))
tmp_df["sa_section"] = tmp_df["sa_initial_codes_to_0digits"].apply(
    lambda x: ", ".join(sorted(x))
)
frequent_mistakes = (
    tmp_df.groupby(["cc_codes_str", "sa_codes_str", "sa_section"])
    .size()
    .sort_values(ascending=False)
    .reset_index(name="count")
)
print(frequent_mistakes[frequent_mistakes["count"] > min_mistakes])

examples = pd.DataFrame()
columns = [
    "unique_id",
    "soc2020_job_title",
    "soc2020_job_description",
    "merged_industry_desc",
    "cc_codes_str",
    "sa_codes_str",
    "sa_section",
]
for _, row in frequent_mistakes[frequent_mistakes["count"] > min_mistakes].iterrows():
    msk = (tmp_df["sa_codes_str"] == row.sa_codes_str) & (
        tmp_df["cc_codes_str"] == row.cc_codes_str
    )
    examples = pd.concat([examples, tmp_df.loc[msk, columns]])

# set pandas to print all columns
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
print(examples)


# %%
tmp_df[(tmp_df.cc_codes_str == "43999") & (tmp_df.sa_codes_str == "41202")][
    [*columns, "sic_ind_occ1", "sic_ind1"]
]

# %%
