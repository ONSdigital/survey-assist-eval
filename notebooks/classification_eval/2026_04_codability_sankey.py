"""Notebook to visualise the codability gain/loss using a Sankey diagram.

Loads preprocessed data with both clerical and SA codings,
calculates various metrics and visualises them.
Expects environment variable BUCKET_PREFIX to be set.

Disabled check for too long lines (f strings) and variables names (uppercase for constants)
"""

# ruff: noqa: PLR2004
# pylint: disable=C0301,C0103,W0104,R0914,R0801

# %%
import os
import re

import dotenv
import pandas as pd
import plotly.graph_objects as go

from notebooks.classification_eval.helper_group_plotly import create_grouped_selector
from notebooks.november_test.helper_load_data import combine_small_groups
from survey_assist_eval.data_cleaning.sic_codes import (
    get_codability_level,
)

# %%
bucket_prefix = dotenv.get_key(".env", "EVALUATION_BUCKET")
out_dir = "data/figures/tlfs_it11/"  # needs local folder unfortunately, set to None to skip saving
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

work_folder = f"{bucket_prefix}evaluation-pipeline/two_prompt_pipeline/2026_03_tlfs_it11_gemini25_europe_west9/"

# %% read data prepared in 2026_02_prep_tlfs_iteration_data.py script
combined_df = pd.read_parquet(f"{work_folder}sa_cc_cims_combined.parquet")
# ensure code columns are sets (they were saved as arrays in parquet)
for col in [col for col in combined_df.columns if "codes" in col.lower()]:
    print(f"Converting {col} to sets...")
    combined_df[col] = combined_df[col].apply(set)

# %%
for prefix in ["cc", "sa", "kb", "cims"]:
    combined_df[f"{prefix}_codability_level"] = combined_df.apply(
        lambda row, pre=prefix: get_codability_level(row[f"{pre}_initial_codes"]),
        axis=1,
    )

# %%
thr = sum(combined_df["sic_section"] == "-9") + 1
combined_df_sic = combine_small_groups(
    combined_df, "sic_section", group_size_threshold=thr
)

# %%
# create sankey diagram showing the flow of codability levels from clerical to SA coding
# and from SA to KB and CIMS coding, with counts of records in each flow
# we will create a sankey diagram with 4 stages: Clerical, SA, KB, CIMS
# and flows between them based on codability levels
# we will also create a separate sankey diagram for each SIC section
# and a combined one for all data


def create_sankey_codability_gain_loss(
    input_df: pd.DataFrame,
    left_col: str = "kb_codability_level",
    middle_col: str = "cc_codability_level",
    right_col: str = "cims_codability_level",
    labels: list[str] | None = None,
) -> go.Figure:
    """Create a Sankey diagram to visualise codability gain/loss.

    Args:
        input_df: DataFrame containing initial and final codability levels.
        left_col: Name of the column representing initial codability levels.
        middle_col: Name of the column representing middle codability levels.
        right_col: Name of the column representing final codability levels.
        title_suffix: Suffix to add to the title of the figure, e.g. section name.
        labels: Optional list of labels for the left, middle, and right columns. If None, column names will be used.

    Return:
        A Plotly Figure object representing the Sankey diagram, or None if columns not found.
    """
    if not all(col in input_df.columns for col in [left_col, middle_col, right_col]):
        raise ValueError(
            f"Columns {left_col} or {right_col} not found in input DataFrame."
        )
    input_df = input_df[[left_col, middle_col, right_col]].copy()

    label_list = list(
        pd.unique(input_df[[left_col, middle_col, right_col]].values.ravel("K"))
    )
    label_list.sort(key=lambda x: (-int(re.sub(r"\D", "", "0" + x)), x))

    input_df[left_col + "_num"] = input_df[left_col].apply(label_list.index)
    input_df[middle_col + "_num"] = input_df[middle_col].apply(label_list.index) + len(
        label_list
    )

    input_df[right_col + "_num"] = input_df[right_col].apply(
        label_list.index
    ) + 2 * len(label_list)

    input_df = input_df.sort_values(
        by=[left_col + "_num", middle_col + "_num", right_col + "_num"]
    ).reset_index(drop=True)

    sankey_df = [
        input_df.groupby([left_col, left_col + "_num", middle_col, middle_col + "_num"])
        .size()
        .reset_index(),
        input_df.groupby(
            [middle_col, middle_col + "_num", right_col, right_col + "_num"]
        )
        .size()
        .reset_index(),
    ]
    sankey_df[0]["gain"] = sankey_df[0][left_col] == sankey_df[0][middle_col]
    sankey_df[1]["gain"] = sankey_df[1][middle_col] == sankey_df[1][right_col]

    # add proportion to label list
    label_list2 = []
    for col_name in [left_col, middle_col, right_col]:
        for lab in label_list:
            count = sum(input_df[col_name] == lab)
            prop = 100 * count / len(input_df) if len(input_df) > 0 else 0
            label_list2.append(f"{lab}: {prop:.1f}% ({count})")
    label_colors = (["#1a9641"] + ["#a6d96a"] * (len(label_list) - 2) + ["#fdae61"]) * 3

    # create flow/link data for sankey diagram
    link: dict[str, list] = {
        "source": [],
        "target": [],
        "value": [],
        "color": [],
    }
    for ind, (col1, col2) in enumerate(
        [
            (left_col + "_num", middle_col + "_num"),
            (middle_col + "_num", right_col + "_num"),
        ]
    ):
        link["source"].extend(sankey_df[ind][col1])
        link["target"].extend(sankey_df[ind][col2])
        link["value"].extend(sankey_df[ind][0].tolist())
        link["color"].extend(
            sankey_df[ind]["gain"]
            .apply(
                lambda x: (
                    "rgba(166,217,106,0.3)"
                    if x > 0
                    else ("rgba(180,180,180,0.3)" if x == 0 else "rgba(253,174,97,0.3)")
                )
            )
            .tolist()
        )
    link["customdata"] = [
        (
            f"{100 * count / len(input_df):.1f}% ({count})"
            if len(input_df) > 0
            else "0.0% (0)"
        )
        for count in link["value"]
    ]
    link["hovertemplate"] = "%{customdata}<extra></extra>"  # type: ignore

    sankey_fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 15,
                    "thickness": 20,
                    "line": {"color": "black", "width": 0.5},
                    "color": label_colors,
                    "label": label_list2,
                    "hovertemplate": "Count %{value}<extra></extra>",
                },
                link=link,
            )
        ]
    )
    # label the left and right sides
    if not labels:
        labels = [left_col, middle_col, right_col]
    sankey_fig.add_annotation(
        x=-0.02, y=1.05, text=labels[0], showarrow=False, font={"size": 12}
    )
    sankey_fig.add_annotation(
        x=1.01, y=1.05, text=labels[2], showarrow=False, font={"size": 12}
    )
    sankey_fig.add_annotation(
        x=0.5, y=1.05, text=labels[1], showarrow=False, font={"size": 12}
    )

    sankey_fig.update_layout(
        title_text=f"Comparison of Codability Levels {labels[3] if len(labels) > 3 else ''}",
        font_size=10,
        height=600,
        width=1000,
    )
    return sankey_fig


# %%
fig, group_lab = create_grouped_selector(
    combined_df_sic,
    group_col="sic_section",
    default_group="Total",
    figure_builder=create_sankey_codability_gain_loss,
    labels=["SurveyAssist", "Clerical Codability", "CIMS"],
)
fig.show()

if out_dir:
    fig.write_html(f"{out_dir}sankey_codability_comparison.html")
# %%
