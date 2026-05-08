"""Notebook to visualise the codability gain/loss using a Sankey diagram.

Loads preprocessed data with clerical, SurveyAssist, and CIMS codings,
derives codability levels, and visualises the transitions between methods.
Expects environment variable EVALUATION_BUCKET_NAME to be set.

Disabled check for too long lines (f strings) and variables names (uppercase for constants)
"""

# pylint: disable=C0301,C0103,R0914,R0801

# %%
import os
import re

import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from notebooks.classification_eval.helper_group_plotly import create_grouped_selector
from notebooks.november_test.helper_load_data import combine_small_groups
from survey_assist_eval.data_cleaning.code_standard import (
    get_codability_level,
)

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
for prefix in ["cc", "sa", "cims"]:
    combined_df[f"{prefix}_codability_level"] = combined_df.apply(
        lambda row, pre=prefix: get_codability_level(
            row[f"{pre}_initial_codes"], code_type="SIC"
        ),
        axis=1,
    )

# %%
thr = sum(combined_df["sic_section"] == "-9") + 1
combined_df_sic = combine_small_groups(
    combined_df, "sic_section", group_size_threshold=thr
)


# %%
def _flow_color_from_levels(level1: str, level2: str) -> str:
    """Determine flow color based on codability level changes."""
    if level1 == level2:
        return "rgba(166,217,106,0.3)"  # green for no change (gain)
    big_change = {
        ("Uncodable", "Sub-class (5-digits)"),
        ("Uncodable", "Class (4-digits)"),
        ("Section (letter)", "Sub-class (5-digits)"),
    }
    if (level1, level2) in big_change or (level2, level1) in big_change:
        return "rgba(253,174,97,0.3)"  # orange for big disagreement

    return "rgba(180,180,180,0.3)"  # gray for small disagreement


def _flows_from_records(
    in_df: pd.DataFrame,
    cols: tuple[str, str],
    ind: int,
    *,
    levels: list[str],
    example_col: str | None = None,
) -> pd.DataFrame:
    in_df["source"] = in_df[cols[0]].apply(levels.index) + ind * len(levels)
    in_df["target"] = in_df[cols[1]].apply(levels.index) + (ind + 1) * len(levels)
    grouped = in_df.groupby([cols[0], "source", cols[1], "target"])
    # add size and most common examples
    df = (
        grouped.size().reset_index(name="value")
        if not example_col
        else grouped.aggregate(
            value=(example_col, "size"),
            examples=(
                example_col,
                lambda x: "<br>Examples: "
                + ", ".join(x.dropna().astype(str).value_counts().index[:3]),
            ),
        ).reset_index()
    )
    if not example_col:
        df["examples"] = ""
    df["color"] = df.apply(
        lambda row: _flow_color_from_levels(row[cols[0]], row[cols[1]]), axis=1
    )
    df["customdata"] = df.apply(
        lambda row: (
            f"Count: {100 * row['value'] / len(in_df):.1f}% ({row['value']}) {row['examples']}"
            if len(in_df) > 0
            else "0.0% (0)"
        ),
        axis=1,
    )
    return df[["source", "target", "value", "color", "customdata"]]


# %%
# Create a three-stage Sankey diagram comparing codability levels across
# SurveyAssist, clerical coding, and CIMS for the full dataset and each SIC group.


def create_sankey_codability_gain_loss(
    input_df: pd.DataFrame,
    *,
    column_names: list[str] | None = None,
    column_labels: list[str] | None = None,
    example_col: str | None = None,
    levels: list[str] | None = None,
) -> go.Figure:
    """Create a Sankey diagram to visualise codability gain/loss.

    Args:
        input_df: DataFrame containing the codability levels for the plotted stages.
        column_names: List of column names in input_df representing the codability levels
            for different methods. The order of columns determines the flow direction in the Sankey diagram.
            Defaults to ["sa_codability_level", "cc_codability_level", "cims_codability_level"].
        column_labels: Optional display labels for the three stages. If not provided, column names will be used.
        example_col: Optional column name in input_df containing example descriptions to show on hover.
            If not provided, no examples will be shown.
        levels: Optional list of codability levels to include in the diagram. If not provided,
            all levels in the data will be included.

    Return:
        A Plotly Figure object representing the Sankey diagram.

    Raises:
        ValueError: If any of the requested stage columns are missing.
    """
    if column_names is None:
        column_names = [
            "sa_codability_level",
            "cc_codability_level",
            "cims_codability_level",
        ]

    selected_columns = column_names + (
        [example_col] if example_col in input_df.columns else []
    )
    if not all(col in input_df.columns for col in selected_columns):
        raise ValueError(f"Column(s) {selected_columns} not found in input DataFrame.")
    input_df = input_df[selected_columns].copy()

    if not levels:
        levels = sorted(pd.unique(input_df[column_names].values.ravel("K")))
        levels.sort(key=lambda x: (-int(re.sub(r"\D", "", "0" + x)), x))

    # add proportion to label list
    levels2 = []
    for col_name in column_names:
        for lab in levels:
            count = sum(input_df[col_name] == lab)
            prop = 100 * count / len(input_df) if len(input_df) > 0 else 0
            levels2.append(f"{lab}: {prop:.1f}% ({count})")
    label_colors = (["#1a9641"] + ["#a6d96a"] * (len(levels) - 2) + ["#fdae61"]) * 3
    node = {"color": label_colors, "label": levels2}

    # create flow/link data for sankey diagram
    link_df = pd.concat(
        [
            _flows_from_records(
                input_df,
                ind=i,
                cols=(column_names[i], column_names[i + 1]),
                levels=levels,
                example_col=example_col,
            )
            for i in range(len(column_names) - 1)
        ],
        ignore_index=True,
    )
    link: dict[str, object] = {
        "source": link_df["source"].tolist(),
        "target": link_df["target"].tolist(),
        "value": link_df["value"].tolist(),
        "color": link_df["color"].tolist(),
        "customdata": link_df["customdata"].tolist(),
    }
    link["hovertemplate"] = "%{customdata}<extra></extra>"

    sankey_fig = go.Figure(data=[go.Sankey(node=node, link=link)])
    # label the left and right sides
    if not column_labels:
        column_labels = [
            column_name.split("_", maxsplit=1)[0].upper()
            for column_name in column_names
        ]
    for i, col_label in enumerate(column_labels):
        sankey_fig.add_annotation(
            x=-0.02 + i * (1.03 / (len(column_names) - 1)),
            y=1.05,
            text=col_label,
            showarrow=False,
            font={"size": 12},
        )

    sankey_fig.update_layout(
        title_text="Comparison of Codability Levels Across Methods",
        font_size=10,
        height=600,
        width=1000,
    )
    return sankey_fig


# %%
level_names = sorted(
    combined_df_sic["sa_codability_level"].unique().tolist(),
    key=lambda x: (-int(re.sub(r"\D", "", "0" + x)), x),
)

fig = create_grouped_selector(
    combined_df_sic,
    group_col="sic_section",
    default_group="Total",
    figure_builder=create_sankey_codability_gain_loss,
    column_names=[
        "sa_codability_level",
        "cc_codability_level",
        "cims_codability_level",
    ],
    column_labels=["SurveyAssist", "Clerical Codability", "CIMS"],
    example_col="merged_industry_desc",
    levels=level_names,
)
fig.show()

if out_dir:
    fig.write_html(f"{out_dir}it11_cc_sa_cims_sankey_codability.html")

# %%
