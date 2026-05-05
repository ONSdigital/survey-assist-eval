"""Notebook to compare clerical coding vs SurveyAssist model coding performance.

Loads preprocessed data with both clerical and SA codings,
calculates various metrics and visualises them.
Expects environment variable EVALUATION_BUCKET_NAME to be set.

Disabled check for too long lines (f strings) and variables names (uppercase for constants)
"""

# ruff: noqa: PLR2004
# pylint: disable=C0301,C0103,W0104,R0801

# %%
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from notebooks.classification_eval.helper_group_plotly import create_grouped_selector
from notebooks.november_test.helper_load_data import combine_small_groups
from survey_assist_eval.data_cleaning.prep_data import get_clean_n_digit_codes
from survey_assist_eval.evaluation.metrics import (
    calc_simple_metrics,
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
# OPTIONAL: add cc for known sections, it impacts mostly F (10%) and a bit G (3%), N(2%)
msk = combined_df["cc_initial_codes"].map(len) == 0
combined_df["cc_initial_codes_add_section"] = combined_df["cc_initial_codes"]
combined_df.loc[msk, "cc_initial_codes_add_section"] = combined_df.loc[
    msk, "sic_section"
].apply(lambda x: get_clean_n_digit_codes(x, n=5)[0])

combined_df["cc_initial_codes"] = combined_df["cc_initial_codes_add_section"]

# %% create code columns at each digit level for clerical and SA initial codes
stage_cols = {
    #    "no direct lookup": ("cc_initial_codes", "sa_without_kb_initial_codes"),
    "SurveyAssist": ("cc_initial_codes", "sa_initial_codes"),
}
if "cims_initial_codes" in combined_df.columns:
    stage_cols["CIMS"] = ("cc_initial_codes", "cims_initial_codes")

for col in set().union(*stage_cols.values()):
    for DIGITS in [0, 2, 3, 4, 5]:
        combined_df[f"{col}_to_{DIGITS}digits"] = combined_df[col].apply(
            lambda x, n=DIGITS: get_clean_n_digit_codes(x, n=n)[0]
        )

# %% create groups by SIC sections
thr = sum(combined_df["sic_section"] == "-9") + 1
combined_df_sic = combine_small_groups(
    combined_df, "sic_section", group_size_threshold=thr
)

# %% calculate metrics at different digit levels for different methods
eval_metrics = {}
for stage, col_names in stage_cols.items():
    for DIGITS in [0, 2, 3, 4, 5]:
        for sic in sorted(combined_df_sic["sic_section"].unique()):
            print(f"Processing {stage} codes to {DIGITS} digits for section {sic}...")
            sub_df = combined_df_sic[combined_df_sic["sic_section"] == sic].copy()
            eval_metrics[(DIGITS, stage, "sa_cc", sic)] = calc_simple_metrics(
                sub_df,
                truth_col=f"{col_names[0]}_to_{DIGITS}digits",
                initial_model_col=f"{col_names[1]}_to_{DIGITS}digits",
                final_model_col=None,
            )
            eval_metrics[(DIGITS, stage, "cc_cc", sic)] = calc_simple_metrics(
                sub_df,
                truth_col=f"{col_names[0]}_to_{DIGITS}digits",
                initial_model_col=f"{col_names[0]}_to_{DIGITS}digits",
                final_model_col=None,
            )


# %%
plot_df = pd.DataFrame(
    [
        {
            "sic_section": k[3],
            "digits": str(k[0]) if k[0] > 0 else "S",
            "method": k[2][0:2].upper() + " " + k[1],
            "codability": v.codability_metrics.initial_codable_prop,
            "f1": v.ambiguity_metrics.f1 if k[2] == "sa_cc" else None,
            "precision": v.ambiguity_metrics.precision if k[2] == "sa_cc" else None,
            "recall": v.ambiguity_metrics.recall if k[2] == "sa_cc" else None,
            "accuracy": v.ambiguity_metrics.accuracy if k[2] == "sa_cc" else None,
            "confusion_matrix": (
                f"TP={v.ambiguity_metrics.TP}, FP={v.ambiguity_metrics.FP}, FN={v.ambiguity_metrics.FN}, TN={v.ambiguity_metrics.TN}"
                if k[2] == "sa_cc"
                else None
            ),
            "OO Accuracy": (
                v.initial_accuracy_metrics.accuracy_oo_unambiguous,
                v.initial_accuracy_metrics.matches_oo,
                v.ambiguity_metrics.TN,
            ),
            "OM Accuracy": (
                v.initial_accuracy_metrics.accuracy_om_unambiguous,
                v.initial_accuracy_metrics.matches_om,
                v.ambiguity_metrics.FP + v.ambiguity_metrics.TN,
            ),
            "MO Accuracy": (
                v.initial_accuracy_metrics.accuracy_mo_unambiguous,
                v.initial_accuracy_metrics.matches_mo,
                v.ambiguity_metrics.FN + v.ambiguity_metrics.TN,
            ),
            "MM Accuracy": (
                v.initial_accuracy_metrics.accuracy_mm_total,
                v.initial_accuracy_metrics.matches_mm,
                v.initial_accuracy_metrics.total_records,
            ),
        }
        for k, v in eval_metrics.items()
    ]
)

# drop "CC ***" (as we only need one clerical stats)
plot_df = plot_df[
    ~plot_df.method.str.startswith("CC") | plot_df.method.str.endswith("Assist")
].copy()
# fix order categories
method_dtype = pd.CategoricalDtype(
    categories=["SurveyAssist", "CIMS", "Clerical Coding"], ordered=True
)
plot_df["method"] = (
    plot_df["method"]
    .str.replace("SA SurveyAssist", "SurveyAssist", regex=False)
    .str.replace("SA CIMS", "CIMS", regex=False)
    .str.replace("CC SurveyAssist", "Clerical Coding", regex=False)
    .astype(method_dtype)
)
plot_df.method.value_counts()


# %%
# melt for easier plotting
def create_f1_plot(
    in_df: pd.DataFrame,
    default_group: str = "Total",
    ylim: tuple[float, float] = (-0.01, 1.01),
) -> go.Figure:
    """Create a line plot for codability, precision, recall, and F1 metrics.

    Args:
        in_df: DataFrame containing the metrics to plot. Expected to have columns for 'digits', 'method', 'sic_section', and the metrics to plot.
        default_group: The default group to highlight in the plot.
        ylim: Tuple specifying the y-axis limits.

    Returns:
        A Plotly Figure object.
    """
    plot_df_f1 = in_df.melt(
        id_vars=["digits", "method", "sic_section", "confusion_matrix"],
        value_vars=["codability", "precision", "recall", "f1", "accuracy"],
        var_name="metrics",
        value_name="value",
    ).sort_values(["method", "sic_section"], ascending=[True, False])

    # add wald CI for codability
    # n = combined_df.shape[0]
    # plot_df_f1["ci"] = 1.96 * (plot_df_f1["value"] * (1 - plot_df_f1["value"]) / n).pow(
    #    0.5
    # )
    # plot_df_f1.loc[~plot_df_f1["metrics"].isin(["codability", "accuracy"]), "ci"] = None

    color_discrete_map = None
    legend_placement = "right"
    sections_used = sorted(plot_df_f1["sic_section"].unique())
    if (
        default_group is not None
        and (default_group in sections_used)
        and (len(sections_used) <= 2)
    ):
        # change the colorscheme so that the default group is grey
        color_discrete_map = (
            {default_group: "lightgrey"}
            if len(sections_used) == 2
            else {default_group: "grey"}
        )
        legend_placement = "bottom"

    fig = px.line(
        plot_df_f1,
        x="digits",
        y="value",
        color="sic_section",
        color_discrete_map=color_discrete_map,
        line_dash="method",
        facet_col="metrics",
        title="Ambiguity Decision Metrics by Number of Digits and Method",
        markers=True,
        template="simple_white",
        hover_data={"confusion_matrix": True, "value": ":.2%"},
        # error_y="ci",
    )
    # drop first part of facet annotation
    for i in fig.layout.annotations:  # type: ignore
        i.text = i.text.split("=", maxsplit=1)[-1].capitalize()
    # display y axes as percentages and remove axis title
    fig.update_yaxes(
        tickformat=".0%",
        title_text="",
        showgrid=True,
        gridcolor="lightgrey",
        range=ylim,
    )

    # add text to footnote
    fig.update_layout(margin={"b": 160 if legend_placement == "bottom" else 130})
    fig.add_annotation(
        text=(
            """
    Codability: Percentage of records identified as unambiguous by either the model or clerical coders.<br>
    Precision: Among cases flagged as ambiguous by the model, the percentage that are truly ambiguous.<br>
    Recall: Among all truly ambiguous cases, the percentage correctly identified by the model.<br>
    F1: The harmonic mean of precision and recall.<br>
    Accuracy: Overall percentage of correct codability/ambiguity decisions.
    """
        ),
        align="left",
        xref="paper",
        yref="paper",
        x=-0.08 + (0.05 if legend_placement == "bottom" else 0),
        y=-0.45,
        showarrow=False,
        font={"size": 10},
    )
    # update legend placement
    if legend_placement == "bottom":
        fig.update_layout(
            legend={
                "x": 0,
                "y": -0.15,
                "xanchor": "left",
                "yanchor": "top",
                "orientation": "h",
                "title": None,
            }
        )

    return fig


ylimits = (
    plot_df[["codability", "f1", "precision", "recall", "accuracy"]].min().min() - 0.01,
    1.01,
)

fig1 = create_f1_plot(plot_df, default_group="Total", ylim=ylimits)
fig1.update_layout(height=500, width=1200)
fig1.show()

fig2 = create_grouped_selector(
    input_df=plot_df,
    group_col="sic_section",
    default_group="Total",
    figure_builder=create_f1_plot,
    include_default_group=True,
    ylim=ylimits,
)
fig2.show()

if out_dir:
    #     fig1.write_html(f"{out_dir}/cc_sa_cims_initial_codes_ambiguity_decision.html")
    fig2.write_html(f"{out_dir}/it11_cc_sa_cims_ambiguity_decision.html")


# %%
# melt for easier plotting
def create_accu_plot(
    in_df: pd.DataFrame,
    default_group: str = "Total",
    ylim: tuple[float, float] = (-0.01, 1.01),
) -> go.Figure:
    """Create a line plot for accuracy metrics (OO, OM, MO, MM).

    Args:
        in_df: DataFrame containing the metrics to plot. Expected to have columns for 'digits', 'method', 'sic_section',
            and the accuracy metrics as tuples of (accuracy, matches, total).
        default_group: The default group to highlight in the plot.
        ylim: Tuple specifying the y-axis limits.

    Returns:
        A Plotly Figure object.
    """
    metric_order = ["OO Accuracy", "OM Accuracy", "MO Accuracy", "MM Accuracy"]
    plot_df_accu = (
        in_df[~in_df["method"].str.startswith("Clerical")]
        .melt(
            id_vars=["digits", "method", "sic_section"],
            value_vars=metric_order,
            var_name="metrics",
            value_name="value_tuple",
        )
        .sort_values(["method", "sic_section"], ascending=[True, False])
    )
    # unwrap tuple into three columns
    plot_df_accu[["accu_value", "matches", "total"]] = pd.DataFrame(
        plot_df_accu["value_tuple"].tolist(), index=plot_df_accu.index
    )

    color_discrete_map = None
    legend_placement = "right"
    sections_used = sorted(plot_df_accu["sic_section"].unique())
    if (
        default_group is not None
        and (default_group in sections_used)
        and (len(sections_used) <= 2)
    ):
        # change the colorscheme so that the default group is grey
        color_discrete_map = (
            {default_group: "lightgrey"}
            if len(sections_used) == 2
            else {default_group: "grey"}
        )
        legend_placement = "bottom"

    fig = px.line(
        plot_df_accu,
        x="digits",
        y="accu_value",
        color="sic_section",
        color_discrete_map=color_discrete_map,
        line_dash="method",
        facet_col="metrics",
        category_orders={"metrics": metric_order},
        title="Classification Accuracy Metrics by Number of Digits",
        markers=True,
        template="simple_white",
        hover_data={"matches": True, "total": True, "accu_value": ":.2%"},
    )

    # drop first part of facet annotation
    for i in fig.layout.annotations:  # type: ignore
        i.text = i.text.split("=", maxsplit=1)[1]
    # display y axes as percentages and remove axis title
    fig.update_yaxes(
        tickformat=".0%",
        title_text="",
        showgrid=True,
        gridcolor="lightgrey",
        range=ylim,
    )

    # add text to footnote
    fig.update_layout(margin={"b": 160 if legend_placement == "bottom" else 125})
    fig.add_annotation(
        text=(
            """
    OO: One-to-One match on a subset where the clerical label as well as the model's label are not ambiguous.<br>
    OM: One-to-Many match on a subset where the clerical label is not ambiguous. (Is the clerical label in the model's shortlist?)<br>
    MO: Many-to-One match on a subset where the model is not ambiguous. (Is the model's label in the clerical label shortlist?)<br>
    MM: Many-to-Many match on the full set. (Is there any overlap between the clerical label's and model's shortlists?)
    """
        ),
        align="left",
        xref="paper",
        yref="paper",
        x=-0.08
        + (
            0.05 if legend_placement == "bottom" else 0
        ),  # move right if legend is at the bottom
        y=-0.42,
        showarrow=False,
        font={"size": 10},
    )
    # update legend placement
    if legend_placement == "bottom":
        fig.update_layout(
            legend={
                "x": 0,
                "y": -0.15,
                "xanchor": "left",
                "yanchor": "top",
                "orientation": "h",
                "title": None,
            }
        )

    return fig


ylimits = (plot_df[["OO Accuracy", "OM Accuracy", "MO Accuracy", "MM Accuracy"]].apply(lambda x: x.apply(lambda x: x[0])).min().min() - 0.01, 1.01)  # type: ignore

fig1 = create_accu_plot(plot_df, default_group="Total", ylim=ylimits)
fig1.update_layout(height=500, width=1000)
fig1.show()

fig2 = create_grouped_selector(
    input_df=plot_df,
    group_col="sic_section",
    default_group="Total",
    figure_builder=create_accu_plot,
    include_default_group=True,
    ylim=ylimits,
)
fig2.show()

if out_dir:
    # fig1.write_html(f"{out_dir}/cc_sa_cims_initial_codes_accuracy_metrics.html")
    fig2.write_html(f"{out_dir}/it11_cc_sa_cims_accuracy_metrics.html")

# %%
