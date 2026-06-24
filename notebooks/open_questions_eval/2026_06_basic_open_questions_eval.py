"""A script to evaluate the open questions."""

# pylint: disable=C0103

# %%
import os

import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from notebooks.plotting_functions.filterable_plots import (
    build_filterable_dashboard,
    build_filterable_plot,
)
from notebooks.plotting_functions.plotting_helpers import (
    get_trace_colour_map,
)
from notebooks.plotting_functions.standard_plots import (
    build_histogram,
)
from survey_assist_eval.evaluation.open_questions_metrics import (
    add_text_stats_columns,
    compare_text_stats,
    filter_nonempty_object_column,
)

# %%
EVALUATION_FOLDER = "/evaluation-pipeline/two_prompt_pipeline"
STG_FILES_TO_COMPARE = {
    "west2": "2026_03_full_2k_gemini25_europe_west2/STG3.parquet",
    "west9": "2026_03_tlfs_it11_gemini25_europe_west9/STG3.parquet",
}

# %%
MAX_WORD_COUNT_THRESHOLD = 15
MAX_NUM_SENTENCE_THRESHOLD = 1
MAX_WORD_COUNT_PER_SENTENCE_THRESHOLD = 20
MIN_WORD_COUNT_THRESHOLD = 3

PLOT_TEXT_STAT = "word_count"

# %%
out_dir = "data/figures/open_questions_evals/"  # set as None to not save
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")

print(f"Using bucket for data loading: {bucket_name}")

base_folder = f"gs://{bucket_name}{EVALUATION_FOLDER}/"

stg_dfs_dict = {
    label: pd.read_parquet(f"{base_folder}{path}")
    for label, path in STG_FILES_TO_COMPARE.items()
}


# %%
stg_dfs_followup_dict: dict[str, pd.DataFrame] = {}
for label, df in stg_dfs_dict.items():
    stg_dfs_followup_dict[label] = add_text_stats_columns(
        df=filter_nonempty_object_column(df, column="followup_question"),
        text_column="followup_question",
    )

stg_text_stat_comparison = compare_text_stats(
    stg_dfs_followup_dict,
    prefix="followup_question_",
    word_threshold=MAX_WORD_COUNT_THRESHOLD,
    sentence_threshold=MAX_NUM_SENTENCE_THRESHOLD,
    long_sentence_threshold=MAX_WORD_COUNT_PER_SENTENCE_THRESHOLD,
    short_word_count_threshold=MIN_WORD_COUNT_THRESHOLD,
)

print(stg_text_stat_comparison)

# %%
filterable_plots = []

eval_col = f"followup_question_{PLOT_TEXT_STAT}"

for label, df in stg_dfs_followup_dict.items():
    x_col_title = (
        eval_col.replace("followup_question_", "").replace("_", " ").capitalize()
    )

    fig = build_filterable_plot(
        input_df=df,
        group_col="sic_section",
        reference_group="Total",
        figure_builder=build_histogram,
        groups_order=None,
        colour_palette=None,
        x_col=eval_col,
        nbinsx=20,
        layout={
            "xtitle": x_col_title,
            "ytitle": "Proportion",
            "title": f"{label.capitalize()} Follow-up Question {x_col_title} (Total n = {len(df)})",
        },
        showlegend=True,
        histnorm="probability",
    )

    filterable_plots.append(fig)
# %%

simple_language_dashboard = build_filterable_dashboard(
    filterable_plots=filterable_plots,
    reference_group="Total",
    group_colour_map=get_trace_colour_map(  # use colour map from filterable_plots
        go.Figure([trace for f in filterable_plots for trace in f.data])
    ),
)

simple_language_dashboard.show()
# %%

if out_dir:
    simple_language_dashboard.write_html(
        f"{out_dir}/simple_langauge_eval_dashboard.html"
    )

# %%
