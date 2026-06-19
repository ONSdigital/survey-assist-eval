"""A script to evaluate the open questions."""

# pylint: disable=C0103

# %%
import os

import pandas as pd
from dotenv import load_dotenv

from src.survey_assist_eval.evaluation.open_questions_metrics import (
    add_text_stats_columns,
    compare_text_stats,
    filter_nonempty_object_column,
)
from src.survey_assist_eval.evaluation.plotting_helpers import (
    build_filterable_dashboard,
    build_filterable_plot,
    build_histogram,
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

comparison = compare_text_stats(
    stg_dfs_followup_dict,
    prefix="followup_question_",
    word_threshold=MAX_WORD_COUNT_THRESHOLD,
    sentence_threshold=MAX_NUM_SENTENCE_THRESHOLD,
    long_sentence_threshold=MAX_WORD_COUNT_PER_SENTENCE_THRESHOLD,
    short_word_count_threshold=MIN_WORD_COUNT_THRESHOLD,
)

print(comparison)

# %%
stg3_followup_df = stg_dfs_dict["west9"]
filterable_plots = []
followup_eval_cols = [
    col for col in stg3_followup_df.columns if "followup_question_" in col
]
for eval_col in followup_eval_cols:
    x_col_title = (
        eval_col.replace("followup_question_", "").replace("_", " ").capitalize()
    )

    fig = build_filterable_plot(
        input_df=stg3_followup_df,
        group_col="sic_section",
        default_group="All",
        figure_builder=build_histogram,
        groups_order=None,
        include_default_group=True,
        x_col=eval_col,
        nbinsx=20,
        layout={
            "xtitle": x_col_title,
            "title": f"Follow-up Question{x_col_title} Distribution by SIC Section",
        },
        showlegend=True,
    )

    filterable_plots.append(fig)

simple_language_dashboard = build_filterable_dashboard(
    filterable_plots=filterable_plots,
    default_group="All",
    include_default_group=True,
)


# %%

if out_dir:
    simple_language_dashboard.write_html(
        f"{out_dir}/simple_langauge_eval_dashboard.html"
    )

# %%
