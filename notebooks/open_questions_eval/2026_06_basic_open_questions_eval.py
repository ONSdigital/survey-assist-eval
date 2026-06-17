"""A script to evaluate the open questions."""

# pylint: disable=C0103

# %%
import os

import pandas as pd
from dotenv import load_dotenv

from src.survey_assist_eval.evaluation.open_questions_metrics import (
    count_chars_in_column,
    count_words_in_column,
    filter_nonempty_object_column,
)
from src.survey_assist_eval.evaluation.plotting_helpers import (
    build_filterable_dashboard,
    build_filterable_plot,
    build_histogram,
)

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")

print(f"Using bucket for data loading: {bucket_name}")

stg3_folder = (
    f"gs://{bucket_name}/evaluation-pipeline/two_prompt_pipeline/"
    "2026_03_tlfs_it11_gemini25_europe_west9/"
)
stg3_file = f"{stg3_folder}STG3.parquet"
stg3_df = pd.read_parquet(stg3_file)

# %%
stg3_followup_df = filter_nonempty_object_column(df=stg3_df, column="followup_question")
stg3_followup_df["followup_question_char_count"] = count_chars_in_column(
    stg3_followup_df, "followup_question"
)

stg3_followup_df["followup_question_word_count"] = count_words_in_column(
    stg3_followup_df, "followup_question"
)
# %%
input_df = stg3_followup_df.copy()
group_col = "sic_section"
groups_order = None
default_group = "All"
figure_builder = build_histogram
include_default_group = True
kwargs = {
    "x_col": "followup_question_word_count",
    "nbins": 20,
    "xtitle": "Follow-up Question Word Count",
}

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
        nbins=20,
        xtitle=x_col_title,
        title=f"Follow-up Question{x_col_title} Distribution by SIC Section",
        showlegend=True,
    )

    filterable_plots.append(fig)

build_filterable_dashboard(
    filterable_plots=filterable_plots,
    default_group="All",
)


# %%
