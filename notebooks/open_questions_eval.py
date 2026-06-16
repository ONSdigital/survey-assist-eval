"""A script to evaluate the open questions (Non-LLM)."""

# %%

# pylint: disable=C0301

import pandas as pd
import plotly.express as px

from src.survey_assist_eval.evaluation.open_questions_metrics import (
    count_chars_in_column,
    count_words_in_column,
)

# %%
BUCKET_NAME = "ons-survey-assist-dev-evaluation-data"

WORK_FOLDER = (
    f"gs://{BUCKET_NAME}/evaluation-pipeline/two_prompt_pipeline/"
    "2026_03_tlfs_it11_gemini25_europe_west9/"
)
STG3_FILE = f"{WORK_FOLDER}STG3.parquet"
stg3_df = pd.read_parquet(STG3_FILE)

# %%
#
stg3_df["followup_question_char_count"] = count_chars_in_column(
    stg3_df, "followup_question"
)
px.histogram(
    stg3_df[stg3_df["followup_question_char_count"] > 0]["followup_question_char_count"]
).show()

stg3_df["followup_question_word_count"] = count_words_in_column(
    stg3_df, "followup_question"
)
px.histogram(
    stg3_df[stg3_df["followup_question_word_count"] > 0]["followup_question_word_count"]
).show()
