"""A script to evaluate the open questions."""

# pylint: disable=C0103

# %%
import os

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

from src.survey_assist_eval.evaluation.open_questions_metrics import (
    count_chars_in_column,
    count_words_in_column,
    filter_nonempty_object_column,
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
#
stg3_followup_df = filter_nonempty_object_column(df=stg3_df, column="followup_question")
stg3_followup_df["followup_question_char_count"] = count_chars_in_column(
    stg3_followup_df, "followup_question"
)
px.histogram(stg3_followup_df["followup_question_char_count"]).show()

stg3_followup_df["followup_question_word_count"] = count_words_in_column(
    stg3_followup_df, "followup_question"
)
