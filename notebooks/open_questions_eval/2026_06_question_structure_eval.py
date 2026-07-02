"""A script to show question structure metrics."""

# pylint: disable=C0103
# pylint: disable=duplicate-code

# %%
import os

import pandas as pd
from dotenv import load_dotenv

from survey_assist_eval.data_cleaning.open_questions_eval_prep import (
    filter_nonempty_object_column,
)
from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    add_question_structure_columns,
    summarise_question_structure_columns,
)

# %%
EVALUATION_FOLDER = "/evaluation-pipeline/two_prompt_pipeline"
STG_FILE = "2026_03_tlfs_it11_gemini25_europe_west9/STG3.parquet"

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")
base_folder = f"gs://{bucket_name}{EVALUATION_FOLDER}/"
stg_df = pd.read_parquet(f"{base_folder}{STG_FILE}")
# %%
stg_df_followup = add_question_structure_columns(
    df=filter_nonempty_object_column(stg_df, column="followup_question"),
    text_column="followup_question",
)

stg_df_followup_question_quality_summary = summarise_question_structure_columns(
    df=stg_df_followup,
    prefix="followup_question_",
)

print(stg_df_followup_question_quality_summary)
# %%
