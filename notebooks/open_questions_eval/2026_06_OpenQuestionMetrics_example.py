"""A script show an example of using OpenQuestionMetrics."""

# pylint: disable=C0103

# %%
import os

import pandas as pd
from dotenv import load_dotenv

from src.survey_assist_eval.evaluation.open_questions_metrics import (
    evaluate_open_question,
)

# %%
EVALUATION_FOLDER = "/evaluation-pipeline/two_prompt_pipeline"
STG_FILE = "2026_03_tlfs_it11_gemini25_europe_west9/STG3.parquet"


# %%
MAX_WORD_COUNT_THRESHOLD = 15
MAX_NUM_SENTENCE_THRESHOLD = 1
MAX_WORD_COUNT_PER_SENTENCE_THRESHOLD = 20
MIN_WORD_COUNT_THRESHOLD = 3

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")

base_folder = f"gs://{bucket_name}{EVALUATION_FOLDER}/"

stg_df = pd.read_parquet(f"{base_folder}{STG_FILE}")

# %%

metrics = evaluate_open_question(
    stg_df,
    text_column="followup_question",
    word_threshold=MAX_WORD_COUNT_THRESHOLD,
    sentence_threshold=MAX_NUM_SENTENCE_THRESHOLD,
    long_sentence_threshold=MAX_WORD_COUNT_PER_SENTENCE_THRESHOLD,
    short_word_count_threshold=MIN_WORD_COUNT_THRESHOLD,
)

metrics.report_metrics()
# %%
