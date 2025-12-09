# %%
"""This file is a notebook (convert with `jupytext`) for investigation of feedback
from the SurveyAssist testing.
"""
import logging

import dotenv
import pandas as pd

from survey_assist_utils.evaluation.text_analysis import TextAnalyser

# %matplotlib inline

# %%
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

bucket_prefix = dotenv.get_key(".env", "BUCKET_PREFIX")
if not bucket_prefix:
    raise ValueError("BUCKET_PREFIX not found in .env file. Please set it.")

project_id = dotenv.get_key(".env", "PROJECT_ID")
if not project_id:
    raise ValueError("PROJECT_ID not found in .env file. Please set it.")

# %%
cleaned_evaluation_df = pd.read_parquet(
    f"{bucket_prefix}evaluation_df_with_sa_clean_codes.parquet"
)


# %%
def feedback_cleaner(text: str):
    """Helper function to clean feedback text."""
    return text if text not in ("", " ", None) else "None"


feedback_ta = TextAnalyser(
    cleaned_evaluation_df,
    "feedback_comments",
    project_id,
    additional_kwargs={
        "model_name": "text-embedding-004",
        "model_task_type": "CLASSIFICATION",
        "max_batch_size": 250,
        "cleaning_func": feedback_cleaner,
        "example_null_responses": ["none", "no", "na", "no feedback", "nope"],
        "null_marker_threshold": 0.4,
    },
)

# %%
feedback_ta.investigate_clusters(kmin=1, kmax=30)

# %%
feedback_ta.drop_null_responses()
feedback_ta.investigate_clusters(kmin=1, kmax=30)

# %%
feedback_ta.apply_kmeans(k=7)
feedback_ta.visualise_dim_reduced()

# %%
