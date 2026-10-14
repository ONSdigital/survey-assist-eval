"""A script to test the SAYT semantic model with different retrievers."""

# pylint: disable=C0103
# %%
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SemanticRetrieverSpec,
)
from survey_assist_utils.logging import get_logger

from notebooks.sayt.sayt_utils import build_lookup_suggester, build_sayt_corpus_from_df
from notebooks.sayt.suggester_eval import run_eval_for_suggesters

# %%
CODE_TYPE = "sic"
MAX_SUGGESTIONS = 9
CORRECT_CODE_COL = "correct_sic_code"
NUM_CHARACTERS_LIST = list(range(4, 10))
OUTPUT_DIR = "data/figures/sayt/SA_898_semantic_model/search_doc_only_sa_embed_core"

# %%
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")


logger = get_logger(__name__)

# %%
test_df = pd.read_parquet(
    f"gs://{bucket_name}/evaluation-pipeline/original_datasets/sic_2k/sic_2k_test_data.parquet"
)

is_self_employed = test_df["sic2007_employee"] == "-9"

test_df["full_entry"] = np.where(
    is_self_employed,
    test_df["sic2007_self_employed"],
    test_df["sic2007_employee"],
)
test_df["employment_status"] = np.where(is_self_employed, "self_employed", "employed")

test_df = test_df.rename(columns={"clerical_codes": CORRECT_CODE_COL})

# %%
sic_kb_for_classifai = pd.read_csv(
    f"gs://{bucket_name}/sic_knowledgebase/sic_kb_for_sayt.csv", dtype=str
)

_, sayt2_corpus = build_sayt_corpus_from_df(
    df=sic_kb_for_classifai,
    search_text_col="search_text",
    display_text_col="display_text",
    code_col="code",
    code_type=CODE_TYPE,
    incl_code_in_display=True,
)

# %%

suggesters = {
    "all_MiniLM_L6_v2": build_lookup_suggester(
        corpus=sayt2_corpus,
        retrievers=[
            PrefixRetrieverSpec(weight=1.0),
            NgramRetrieverSpec(weight=1.0),
            SemanticRetrieverSpec(weight=1.0, model="all-MiniLM-L6-v2"),
        ],
    ),
    "nomic_embed_text_v1_5": build_lookup_suggester(
        corpus=sayt2_corpus,
        retrievers=[
            PrefixRetrieverSpec(weight=1.0),
            NgramRetrieverSpec(weight=1.0),
            SemanticRetrieverSpec(weight=1.0, model="nomic-ai/nomic-embed-text-v1.5"),
        ],
    ),
}

# %%
suggestions_df, fig, metrics_table = run_eval_for_suggesters(
    df=test_df,
    suggesters_dict=suggesters,
    num_chars=NUM_CHARACTERS_LIST,
    suggestions_limit=MAX_SUGGESTIONS,
    correct_codes_col=CORRECT_CODE_COL,
    output_dir=f"{OUTPUT_DIR}_all",
    code_type=CODE_TYPE,
)

metrics_table.head()
metrics_table.to_csv(f"{OUTPUT_DIR}_all/metrics_table.csv")
# %%
