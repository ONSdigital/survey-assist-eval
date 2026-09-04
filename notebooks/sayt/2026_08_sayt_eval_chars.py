# %%
"""Run small tests for industry descriptions SAYT.

Expects following environment variables to be set:
- EVALUATION_BUCKET_NAME: name of GCS bucket where the data is stored
The variables are loaded from the ".env" file.
"""

# pylint: disable=invalid-name, duplicate-code

# %%
import os

import pandas as pd
from dotenv import load_dotenv
from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SemanticRetrieverSpec,
)
from survey_assist_utils.logging import get_logger

from notebooks.sayt.sayt_utils import (
    build_lookup_suggester,
)
from notebooks.sayt.suggester_eval import run_eval_for_suggesters

# %%
SIC_CODE_LENGTH = 5
MAX_SUGGESTIONS = 9
CORRECT_CODE_COL = "correct_sic_code"
NUM_CHARACTERS_LIST = range(4, 10)

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")

OUTPUT_DIR = "data/figures/sayt/min_characters"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

logger = get_logger(__name__)
logger.info("Location specs", bucket_name=bucket_name, output_dir=OUTPUT_DIR)

# %%
test_df = pd.read_excel(
    f"gs://{bucket_name}/evaluation-pipeline/SAYT/SAYT matching.xlsx",
    dtype=str,
    nrows=100,  # Excel formatting causes 10s of thousands of blank input rows after the real 100
    header=1,  # first row is header
)
rename_columns = {
    "Correct SIC code": "correct_sic_code",
    "Full entry looking for": "full_entry",
    "Position of correct SIC ": "rank_5chars_Blaise (as reported from SAYT team)",
    "Position of correct SIC .1": "_rank_5chars_sa_shared",
}

test_df = test_df.rename(columns=rename_columns)
test_df = test_df[rename_columns.values()]

# clean the rank values reported by the SAYT team
for col in [
    "rank_5chars_Blaise (as reported from SAYT team)",
    "_rank_5chars_sa_shared",
]:
    test_df[col] = pd.to_numeric(
        test_df[col].replace({"5 or 12": "5"}), errors="coerce"
    )

# %%
LOOKUP_FILE_NAME = f"gs://{bucket_name}/evaluation-pipeline/SAYT/Lookup_IT3_Final.csv"
sayt_df = pd.read_csv(LOOKUP_FILE_NAME, dtype=str)
sayt_df["code"] = sayt_df["SIC07"].apply(
    lambda x: x if len(x) == SIC_CODE_LENGTH else f"0{x}"
)
sayt_df["display_text_with_code"] = sayt_df["SIC_lookup"] + ": " + sayt_df["code"]

sayt_corpus = list(
    zip(sayt_df["SIC_lookup"], sayt_df["display_text_with_code"], strict=False)
)

# %%
# Commented out due to workstation limitations.
# sic_kb_for_classifai = pd.read_csv(
#     f"gs://{bucket_name}/sic_knowledgebase/sic_kb_for_sayt.csv", dtype=str
# )
# sic_kb_for_classifai["display_text_with_code"] = (
#     sic_kb_for_classifai["display_text"] + ": " + sic_kb_for_classifai["code"]
# )


# sayt2_corpus = list(
#     zip(
#         sic_kb_for_classifai["search_text"],
#         sic_kb_for_classifai["display_text_with_code"],
#         strict=False,
#     )
# )

# %%
# create suggesters using only one of the retrievers:
# PrefixRetrieverSpec, NgramRetrieverSpec, or SemanticRetrieverSpec.

suggesters_one = {
    "Blaise proxy method (prefix only)": build_lookup_suggester(
        sayt_corpus, retrievers=[PrefixRetrieverSpec()]
    ),
    "Blaise proxy method (ngram only)": build_lookup_suggester(
        sayt_corpus, retrievers=[NgramRetrieverSpec()]
    ),
    "Semantic retriever only": build_lookup_suggester(
        sayt_corpus, retrievers=[SemanticRetrieverSpec()]
    ),
}

# %%
# Test for interactions between suggesters

suggesters_two = {
    "Blaise proxy method (prefix and ngram)": build_lookup_suggester(
        sayt_corpus, retrievers=[PrefixRetrieverSpec(), NgramRetrieverSpec()]
    ),
    "Hybrid approach (prefix and semantic)": build_lookup_suggester(
        sayt_corpus, retrievers=[PrefixRetrieverSpec(), SemanticRetrieverSpec()]
    ),
    "Hybrid approach (ngram and semantic)": build_lookup_suggester(
        sayt_corpus, retrievers=[NgramRetrieverSpec(), SemanticRetrieverSpec()]
    ),
}

# %%
suggesters_three = {
    "Hybrid approach (ngram, prefix and semantic)": build_lookup_suggester(
        sayt_corpus,
        retrievers=[
            NgramRetrieverSpec(),
            PrefixRetrieverSpec(),
            SemanticRetrieverSpec(),
        ],
    ),
}

# # %%
# # create suggesters using only one of the retrievers:
# # PrefixRetrieverSpec, NgramRetrieverSpec, or SemanticRetrieverSpec.

# suggesters_one = {
#     "Blaise proxy method (prefix only)": build_lookup_suggester(
#         sayt2_corpus, retrievers=[PrefixRetrieverSpec()]
#     ),
#     "Blaise proxy method (ngram only)": build_lookup_suggester(
#         sayt2_corpus, retrievers=[NgramRetrieverSpec()]
#     ),
#     "Semantic retriever only": build_lookup_suggester(
#         sayt2_corpus, retrievers=[SemanticRetrieverSpec()]
#     ),
# }

# # %%
# # Test for interactions between suggesters

# suggesters_two = {
#     "Blaise proxy method (prefix and ngram)": build_lookup_suggester(
#         sayt2_corpus, retrievers=[PrefixRetrieverSpec(), NgramRetrieverSpec()]
#     ),
#     "Hybrid approach (prefix and semantic)": build_lookup_suggester(
#         sayt2_corpus, retrievers=[PrefixRetrieverSpec(), SemanticRetrieverSpec()]
#     ),
#     "Hybrid approach (ngram and semantic)": build_lookup_suggester(
#         sayt2_corpus, retrievers=[NgramRetrieverSpec(), SemanticRetrieverSpec()]
#     ),
# }

# # %%
# suggesters_three = {
#     "Hybrid approach (ngram, prefix and semantic)": build_lookup_suggester(
#         sayt2_corpus,
#         retrievers=[
#             NgramRetrieverSpec(),
#             PrefixRetrieverSpec(),
#             SemanticRetrieverSpec(),
#         ],
#     ),
# }

# %%
suggestions_df_one, fig_one, metrics_table_one = run_eval_for_suggesters(
    df=test_df,
    suggesters_dict=suggesters_one,
    num_chars=NUM_CHARACTERS_LIST,
    suggestions_limit=MAX_SUGGESTIONS,
    correct_codes_col=CORRECT_CODE_COL,
    output_dir=f"{OUTPUT_DIR}_one",
)

# %%
suggestions_df_two, fig_two, metrics_table_two = run_eval_for_suggesters(
    df=test_df,
    suggesters_dict=suggesters_two,
    num_chars=NUM_CHARACTERS_LIST,
    suggestions_limit=MAX_SUGGESTIONS,
    correct_codes_col=CORRECT_CODE_COL,
    output_dir=f"{OUTPUT_DIR}_two",
)

# %%
suggestions_df_three, fig_three, metrics_table_three = run_eval_for_suggesters(
    df=test_df,
    suggesters_dict=suggesters_three,
    num_chars=NUM_CHARACTERS_LIST,
    suggestions_limit=MAX_SUGGESTIONS,
    correct_codes_col=CORRECT_CODE_COL,
    output_dir=f"{OUTPUT_DIR}_three",
)

# %%
suggesters_all = {**suggesters_one, **suggesters_two, **suggesters_three}

# %%
suggestions_df_all, fig_all, metrics_table_all = run_eval_for_suggesters(
    df=test_df,
    suggesters_dict=suggesters_all,
    num_chars=NUM_CHARACTERS_LIST,
    suggestions_limit=MAX_SUGGESTIONS,
    correct_codes_col=CORRECT_CODE_COL,
    output_dir=f"{OUTPUT_DIR}_all",
)

# %%
metrics_table_one.head()

# %%
metrics_table_all.head()
