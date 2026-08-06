# %%
"""Run small tests for industry/organisation descriptions SAYT.

Expects following environment variables to be set:
- EVALUATION_BUCKET_NAME: name of GCS bucket where the data is stored
The variables are loaded from the ".env" file.
"""

# pylint: disable=invalid-name

# %%
import logging
import os

import pandas as pd
from dotenv import load_dotenv
from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SAYTSuggester,
    SemanticRetrieverSpec,
)
from survey_assist_utils.logging import get_logger

from notebooks.sayt.sayt_utils import (
    build_lookup_suggester,
    create_figure,
    get_suggestions_for_collection,
    melt_results_for_analysis,
    validate_one_code,
)

# %%
EXTENDED_RUN = False  # set to True to include more suggesters and debug messages
SIC_CODE_LENGTH = 5
MAX_SUGGESTIONS = 9  # for the evaluation we will look at ranks up to 9 only

if EXTENDED_RUN:
    logging.getLogger("survey_assist_e...").setLevel(logging.DEBUG)

load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")

OUTPUT_DIR = "data/figures/sayt"
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
# check the codes are well formed
print(
    f'Clerical codes validated: {
        test_df["correct_sic_code"]
        .apply(validate_one_code,
               code_length=SIC_CODE_LENGTH)
               .all()
               }'
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
sic_kb_for_classifai = pd.read_csv(
    f"gs://{bucket_name}/sic_knowledgebase/sic_kb_for_sayt.csv", dtype=str
)
sic_kb_for_classifai["display_text_with_code"] = (
    sic_kb_for_classifai["display_text"] + ": " + sic_kb_for_classifai["code"]
)


sayt2_corpus = list(
    zip(
        sic_kb_for_classifai["search_text"],
        sic_kb_for_classifai["display_text_with_code"],
        strict=False,
    )
)

# %%
# define bunch of different suggesters to evaluate
suggesters = {
    "Blaise proxy method (prefix + n_grams)": build_lookup_suggester(
        sayt_corpus, semantic_weight=None
    ),
    # "Hybrid method including semantic retriever": build_lookup_suggester(
    #     sayt_corpus, semantic_weight=1.0
    # ),
    # "Hybrid method with extended knowledge base": build_lookup_suggester(
    #     sayt2_corpus, semantic_weight=1.0
    # ),
}

if EXTENDED_RUN:
    suggesters.update(
        {
            "Ngrams only": SAYTSuggester(
                sayt_corpus, retrievers=[NgramRetrieverSpec()]
            ),
            "Prefix only": SAYTSuggester(
                sayt_corpus, retrievers=[PrefixRetrieverSpec()]
            ),
            "Semantic only": SAYTSuggester(
                sayt_corpus, retrievers=[SemanticRetrieverSpec()]
            ),
            "Hybrid sem_w=0.5": build_lookup_suggester(
                sayt_corpus, semantic_weight=0.5
            ),
            "Hybrid sem_w=1.5": build_lookup_suggester(
                sayt_corpus, semantic_weight=1.5
            ),
            "Blaise proxy method (prefix + n_grams) "
            "with extended knowledge base": build_lookup_suggester(
                sayt2_corpus, semantic_weight=None
            ),
        }
    )

# %%
suggestions_df = get_suggestions_for_collection(test_df, suggesters_dict=suggesters)[0]

# %%
melt_df = melt_results_for_analysis(df=suggestions_df)

# %%
create_figure(melt_df, output_dir=OUTPUT_DIR)

# %%
# log the dimensions of the retriever index matrices for each retriever
for suggester_name, suggester_obj in suggesters.items():
    configured_retrievers = getattr(suggester_obj, "_retrievers", [])
    for configured_retriever in configured_retrievers:
        name = configured_retriever.name
        retriever = configured_retriever.retriever
        index_obj = getattr(retriever, "_index", None)
        vector_store = (
            getattr(index_obj, "_vector_store", None) if index_obj is not None else None
        )
        vectors = (
            getattr(vector_store, "vectors", None) if vector_store is not None else None
        )
        if vectors is None or "embeddings" not in vectors:
            continue

        shape = vectors["embeddings"].to_numpy().shape
        logger.info(
            "Retriever index shape",
            sayt_suggester_name=suggester_name,
            retriever_name=name,
            matrix_shape=shape,
        )

# %%
