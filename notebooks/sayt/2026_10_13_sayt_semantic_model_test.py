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
OUTPUT_DIR = "data/figures/sayt/SA_898_semantic_model"

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
    "nomic_embed_plain": build_lookup_suggester(
        corpus=sayt2_corpus,
        retrievers=[
            PrefixRetrieverSpec(weight=1.0),
            NgramRetrieverSpec(weight=1.0),
            SemanticRetrieverSpec(weight=1.0, model="nomic-ai/nomic-embed-text-v1.5"),
        ],
    ),
    "nomic_embed_search_doc": build_lookup_suggester(
        corpus=sayt2_corpus,
        retrievers=[
            PrefixRetrieverSpec(weight=1.0),
            NgramRetrieverSpec(weight=1.0),
            SemanticRetrieverSpec(
                weight=1.0,
                model="nomic-ai/nomic-embed-text-v1.5",
                corpus_prefix="search_document: ",
            ),
        ],
    ),
    "nomic_embed_search_query": build_lookup_suggester(
        corpus=sayt2_corpus,
        retrievers=[
            PrefixRetrieverSpec(weight=1.0),
            NgramRetrieverSpec(weight=1.0),
            SemanticRetrieverSpec(
                weight=1.0,
                query_prefix="search_query: ",
                model="nomic-ai/nomic-embed-text-v1.5",
            ),
        ],
    ),
    "nomic_embed_search_doc_and_query": build_lookup_suggester(
        corpus=sayt2_corpus,
        retrievers=[
            PrefixRetrieverSpec(weight=1.0),
            NgramRetrieverSpec(weight=1.0),
            SemanticRetrieverSpec(
                weight=1.0,
                model="nomic-ai/nomic-embed-text-v1.5",
                corpus_prefix="search_document: ",
                query_prefix="search_query: ",
            ),
        ],
    ),
    "nomic_embed_classification": build_lookup_suggester(
        corpus=sayt2_corpus,
        retrievers=[
            PrefixRetrieverSpec(weight=1.0),
            NgramRetrieverSpec(weight=1.0),
            SemanticRetrieverSpec(
                weight=1.0,
                corpus_prefix="classification: ",
                query_prefix="classification: ",
                model="nomic-ai/nomic-embed-text-v1.5",
            ),
        ],
    ),
    "nomic_embed_clustering": build_lookup_suggester(
        corpus=sayt2_corpus,
        retrievers=[
            PrefixRetrieverSpec(weight=1.0),
            NgramRetrieverSpec(weight=1.0),
            SemanticRetrieverSpec(
                weight=1.0,
                model="nomic-ai/nomic-embed-text-v1.5",
                corpus_prefix="clustering: ",
                query_prefix="clustering: ",
            ),
        ],
    ),
    "nomic_embed_search_document_both": build_lookup_suggester(
        corpus=sayt2_corpus,
        retrievers=[
            PrefixRetrieverSpec(weight=1.0),
            NgramRetrieverSpec(weight=1.0),
            SemanticRetrieverSpec(
                weight=1.0,
                model="nomic-ai/nomic-embed-text-v1.5",
                corpus_prefix="search_document: ",
                query_prefix="search_document: ",
            ),
        ],
    ),
    "nomic_embed_search_query_both": build_lookup_suggester(
        corpus=sayt2_corpus,
        retrievers=[
            PrefixRetrieverSpec(weight=1.0),
            NgramRetrieverSpec(weight=1.0),
            SemanticRetrieverSpec(
                weight=1.0,
                model="nomic-ai/nomic-embed-text-v1.5",
                corpus_prefix="search_query: ",
                query_prefix="search_query: ",
            ),
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
    output_dir=f"{OUTPUT_DIR}",
    code_type=CODE_TYPE,
)

metrics_table.head()
metrics_table.to_csv(f"{OUTPUT_DIR}/metrics_table.csv")

# import ast

# metrics_table["precision_at_k"] = metrics_table["precision_at_k"].apply(ast.literal_eval)
# metrics_table["recall_at_k"] = metrics_table["recall_at_k"].apply(ast.literal_eval)

# # %%
# import plotly.express as px

# plot_df = metrics_table.copy()

# parts = plot_df["suggestions_col"].str.extract(
#     r"^suggestions_(\d+)chars_(.+)$"
# )
# plot_df["char_count"] = parts[0].astype(int)
# plot_df["suggester"] = parts[1]


# metric = 'unmatched_query_count'  # Or "ave_time_per_query_ms", "matched_query_count", etc.

# fig = px.line(
#     plot_df.sort_values("char_count"),
#     x="char_count",
#     y=metric,
#     color="suggester",
#     markers=True,
#     labels={"char_count": "Characters typed", metric: metric},
# )
# fig.show()

# # %%
# plot_df = plot_df.copy()

# for k in range(1,MAX_SUGGESTIONS):
#     plot_df[f"precision_at_{k}"] = plot_df["precision_at_k"].apply(
#         lambda values: values[k]
#     )
#     plot_df[f"recall_at_{k}"] = plot_df["recall_at_k"].apply(
#         lambda values: values[k]
#     )

# metrics = [
#     "ave_time_per_query_ms",
#     "unmatched_query_count",
#     "mrr",
#     "precision_at_1",
#     "recall_at_1",
#     "precision_at_3",
#     "recall_at_3",
#     "precision_at_5",
#     "recall_at_5",
#     "precision_at_8",
#     "recall_at_8",

# ]

# long_df = plot_df.melt(
#     id_vars=["char_count", "suggester"],
#     value_vars=metrics,
#     var_name="metric",
#     value_name="value",
# )

# fig = px.line(
#     long_df.sort_values("char_count"),
#     x="char_count",
#     y="value",
#     color="suggester",
#     facet_col="metric",
#     facet_col_wrap=2,
#     markers=True,
#     category_orders={"metric": metrics},
#     labels={"char_count": "Characters typed", "value": "Metric value"},
# )
# fig.update_yaxes(matches=None)  # Each metric needs its own y-axis scale.
# fig.update_xaxes(dtick=1)
# fig.for_each_annotation(lambda annotation: annotation.update(
#     text=annotation.text.split("=", 1)[-1]
# ))
# fig.show()
# %%
