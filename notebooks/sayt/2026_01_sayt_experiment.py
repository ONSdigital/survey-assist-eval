"""Experimenting with different approaches to improve SAYT results.

This is initial experiment not using the SAYTSuggester class.
"""

# ruff: noqa: PLR2004
# pylint: disable=C0301,C0103,C0115,C0116,W0621,W0105,R0903,R0801
# %%
import os

import pandas as pd
from classifai.indexers import VectorStore
from classifai.indexers.dataclasses import VectorStoreSearchInput
from classifai.vectorisers import GcpVectoriser, HuggingFaceVectoriser, VectoriserBase
from dotenv import find_dotenv, get_key
from sklearn.feature_extraction.text import CountVectorizer

env_file = find_dotenv(".env")
if not env_file:
    raise FileNotFoundError("No .env file found in the directory tree.")

print(f"Environment variables will be read from {env_file}")

bucket_name = get_key(env_file, "EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")

print(f"Using bucket for data loading: {bucket_name}")

project_id = get_key(env_file, "PROJECT_ID")
if not project_id:
    raise ValueError("PROJECT_ID not found in .env file. Please set it.")

print(f"Using project ID for GCP vectoriser: {project_id}")

out_dir = "data/SAYT_semantic_search_results"

if out_dir.startswith("gs://"):
    raise ValueError(
        "Output directory at this moment needs to be local to enable direct classifai loading."
    )

os.makedirs(out_dir, exist_ok=True)

# %%
lookup_file_name = f"gs://{bucket_name}/evaluation-pipeline/SAYT/Lookup_IT3_Final.csv"
sayt_df = pd.read_csv(lookup_file_name, dtype=str).rename(
    columns={"SIC07": "label", "SIC_lookup": "text"}
)
sayt_df["label"] = sayt_df["label"].apply(lambda x: x if len(x) == 5 else f"0{x}")
# Save the corpus in the format expected by classifai VectorStore (csv with 'label' and 'text' columns) in a local directory
tmp_corpus_file = out_dir + "/sayt_corpus.txt"
sayt_df.to_csv(tmp_corpus_file, index=False)
tmp_db_dir = out_dir + "/sayt_vector_store"

# %%
matching_df = pd.read_excel(
    f"gs://{bucket_name}/evaluation-pipeline/SAYT/SAYT matching.xlsx",
    dtype=str,
    nrows=100,  # Excel formatting causes 10s of thousands of blank input rows after the real 100
)

matching_df["Correct SIC code"] = matching_df["Correct SIC code"].apply(
    lambda x: x if len(x) == 5 else f"0{x}"
)
matching_df["query_id"] = [f"{idx}" for idx in range(len(matching_df))]

# %%
hfv = HuggingFaceVectoriser("sentence-transformers/all-MiniLM-L6-v2")
# hfv = HuggingFaceVectoriser( # Note - this version requires current `main` branch of classifAI, not yet available in existing releases
#   'nomic-ai/nomic-embed-text-v1.5',
#   model_kwargs={'trust_remote_code': True},
#   model_kwargs={'trust_remote_code': True}
# )
gcpv = GcpVectoriser(
    project_id=project_id,  # Note - this version requires access to a GCP project where VertexAI has been enabled
    task_type="SEMANTIC_SIMILARITY",
    location="europe-west2",
    model_name="text-embedding-004",
    vertexai=True,
)


# %%
class MockSAYTVectoriser(VectoriserBase):
    # Note:
    # This isn't a perfect replication of the SAYT
    #   vectorisation process, but it is pretty close.
    # The 'hybrid' scoring approach is also dropped here,
    #   which may have a small impact
    #   (different approach based on number of search results).
    def __init__(self, corpus):
        self.model = CountVectorizer(
            analyzer="char_wb",
            lowercase=True,
            stop_words="english",
            max_df=0.2,
            ngram_range=(3, 3),
        )
        self.model.fit(corpus)
        super().__init__()

    def transform(self, texts: list[str] | str):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.transform(texts).toarray()


# %%
hf_vs = VectorStore(
    file_name=tmp_corpus_file,
    data_type="csv",
    vectoriser=hfv,
    batch_size=8,
    output_dir=f"{tmp_db_dir}/hf_vs",
    overwrite=True,
)

# Note: you'll need an active GCP project to use this one - if that's not available,
#       remove all GCP related things in this file, and just look at the
#       huggingface vectoriser results instead (it's almost as good).
gcp_vs = VectorStore(
    file_name=tmp_corpus_file,
    data_type="csv",
    vectoriser=gcpv,
    batch_size=128,
    output_dir=f"{tmp_db_dir}/gcp_vs",
    overwrite=True,
)

# %%
sayt_v = MockSAYTVectoriser(sayt_df["text"])
sayt_vs = VectorStore(
    file_name=tmp_corpus_file,
    data_type="csv",
    vectoriser=sayt_v,
    batch_size=8,
    output_dir=f"{tmp_db_dir}/sayt_vs",
    overwrite=True,
)

# %%
matching_search_data_full = VectorStoreSearchInput(
    {
        "id": matching_df.index.to_list(),
        "query": matching_df["Full entry looking for"].to_list(),
    }
)

matching_search_data_partial = VectorStoreSearchInput(
    {
        "id": matching_df.index.to_list(),
        "query": matching_df["Search term used first 5 characters"].to_list(),
    }
)


def collate_matches(vector_store, search_input, out_col_name):
    search_results = vector_store.search(search_input, n_results=10)
    search_results["doc_label"] = search_results["doc_label"].apply(
        lambda x: x if len(x) == 5 else f"0{x}"
    )
    search_results_gpby = search_results.groupby("query_id", as_index=False)
    return search_results_gpby.agg(list)[["query_id", "doc_label", "doc_text"]].rename(
        columns={
            "doc_label": out_col_name + "_codes",
            "doc_text": out_col_name + "_descriptions",
        }
    )


hf_matches_full = collate_matches(
    hf_vs, matching_search_data_full, "hf_ordered_matches_FULL"
)
gcp_matches_full = collate_matches(
    gcp_vs, matching_search_data_full, "gcp_ordered_matches_FULL"
)
sayt_matches_full = collate_matches(
    sayt_vs, matching_search_data_full, "sayt_ordered_matches_FULL"
)
hf_matches_partial = collate_matches(
    hf_vs, matching_search_data_partial, "hf_ordered_matches_PARTIAL"
)
gcp_matches_partial = collate_matches(
    gcp_vs, matching_search_data_partial, "gcp_ordered_matches_PARTIAL"
)
sayt_matches_partial = collate_matches(
    sayt_vs, matching_search_data_partial, "sayt_ordered_matches_PARTIAL"
)

# %%
matching_df = pd.merge(
    matching_df,
    hf_matches_full,
    left_on="query_id",
    right_on="query_id",
    suffixes=("", ""),
)
matching_df = pd.merge(
    matching_df,
    gcp_matches_full,
    left_on="query_id",
    right_on="query_id",
    suffixes=("", ""),
)
matching_df = pd.merge(
    matching_df,
    sayt_matches_full,
    left_on="query_id",
    right_on="query_id",
    suffixes=("", ""),
)
matching_df = pd.merge(
    matching_df,
    hf_matches_partial,
    left_on="query_id",
    right_on="query_id",
    suffixes=("", ""),
)
matching_df = pd.merge(
    matching_df,
    gcp_matches_partial,
    left_on="query_id",
    right_on="query_id",
    suffixes=("", ""),
)
matching_df = pd.merge(
    matching_df,
    sayt_matches_partial,
    left_on="query_id",
    right_on="query_id",
    suffixes=("", ""),
)

useful_cols = [
    "Correct SIC code",
    "Full entry looking for",
    "Search term used first 5 characters",
    "hf_ordered_matches_FULL_codes",
    "hf_ordered_matches_FULL_descriptions",
    "hf_ordered_matches_PARTIAL_codes",
    "hf_ordered_matches_PARTIAL_descriptions",
    "gcp_ordered_matches_FULL_codes",
    "gcp_ordered_matches_FULL_descriptions",
    "gcp_ordered_matches_PARTIAL_codes",
    "gcp_ordered_matches_PARTIAL_descriptions",
    "sayt_ordered_matches_FULL_codes",
    "sayt_ordered_matches_FULL_descriptions",
    "sayt_ordered_matches_PARTIAL_codes",
    "sayt_ordered_matches_PARTIAL_descriptions",
]

# %%
print("Saving output file in .csv and .parquet formats")
matching_df[useful_cols].to_csv(
    f"{out_dir}/SAYT_semantic_search_results.csv", index=False
)
try:
    matching_df[useful_cols].to_parquet(
        f"{out_dir}/SAYT_semantic_search_results.parquet", index=False
    )
except ImportError:
    print("failed to save parquet file of results, skipping...")


# %%
def get_accuracy(row, col, top_k):
    ans = row["Correct SIC code"]
    top_k = row[col][:top_k]
    return ans in top_k


# %%
for solution in ["hf", "gcp", "sayt"]:
    print(
        f"\n\nUsing {solution} vectoriser approach:\n--------------------------------"
    )
    for top_k in [1, 3, 5, 10]:
        print(f"---------------------------\nAccuracy for top-{top_k} matches:")
        for col in [
            f"{solution}_ordered_matches_FULL_codes",
            f"{solution}_ordered_matches_PARTIAL_codes",
        ]:
            print(
                f"""{col} accuracy: {
                    (matching_df.apply(lambda x, col=col, top_k=top_k: get_accuracy(x, col, top_k), axis=1)).sum()
                    }%"""
            )
# %%
"""
'Hybrid' search idea:
---------------------
- create both (replication of) current BLAISE n-gram vectoriser, and a 'semantic' vectoriser <-- both normalised
- search against both for two distinct shortlists
- calculate weighting for n-gram vs semantic scores based on number of characters entered
- create merged shortlist based on initial ones & weighting (TBD - how to handle same codes, different examples)
- considerations: speed/performance? Would need to be quite quick, may impact whether to use GCP vectoriser

Other ideas:
------------
- TF-IDF with or without PCA to replace current n-gram solution?
"""
