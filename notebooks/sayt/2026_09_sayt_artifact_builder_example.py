"""Build a SAYT artifact from the sic_kb_for_sayt lookup for later notebook loading.

Run this notebook before ``2026_09_sayt_artifact_loader_example.py``.

Expects following environment variables to be set:
- EVALUATION_BUCKET_NAME: name of GCS bucket where the data is stored
The variables are loaded from the ".env" file.

Note: This is an updated version of 2026_06_01_sayt_artifact_builder_example.py,
      now using the extended sic_kb_for_sayt lookup, the new repository
      survey-assist-embed-core and the updated SAYT artifact structure.
"""

# pylint: disable=C0103,R0801,duplicate-code

# %%
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    NgramWeightSpec,
    PrefixRetrieverSpec,
    PrefixWeightSpec,
    SAYTBuilder,
    SemanticRetrieverSpec,
    SemanticWeightSpec,
    WeightSpecs,
)

from notebooks.sayt.sayt_utils import build_sayt_corpus_from_df

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")

LOOKUP_FILE_NAME = f"gs://{bucket_name}/sic_knowledgebase/sic_kb_for_sayt.csv"
ARTIFACT_DIR = (
    Path(__file__).parent.parent.parent / "data" / "sayt_artifacts" / "sic_kb_for_sayt"
)
RETRIEVERS = [
    PrefixRetrieverSpec(),
    NgramRetrieverSpec(),
    SemanticRetrieverSpec(),
]

WEIGHTSPECS = WeightSpecs(
    specs=[
        PrefixWeightSpec(),
        NgramWeightSpec(),
        SemanticWeightSpec(),
    ]
)

print(f"Using bucket for data loading: {bucket_name}")
print("Working directory:", Path.cwd().resolve())
print("Artifact output directory:", ARTIFACT_DIR.resolve())

# %%
sayt_df = pd.read_csv(LOOKUP_FILE_NAME, dtype=str)
_, sayt_corpus = build_sayt_corpus_from_df(
    sayt_df,
    search_text_col="search_text",
    display_text_col="display_text",
    code_col="code",
)
print(f"Loaded {len(sayt_corpus)} lookup rows from {LOOKUP_FILE_NAME}")

# %%
ARTIFACT_DIR.parent.mkdir(parents=True, exist_ok=True)

# Semantic artifact builds may take longer the first time if the model cache
# needs to be created locally.
artifact_path = SAYTBuilder(
    sayt_corpus,
    retrievers=RETRIEVERS,
    weights=WEIGHTSPECS,
    min_chars=3,
    max_suggestions=5,
).build_artifact(ARTIFACT_DIR, overwrite=True)

print("Artifact saved to:", artifact_path.resolve())
print("Artifact files:")
for path in sorted(artifact_path.rglob("*")):
    if path.is_file():
        print("-", path.relative_to(artifact_path))

# %%
manifest = json.loads((artifact_path / "manifest.json").read_text(encoding="utf-8"))
print(json.dumps(manifest, indent=2))

# %%
