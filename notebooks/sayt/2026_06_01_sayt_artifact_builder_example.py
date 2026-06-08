"""Build a SAYT artifact from the IT3 lookup for later notebook loading.

Run this notebook before ``2026_06_02_sayt_artifact_loader_example.py``.

Expects following environment variables to be set:
- EVALUATION_BUCKET_NAME: name of GCS bucket where the data is stored
The variables are loaded from the ".env" file.
"""

# ruff: noqa: PLR2004
# pylint: disable=C0103,R0801,duplicate-code

# %%
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from industrial_classification_utils.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SAYTBuilder,
    SemanticRetrieverSpec,
)

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")

LOOKUP_FILE_NAME = f"gs://{bucket_name}/evaluation-pipeline/SAYT/Lookup_IT3_Final.csv"
ARTIFACT_DIR = Path("data") / "sayt_artifacts" / "lookup_it3_final"
RETRIEVERS = [
    PrefixRetrieverSpec(),
    NgramRetrieverSpec(),
    SemanticRetrieverSpec(),
]

print(f"Using bucket for data loading: {bucket_name}")
print("Working directory:", Path.cwd().resolve())
print("Artifact output directory:", ARTIFACT_DIR.resolve())

# %%
sayt_df = pd.read_csv(LOOKUP_FILE_NAME, dtype=str)
sayt_df["code"] = sayt_df["SIC07"].apply(
    lambda x: x if (pd.notna(x) and len(x) == 5) else f"0{x}" if pd.notna(x) else x
)
sayt_df["display_text"] = sayt_df["SIC_lookup"] + ": " + sayt_df["code"]

sayt_corpus = list(zip(sayt_df["SIC_lookup"], sayt_df["display_text"], strict=False))
print(f"Loaded {len(sayt_corpus)} lookup rows from {LOOKUP_FILE_NAME}")

# %%
ARTIFACT_DIR.parent.mkdir(parents=True, exist_ok=True)

# Semantic artifact builds may take longer the first time if the model cache
# needs to be created locally.
artifact_path = SAYTBuilder(
    sayt_corpus,
    retrievers=RETRIEVERS,
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
