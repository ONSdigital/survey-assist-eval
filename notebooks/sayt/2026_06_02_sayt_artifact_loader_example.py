"""Load and demo a SAYT artifact built from the IT3 lookup.

Run ``2026_06_01_sayt_artifact_builder_example.py`` before this notebook.
"""

# pylint: disable=C0103,R0801,duplicate-code

# %%
import json
from pathlib import Path

from industrial_classification_utils.sayt import SAYTSuggester

# %%
ARTIFACT_DIR = Path("data") / "sayt_artifacts" / "lookup_it3_final"
QUERIES = ["car", "cars", "waxi", "auto", "hea", "heal", "health"]

print("Working directory:", Path.cwd().resolve())
print("Artifact input directory:", ARTIFACT_DIR.resolve())

# %%
if not ARTIFACT_DIR.exists():
    raise FileNotFoundError(
        "Artifact directory not found. Run 2026_06_01_sayt_artifact_builder_example.py first."
    )

manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
print(json.dumps(manifest, indent=2))

# %%
# Semantic artifact loading may still take a little time the first time if the
# local model cache is not already available.
loaded_suggester = SAYTSuggester.from_artifact(ARTIFACT_DIR)

# %%
for query in QUERIES:
    print("searching for:", query)
    print("loaded", "->", loaded_suggester.suggest(query, 5))
    print("loaded_scores", "->", loaded_suggester.suggest_with_scores(query, 5))
    print()

# %%
