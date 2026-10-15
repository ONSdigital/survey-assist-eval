"""Load and demo a SAYT artifact built from the sic_kb_for_sayt lookup.

Run ``2026_09_sayt_artifact_builder_example.py`` before this notebook.
"""

# pylint: disable=C0103,R0801,duplicate-code

# %%
import json
from pathlib import Path

from survey_assist_embed_core.sayt import SAYTSuggester
from survey_assist_embed_core.sayt.core import _normalise


# %%
def print_suggester_breakdown(
    suggester_instance: SAYTSuggester, queries: list[str]
) -> None:
    """Print per-retriever and combined suggestions for the supplied queries."""
    for q in queries:
        q_norm = _normalise(q)
        print("searching for:", q)
        for (
            configured
        ) in suggester_instance._retrievers:  # pylint: disable=protected-access
            print(
                configured.name,
                "->",
                configured.retriever.suggest_with_scores(q_norm, 5),
            )
        print("combined", "->", suggester_instance.suggest_with_scores(q, 5))
        print("combined_nice", "->", suggester_instance.suggest(q, 5))
        print()


# %%
ARTIFACT_DIR = (
    Path(__file__).parent.parent.parent / "data" / "sayt_artifacts" / "sic_kb_for_sayt"
)
QUERIES = ["car", "cars", "waxi", "auto", "hea", "heal", "health"]

print("Working directory:", Path.cwd().resolve())
print("Artifact input directory:", ARTIFACT_DIR.resolve())

# %%
if not ARTIFACT_DIR.exists():
    raise FileNotFoundError(
        "Artifact directory not found. Run 2026_09_sayt_artifact_builder_example.py first."
    )

manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
print(json.dumps(manifest, indent=2))

# %%
# Semantic artifact loading may still take a little time the first time if the
# local model cache is not already available.
loaded_suggester = SAYTSuggester.from_artifact(ARTIFACT_DIR)

# %%
print_suggester_breakdown(loaded_suggester, QUERIES)

# %%
