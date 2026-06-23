"""Run small tests for industry/organisation descriptions SAYT.

Expects following environment variables to be set:
- EVALUATION_BUCKET_NAME: name of GCS bucket where the data is stored
The variables are loaded from the ".env" file.
"""

# ruff: noqa: PLR2004
# pylint: disable=protected-access,redefined-outer-name,C0103

# %%
import os

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from industrial_classification_utils.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SAYTSuggester,
    SemanticRetrieverSpec,
)

from survey_assist_eval.data_cleaning.code_standard import get_clean_n_digit_codes

# pylint: disable=R0801

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")

print(f"Using bucket for data loading: {bucket_name}")


def build_lookup_suggester(
    corpus: list[tuple[str, str]], *, semantic_weight: float | None
) -> SAYTSuggester:
    """Build a lookup suggester using the explicit retriever-spec API."""
    retrievers = [PrefixRetrieverSpec(), NgramRetrieverSpec()]
    if semantic_weight is not None:
        retrievers.append(SemanticRetrieverSpec(weight=semantic_weight))
    return SAYTSuggester(corpus, retrievers=retrievers)


# %%
test_df = pd.read_excel(
    f"gs://{bucket_name}/evaluation-pipeline/SAYT/SAYT matching.xlsx",
    dtype=str,
    nrows=100,  # Excel formatting causes 10s of thousands of blank input rows after the real 100
)

test_df = test_df.rename(
    columns={
        "Correct SIC code": "correct_sic_code",
        "Full entry looking for": "full_entry",
    }
)
test_df = test_df[["correct_sic_code", "full_entry"]]

# %%
# check the codes are well formed
clean_codes = test_df["correct_sic_code"].apply(
    lambda x: x if pd.isna(x) else get_clean_n_digit_codes(x, n=5, code_type="SIC")[0]
)
msk = test_df["correct_sic_code"] != clean_codes.map(
    lambda x: x if pd.isna(x) else next(iter(x))
)
if msk.any():
    print("Found malformed codes:")
    print(test_df[msk])

# %%
lookup_file_name = f"gs://{bucket_name}/evaluation-pipeline/SAYT/Lookup_IT3_Final.csv"
sayt_df = pd.read_csv(lookup_file_name, dtype=str)
sayt_df["code"] = sayt_df["SIC07"].apply(lambda x: x if len(x) == 5 else f"0{x}")
sayt_df["display_text"] = sayt_df["SIC_lookup"] + ": " + sayt_df["code"]

sayt_corpus = list(zip(sayt_df["SIC_lookup"], sayt_df["display_text"], strict=False))
sayt_suggester_without_sem = build_lookup_suggester(sayt_corpus, semantic_weight=None)
sayt_suggester_with_sem10 = build_lookup_suggester(sayt_corpus, semantic_weight=1.0)
sayt_suggester_with_sem05 = build_lookup_suggester(sayt_corpus, semantic_weight=0.5)
sayt_suggester_with_sem15 = build_lookup_suggester(sayt_corpus, semantic_weight=1.5)


# %%
def get_suggestions_for_row(row, suggester, num_chars, max_suggestions):
    """Return suggester output for a single input row."""
    return suggester.suggest(
        row["full_entry"][:num_chars],
        num_suggestions=max_suggestions,
    )


def rank_of_correct_code_in_suggestions(
    row, num_chars, suggester_label, correct_code_col="correct_sic_code"
):
    """Return the rank of the correct code in the suggestions, or None if not found."""
    correct_code = row[correct_code_col]
    suggestions = row[f"suggestions_{num_chars}chars_{suggester_label}"]
    for rank, suggest in enumerate(suggestions):
        if suggest[-5:] == correct_code:
            return rank + 1
    return None


# %%
MAX_SUGGESTIONS = 20
for num_chars in [4, 5, 7, 9, 150]:
    for suggester_label, suggester in [
        ("without_sem", sayt_suggester_without_sem),
        ("with_sem05", sayt_suggester_with_sem05),
        ("with_sem10", sayt_suggester_with_sem10),
        ("with_sem15", sayt_suggester_with_sem15),
    ]:

        test_df[f"suggestions_{num_chars}chars_{suggester_label}"] = test_df.apply(
            get_suggestions_for_row,
            suggester=suggester,
            max_suggestions=MAX_SUGGESTIONS,
            num_chars=num_chars,
            axis=1,
        )
        test_df[f"rank_{num_chars}chars_{suggester_label}"] = test_df.apply(
            rank_of_correct_code_in_suggestions,
            correct_code_col="correct_sic_code",
            suggester_label=suggester_label,
            num_chars=num_chars,
            axis=1,
        )

# %%
# melt results by suggester and num_chars for easier analysis
results_df = test_df.melt(
    id_vars=["correct_sic_code", "full_entry"],
    value_vars=[col for col in test_df.columns if col.startswith("rank_")],
    var_name="suggester_numchars",
    value_name="rank",
)
results_df["num_chars"] = results_df["suggester_numchars"].apply(
    lambda x: int(x.split("_")[1].replace("chars", ""))
)
results_df["suggester"] = results_df["suggester_numchars"].apply(
    lambda x: " ".join(x.split("_")[-2:])
)
results_df.loc[results_df["rank"] > MAX_SUGGESTIONS, "rank"] = None
results_df["rank"] = results_df["rank"].fillna(
    MAX_SUGGESTIONS + 2
)  # Treat not found as worst rank

# %%
# compare rank histograms for the two suggesters at different num_chars
fig = px.histogram(
    results_df,
    x="rank",
    color="suggester",
    facet_col="num_chars",
    category_orders={"rank": list(range(0, MAX_SUGGESTIONS + 2))},
    barmode="group",  # next to each other
    # use frequencies
    title="Distribution of Ranks of Correct Code in Suggestions by Number of Characters",
)
# labels on x axis are too crowded, so just show every 5th but use 'NA' for 21
fig.update_xaxes(
    tickmode="array",
    tickvals=[*range(0, MAX_SUGGESTIONS, 5), MAX_SUGGESTIONS + 2],
    ticktext=[str(i) for i in range(0, MAX_SUGGESTIONS, 5)] + ["NA"],
)

fig.update_layout(bargap=0.1, width=1600)
fig.show()

# %%
