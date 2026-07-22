"""Run small tests for industry/organisation descriptions SAYT.

Expects following environment variables to be set:
- EVALUATION_BUCKET_NAME: name of GCS bucket where the data is stored
The variables are loaded from the ".env" file.
"""

# pylint: disable=invalid-name

# %%
import logging
import os
import time

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SAYTSuggester,
    SemanticRetrieverSpec,
)
from survey_assist_utils.logging import get_logger

from survey_assist_eval.data_cleaning.code_standard import get_clean_n_digit_codes

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
def validate_one_code(code: str, n_digits=SIC_CODE_LENGTH) -> bool:
    """Return a set of cleaned codes, or raise ValueError if the code is malformed."""
    if pd.isna(code):
        logger.warning("Code is NaN")
        return False
    clean_codes = get_clean_n_digit_codes(code, n=n_digits, code_type="SIC")
    if len(clean_codes[1]) != 0:
        logger.warning(f"Malformed code: {code}")
        return False
    if len(clean_codes[0]) != 1 or next(iter(clean_codes[0])) != code:
        logger.warning(f"Code {code} cleaned to different code: {clean_codes[0]}")
        return False
    return True


print(
    f'Clerical codes validated: {test_df["correct_sic_code"].apply(validate_one_code).all()}'
)

# %%
LOOKUP_FILE_NAME = f"gs://{bucket_name}/evaluation-pipeline/SAYT/Lookup_IT3_Final.csv"
sayt_df = pd.read_csv(LOOKUP_FILE_NAME, dtype=str)
sayt_df["code"] = sayt_df["SIC07"].apply(
    lambda x: x if len(x) == SIC_CODE_LENGTH else f"0{x}"
)
sayt_df["display_text"] = sayt_df["SIC_lookup"] + ": " + sayt_df["code"]

sayt_corpus = list(zip(sayt_df["SIC_lookup"], sayt_df["display_text"], strict=False))

# %%
sic_kb_for_classifai = pd.read_csv(
    f"gs://{bucket_name}/sic_knowledgebase/sic_kb_for_sayt.csv", dtype=str
)

sayt2_corpus = list(
    zip(
        sic_kb_for_classifai["search_text"],
        sic_kb_for_classifai["display_text"],
        strict=False,
    )
)

# %%
# define bunch of different suggesters to evaluate
suggesters = {
    "Blaise proxy method (prefix + n_grams)": build_lookup_suggester(
        sayt_corpus, semantic_weight=None
    ),
    "Hybrid method including semantic retriever": build_lookup_suggester(
        sayt_corpus, semantic_weight=1.0
    ),
    "Hybrid method with extended knowledge base": build_lookup_suggester(
        sayt2_corpus, semantic_weight=1.0
    ),
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
        if suggest[-SIC_CODE_LENGTH:] == correct_code:
            return rank + 1
    return None


# %%

for prefix_chars in [4, 5, 7, 10]:  # 150]:
    for suggester_name, suggester_obj in suggesters.items():
        logger.info(
            "Starting SAYT suggesting - one loop",
            num_chars=prefix_chars,
            suggester_label=suggester_name,
        )

        t_start = time.perf_counter()
        test_df[f"suggestions_{prefix_chars}chars_{suggester_name}"] = test_df.apply(
            get_suggestions_for_row,
            suggester=suggester_obj,
            max_suggestions=MAX_SUGGESTIONS,
            num_chars=prefix_chars,
            axis=1,
        )
        elapsed = time.perf_counter() - t_start
        logger.info(
            "  -> suggestions done",
            elapsed_sec=elapsed,
            elapsed_per_row_ms=elapsed / len(test_df) * 1000,
        )
        test_df[f"rank_{prefix_chars}chars_{suggester_name}"] = test_df.apply(
            rank_of_correct_code_in_suggestions,
            correct_code_col="correct_sic_code",
            suggester_label=suggester_name,
            num_chars=prefix_chars,
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
    lambda x: " ".join(x.split("_")[2:])
)
results_df.loc[results_df["rank"] > MAX_SUGGESTIONS, "rank"] = None
results_df["rank"] = results_df["rank"].fillna(
    MAX_SUGGESTIONS + 2
)  # Treat not found as worst rank
results_df["rank"] = results_df["rank"].astype(int)

results_df = results_df.sort_values(by=["num_chars", "suggester", "rank"]).reset_index(
    drop=True
)

# %%
# compare rank histograms for the two suggesters at different num_chars
fig = px.histogram(
    results_df,
    x="rank",
    color="suggester",
    facet_col="num_chars",
    category_orders={
        "rank": list(range(0, MAX_SUGGESTIONS + 2)),
        "suggester": sorted(results_df["suggester"].unique().tolist()),
    },
    barmode="group",
    title=(
        "Distribution of Ranks of Correct Code in Suggestions by Number of Characters"
        + " (on SAYT test data of 100 examples)"
    ),
)
fig.update_xaxes(
    tickmode="array",
    tickvals=[*range(1, MAX_SUGGESTIONS + 1, 1), MAX_SUGGESTIONS + 2],
    ticktext=[str(i) for i in range(1, MAX_SUGGESTIONS + 1, 1)] + ["NA"],
)

fig.update_layout(
    bargap=0.1,
    legend={
        "title": "Suggester method",
        "orientation": "h",
        "yanchor": "top",
        "y": -0.2,
        "xanchor": "center",
        "x": 0.5,
    },
)
fig.show()

fig.write_html(f"{OUTPUT_DIR}/sayt_eval_100sample_rank_histograms.html")

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
