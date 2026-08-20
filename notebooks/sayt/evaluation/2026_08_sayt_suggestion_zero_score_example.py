"""A script showing SAYT suggestion zero score examples."""

# pylint: disable=invalid-name
# pylint: disable=duplicate-code

# %%
import os

import pandas as pd
from dotenv import load_dotenv
from IPython.display import display
from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SemanticRetrieverSpec,
)
from survey_assist_utils.logging import get_logger

from notebooks.sayt.sayt_utils import (
    build_lookup_suggester,
    get_suggestions_by_chars,
    validate_one_code,
)

# %%
SIC_CODE_LENGTH = 5
MAX_SUGGESTIONS = 9  # for the evaluation we will look at ranks up to 9 only
correct_code_col = "correct_sic_code"

load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")


logger = get_logger(__name__)


# %%
test_df = pd.read_excel(
    f"gs://{bucket_name}/evaluation-pipeline/SAYT/SAYT matching.xlsx",
    dtype=str,
    nrows=100,  # Excel formatting causes 10s of thousands of blank input rows after the real 100
    header=1,  # first row is header
)
rename_columns = {
    "Correct SIC code": correct_code_col,
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
    f"Clerical codes validated: {
        test_df[correct_code_col]
        .apply(validate_one_code,
               code_length=SIC_CODE_LENGTH)
               .all()
               }"
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
    "Prefix only": build_lookup_suggester(
        sayt_corpus, retrievers=[PrefixRetrieverSpec()], semantic_weight=None
    ),
    "Ngrams only": build_lookup_suggester(
        sayt_corpus, retrievers=[NgramRetrieverSpec()], semantic_weight=None
    ),
    "Semantic only": build_lookup_suggester(
        sayt_corpus,
        retrievers=[SemanticRetrieverSpec(weight=1.0)],
        semantic_weight=None,
    ),
    "All": build_lookup_suggester(
        sayt_corpus,
        retrievers=[PrefixRetrieverSpec(), NgramRetrieverSpec()],
        semantic_weight=1.0,
    ),
}


# %%

test_df_with_suggestions, avg_ms_dict = get_suggestions_by_chars(
    df=test_df,
    suggesters_dict=suggesters,
    characters=[4, 5, 6, 7, 8, 9],
    suggestions_limit=MAX_SUGGESTIONS,
    with_scores=True,
)

suggestions_cols_to_compare = test_df_with_suggestions.columns[
    test_df_with_suggestions.columns.str.startswith("suggestions_")
].tolist()

# %%
# Check for suggestions that have score 0
for col in suggestions_cols_to_compare:
    scores_col = col.replace("suggestions_", "scores_")
    test_df_with_suggestions[f"{col}_has_score_0"] = test_df_with_suggestions[
        scores_col
    ].apply(lambda x: 0 in x if isinstance(x, list) else False)

    count_with_zero = test_df_with_suggestions[f"{col}_has_score_0"].sum()
    if count_with_zero > 0:
        logger.warning(
            f"Entries with score 0 in {col}: {int(count_with_zero)}",
            column=col,
            count=int(count_with_zero),
            total_entries=len(test_df_with_suggestions),
        )

        rows_with_zero = test_df_with_suggestions[
            test_df_with_suggestions[f"{col}_has_score_0"]
        ][["full_entry", correct_code_col, col, scores_col]]

        with pd.option_context(
            "display.max_colwidth",
            None,
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            0,
        ):
            display(rows_with_zero)


# %%
