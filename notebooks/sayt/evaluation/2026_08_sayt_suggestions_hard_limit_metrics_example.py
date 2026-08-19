"""A script showing an example of using SAYT suggestions hard limit metrics."""

# pylint: disable=invalid-name
# pylint: disable=duplicate-code

# %%
import os

import pandas as pd
from dotenv import load_dotenv
from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SemanticRetrieverSpec,
)
from survey_assist_utils.logging import get_logger

from notebooks.sayt.evaluation.hard_limit_metrics_functions import (
    build_sayt_hard_limit_metrics_comparison_table,
    compute_suggestions_hard_limit_metrics,
)
from notebooks.sayt.sayt_utils import (
    build_lookup_suggester,
    build_sayt_corpus_from_df,
    get_suggestions_by_chars,
    validate_one_code,
)
from survey_assist_eval.evaluation.sayt.performance_metrics_functions import (
    build_sayt_metrics_comparison_table,
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
sic_kb_for_classifai = pd.read_csv(
    f"gs://{bucket_name}/sic_knowledgebase/sic_kb_for_sayt.csv", dtype=str
)

_, sayt2_corpus = build_sayt_corpus_from_df(
    df=sic_kb_for_classifai,
    search_text_col="search_text",
    display_text_col="display_text",
    code_col="code",
    expected_code_length=SIC_CODE_LENGTH,
    incl_code_in_display=True,
)

# %%
# define bunch of different suggesters to evaluate
suggesters = {
    "Ngrams only": build_lookup_suggester(
        sayt2_corpus, retrievers=[NgramRetrieverSpec()]
    ),
    "Prefix only": build_lookup_suggester(
        sayt2_corpus, retrievers=[PrefixRetrieverSpec()]
    ),
    "Semantic only": build_lookup_suggester(
        sayt2_corpus, retrievers=[SemanticRetrieverSpec()]
    ),
}

# %%
# get suggestions for the test queries with a hard limit on the number of suggestions returned
test_df_hard_limit, avg_ms_dict = get_suggestions_by_chars(
    df=test_df,
    suggesters_dict=suggesters,
    characters=[4, 5, 7, 10],
    suggestions_limit=MAX_SUGGESTIONS,
    hard_suggestions_limit=True,
    with_scores=True,
)

suggestions_cols_to_compare = test_df_hard_limit.columns[
    test_df_hard_limit.columns.str.startswith("suggestions_")
].tolist()


compare_performance_metrics_hard_limit = build_sayt_metrics_comparison_table(
    test_df_hard_limit,
    suggestions_cols_to_compare=suggestions_cols_to_compare,
    correct_code_col=correct_code_col,
    k_values=[1, 3, 5, MAX_SUGGESTIONS],
    ave_time_per_query_dict=avg_ms_dict,
)

compare_performance_metrics_hard_limit.head()

# %%
# compute hard limit metrics for suggestions when no hard limit applied
test_df_none_hard_limit, avg_ms_dict = get_suggestions_by_chars(
    df=test_df,
    suggesters_dict=suggesters,
    characters=[4, 5, 7, 10],
    suggestions_limit=MAX_SUGGESTIONS,
    hard_suggestions_limit=False,
    with_scores=True,
)

compute_suggestions_hard_limit_metrics(
    test_df_none_hard_limit,
    correct_code_col=correct_code_col,
    suggestions_col=suggestions_cols_to_compare[0],
    cutoff_k=MAX_SUGGESTIONS,
    code_length=SIC_CODE_LENGTH,
    score_col=suggestions_cols_to_compare[0].replace("suggestions_", "scores_"),
).report_metrics()

suggestions_cols_to_compare = test_df_none_hard_limit.columns[
    test_df_none_hard_limit.columns.str.startswith("suggestions_")
].tolist()

compare_hard_limit_metrics = build_sayt_hard_limit_metrics_comparison_table(
    test_df_none_hard_limit,
    suggestions_cols_to_compare=suggestions_cols_to_compare,
    correct_code_col=correct_code_col,
    cutoff_k=MAX_SUGGESTIONS,
)

compare_hard_limit_metrics.head()
# %%
