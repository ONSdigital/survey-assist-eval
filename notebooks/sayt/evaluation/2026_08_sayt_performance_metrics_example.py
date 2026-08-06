"""A script showing an example of using SAYT performance metrics."""

# pylint: disable=invalid-name
# pylint: disable=duplicate-code

# %%
import os

import pandas as pd
from dotenv import load_dotenv
from survey_assist_utils.logging import get_logger

from notebooks.sayt.evaluation.performance_metrics_functions import (
    build_sayt_metrics_comparison_table,
    compute_performance_metrics_from_suggestions,
)
from notebooks.sayt.sayt_utils import (
    build_lookup_suggester,
    get_suggestions_for_row,
    timed_apply,
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
               logger=logger,
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


# %%
ave_elapsed_per_row_list = []
suggestions_cols_to_compare = []
for prefix_chars in [4, 5, 7, 10]:  # 150]:
    for suggester_name, suggester_obj in suggesters.items():
        logger.info(
            "Starting SAYT suggesting - one loop",
            num_chars=prefix_chars,
            suggester_label=suggester_name,
        )

        test_df[f"suggestions_{prefix_chars}chars_{suggester_name}"], avg_ms = (
            timed_apply(
                test_df,
                get_suggestions_for_row,
                suggester=suggester_obj,
                max_suggestions=MAX_SUGGESTIONS,
                num_chars=prefix_chars,
                axis=1,
            )
        )
        logger.info(
            "  -> suggestions done",
            ave_elapsed_per_row_ms=avg_ms,
        )
        ave_elapsed_per_row_list.append(avg_ms)

        suggestions_cols_to_compare.append(
            f"suggestions_{prefix_chars}chars_{suggester_name}"
        )

# %%
# Performance metrics for one suggester and prefix length
metrics = compute_performance_metrics_from_suggestions(
    test_df,
    correct_code_col=correct_code_col,
    suggestions_col=suggestions_cols_to_compare[0],
    code_length=SIC_CODE_LENGTH,
    k_values=[1, 3, 5, MAX_SUGGESTIONS],
    ave_time_per_query=ave_elapsed_per_row_list[0],
)

print(metrics.report_metrics())

# %%
# Comparison of performance metrics for the different suggesters and prefix lengths
compare_performance_metrics = build_sayt_metrics_comparison_table(
    test_df,
    suggestions_cols_to_compare=suggestions_cols_to_compare,
    correct_code_col=correct_code_col,
    k_values=[1, 3, 5, MAX_SUGGESTIONS],
    ave_time_per_query_list=ave_elapsed_per_row_list,
)

compare_performance_metrics.head()

# %%
