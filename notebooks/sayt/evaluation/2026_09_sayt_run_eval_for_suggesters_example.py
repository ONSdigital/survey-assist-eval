"""A script showing an example of running SAYT evaluation for different suggesters.

Expects following environment variables to be set:
- EVALUATION_BUCKET_NAME: name of GCS bucket where the data is stored
The variables are loaded from the ".env" file.
"""

# pylint: disable=invalid-name
# pylint: disable=duplicate-code

# %%
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SemanticRetrieverSpec,
)
from survey_assist_utils.logging import get_logger

from notebooks.sayt.sayt_utils import (
    build_lookup_suggester,
    build_sayt_corpus_from_df,
    validate_one_code,
)
from notebooks.sayt.suggester_eval import run_eval_for_suggesters

# %%
SIC_CODE_LENGTH = 5
MAX_SUGGESTIONS = 9
CORRECT_CODE_COL = "correct_sic_code"
NUM_CHARACTERS_LIST = list(range(4, 10))
OUTPUT_DIR = "data/figures/sayt/min_characters"

# %%
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
correct_codes_col = CORRECT_CODE_COL

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
    "Correct SIC code": correct_codes_col,
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
        test_df[correct_codes_col]
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
suggestions_df, fig, metrics_table = run_eval_for_suggesters(
    df=test_df,
    suggesters_dict=suggesters,
    num_chars=NUM_CHARACTERS_LIST,
    suggestions_limit=MAX_SUGGESTIONS,
    correct_codes_col=correct_codes_col,
    output_dir=f"{OUTPUT_DIR}_all",
)

metrics_table.head()

# %%
# digit match to 2
suggestions_df_digit2, fig_digit2, metrics_table_digit2 = run_eval_for_suggesters(
    df=test_df,
    suggesters_dict=suggesters,
    num_chars=NUM_CHARACTERS_LIST,
    suggestions_limit=MAX_SUGGESTIONS,
    correct_codes_col=correct_codes_col,
    output_dir=f"{OUTPUT_DIR}_digit2",
    code_digit_match_length=2,
)

metrics_table_digit2.head()

# %%
# Example when the correct codes are a list of codes rather than a single code
test_df_list_codes = pd.read_parquet(
    f"gs://{bucket_name}/evaluation-pipeline/original_datasets/sic_2k/sic_2k_test_data.parquet"
)

is_self_employed = test_df_list_codes["sic2007_employee"] == "-9"

test_df_list_codes["full_entry"] = np.where(
    is_self_employed,
    test_df_list_codes["sic2007_self_employed"],
    test_df_list_codes["sic2007_employee"],
)
test_df_list_codes["employment_status"] = np.where(
    is_self_employed, "self_employed", "employed"
)

test_df_list_codes = test_df_list_codes.rename(
    columns={"clerical_codes": correct_codes_col}
)

# %%
suggestions_df_unambiguous, fig_unambiguous, metrics_table_unambiguous = (
    run_eval_for_suggesters(
        df=test_df_list_codes,
        suggesters_dict=suggesters,
        num_chars=NUM_CHARACTERS_LIST,
        suggestions_limit=MAX_SUGGESTIONS,
        correct_codes_col=correct_codes_col,
        output_dir=f"{OUTPUT_DIR}_all_list_codes",
        only_unambiguous_correct_codes=True,
    )
)

metrics_table_unambiguous.head()

# %%
# digit match to 2
(
    suggestions_df_list_codes_digit2,
    fig_list_codes_digit2,
    metrics_table_list_codes_digit2,
) = run_eval_for_suggesters(
    df=test_df_list_codes,
    suggesters_dict=suggesters,
    num_chars=NUM_CHARACTERS_LIST,
    suggestions_limit=MAX_SUGGESTIONS,
    correct_codes_col=correct_codes_col,
    output_dir=f"{OUTPUT_DIR}_digit2_list_codes",
    code_digit_match_length=2,
)

metrics_table_list_codes_digit2.head()
# %%
