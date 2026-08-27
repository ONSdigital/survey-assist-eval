# %%
# pylint: disable=R0801, C0103
"""Check the interactions between weights used for different suggesters."""

import json

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

from notebooks.sayt.sayt_utils import (
    build_lookup_suggester,
)
from notebooks.sayt.suggester_eval import run_eval_for_suggesters

# %%
SIC_CODE_LENGTH = 5
MAX_SUGGESTIONS = 9
CORRECT_CODE_COL = "correct_sic_code"
NUM_CHARACTERS_LIST = range(5, 10)

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")

OUTPUT_DIR = "data/figures/sayt/min_characters"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

logger = get_logger(__name__)
logger.info("Location specs", bucket_name=bucket_name, output_dir=OUTPUT_DIR)

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
folder = "notebooks/sayt/weights"

for characters in NUM_CHARACTERS_LIST:

    main_file_name = f"{folder}/weight_{characters}_test_n_p_s.json"

    if os.path.exists(main_file_name):
        print(f"File {main_file_name} already exists.")
        continue
    for ngram in range(1, 11):
        for prefix in range(1, 11):
            for semantic in range(1, 11):

                sub_file_name = (
                    f"{folder}/w_{characters}_n{ngram}_p{prefix}_s{semantic}.json"
                )

                if os.path.exists(sub_file_name):
                    continue
                suggesters_three = {
                    "ngram, prefix and semantic": build_lookup_suggester(
                        sayt_corpus,
                        retrievers=[
                            NgramRetrieverSpec(weight=ngram / 10),
                            PrefixRetrieverSpec(weight=prefix / 10),
                            SemanticRetrieverSpec(weight=semantic / 10),
                        ],
                    ),
                }

                suggestions_df, fig, metrics_table = run_eval_for_suggesters(
                    df=test_df,
                    suggesters_dict=suggesters_three,
                    num_chars=[characters],
                    suggestions_limit=MAX_SUGGESTIONS,
                    correct_code_col=CORRECT_CODE_COL,
                    output_dir=f"{OUTPUT_DIR}_all",
                )

                data = {
                    "Ngram_weight": ngram,
                    "Prefix_weight": prefix,
                    "Semantic_weight": semantic,
                    "MRR": metrics_table["mrr"][0],
                    "avg_time": metrics_table["ave_time_per_query_ms"][0],
                }
                print(data)

                with open(sub_file_name, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)

# %%
# combine separate test results into one file

folderpath = "notebooks/sayt/weights"

master_dict = {}

for filename in sorted(os.listdir(folderpath)):
    if filename.startswith("w_7_n") and filename.endswith(".json"):
        full_path = os.path.join(folderpath, filename)
        key_name = filename[:-5]  # remove .json from the file name
        test_name = f"test{key_name[3:]}"
        with open(full_path, encoding="utf-8") as f:
            master_dict[test_name] = json.load(f)

with open(
    os.path.join(folderpath, "weight_7_test_n_p_s.json"), "w", encoding="utf-8"
) as f:
    json.dump(master_dict, f, indent=4)

# add removing separate test files after combining
