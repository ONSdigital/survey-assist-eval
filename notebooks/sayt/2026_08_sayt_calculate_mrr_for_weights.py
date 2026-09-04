"""Check the interactions between weights used for different suggesters."""

# pylint: disable=R0801, C0103

# %%
import json
import os

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage as gcs
from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SemanticRetrieverSpec,
)
from survey_assist_utils.logging import get_logger

from notebooks.sayt.sayt_utils import (
    build_lookup_suggester,
    build_sayt_corpus_from_df,
)
from notebooks.sayt.suggester_eval import run_eval_for_suggesters

# %%
SIC_CODE_LENGTH = 5
MAX_SUGGESTIONS = 9
CORRECT_CODE_COL = "correct_sic_code"
NUM_CHARACTERS_LIST = list(range(4, 10))

GRID_GRANULARITY = 10
FOLDER = f"data/sayt/weights_grid_{GRID_GRANULARITY}"

# %%
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)
    print(f"Created folder: {FOLDER}")

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")

OUTPUT_DIR = "data/figures/sayt/character_weights"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

logger = get_logger(__name__)
logger.info("Location specs", bucket_name=bucket_name, output_dir=OUTPUT_DIR)

client = gcs.Client()
blob_name = "evaluation-pipeline/SAYT/weights_by_character/"

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
# LOOKUP_FILE_NAME = f"gs://{bucket_name}/evaluation-pipeline/SAYT/Lookup_IT3_Final.csv"
LOOKUP_FILE_NAME = f"gs://{bucket_name}/sic_knowledgebase/sic_kb_for_sayt.csv"

sayt_df = pd.read_csv(LOOKUP_FILE_NAME, dtype=str)
if LOOKUP_FILE_NAME.endswith("sic_kb_for_sayt.csv"):
    sayt_df["display_text_with_code"] = sayt_df["search_text"] + ": " + sayt_df["code"]
elif LOOKUP_FILE_NAME.endswith("Lookup_IT3_Final.csv"):
    sayt_df["code"] = sayt_df["SIC07"].apply(
        lambda x: x if len(x) == SIC_CODE_LENGTH else f"0{x}"
    )
    sayt_df["display_text_with_code"] = sayt_df["SIC_lookup"] + ": " + sayt_df["code"]

sayt_corpus = build_sayt_corpus_from_df(
    sayt_df, "search_text", "display_text_with_code", "code"
)[1]


# %%


for characters in NUM_CHARACTERS_LIST:

    main_file_name = f"{FOLDER}/weight_test_{characters}chars_n_p_s.json"

    if os.path.exists(main_file_name):
        print(f"File {main_file_name} already exists.")
        continue
    for ngram in range(0, GRID_GRANULARITY + 1):
        for prefix in range(0, GRID_GRANULARITY + 1 - ngram):
            semantic = GRID_GRANULARITY - ngram - prefix

            sub_file_name = (
                f"{FOLDER}/w_{characters}_n{ngram}_p{prefix}_s{semantic}.json"
            )

            if os.path.exists(sub_file_name):
                continue

            retrievers_list = []
            if ngram > 0:
                retrievers_list.append(NgramRetrieverSpec(weight=ngram))
            if prefix > 0:
                retrievers_list.append(PrefixRetrieverSpec(weight=prefix))
            if semantic > 0:
                retrievers_list.append(SemanticRetrieverSpec(weight=semantic))

            suggesters_three = {
                "ngram, prefix and semantic": build_lookup_suggester(
                    sayt_corpus,
                    retrievers=retrievers_list,
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

remove_files = False

for character_file in NUM_CHARACTERS_LIST:
    master_dict = {}
    files_to_delete = []
    final_file_name = f"weight_test_{character_file}chars_n_p_s.json"
    main_file_name = f"{FOLDER}/{final_file_name}"

    if os.path.exists(main_file_name):
        print("Final file already exists.")
    else:
        for filename in sorted(os.listdir(FOLDER)):
            if filename.startswith(f"w_{character_file}_n") and filename.endswith(
                ".json"
            ):
                full_path = os.path.join(FOLDER, filename)
                key_name = filename[:-5]  # remove .json from the file name
                test_name = f"test{key_name[3:]}"
                with open(full_path, encoding="utf-8") as f:
                    master_dict[test_name] = json.load(f)
                files_to_delete.append(full_path)
        # Save locally
        with open(
            os.path.join(FOLDER, final_file_name),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(master_dict, f, indent=4)

        print(f"File {final_file_name} saved.")
        # Save to the bucket
        blob = client.bucket(bucket_name).blob(blob_name + final_file_name)
        blob.upload_from_string(
            json.dumps(master_dict, indent=4), content_type="application/json"
        )

    # remove files
    if remove_files:
        for file_path in files_to_delete:
            os.remove(file_path)
        print("Source files removed.")
    else:
        print("Source files not removed.")
