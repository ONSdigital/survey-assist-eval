"""Check the interactions between weights used for different suggesters."""

# pylint: disable=R0801, C0103, C0301

# %%
import json
import os

import numpy as np
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
    get_suggestions_by_chars,
)
from src.survey_assist_eval.pipeline.shared_components import _write_json
from survey_assist_eval.evaluation.sayt.performance_metrics_functions import (
    compute_performance_metrics_from_suggestions,
)

# %%
SIC_CODE_LENGTH = 5
MAX_SUGGESTIONS = 9
CORRECT_CODE_COL = "correct_sic_code"
SUGGESTERS_NAME = "ngram_prefix_semantic"
NUM_CHARACTERS_LIST = list(range(4, 10))
HARD_LIMIT = False
USE_2K = True

GRID_GRANULARITY = 10
OUTPUT_DIR = "data/sayt/"
FOLDER_PREFIX = f"weights_grid_{GRID_GRANULARITY}"
KB = ""
DF_SIZE = ""

# %%
load_dotenv()
BUCKET_NAME = os.getenv("EVALUATION_BUCKET_NAME")
if not BUCKET_NAME:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")


logger = get_logger(__name__)
logger.info("Location specs", BUCKET_NAME=BUCKET_NAME, output_dir=OUTPUT_DIR)

client = gcs.Client()

# %%
# access data for evaluation
# use either 100 or 2k dataset
if USE_2K:
    DF_SIZE = "_2k"
    test_df = pd.read_parquet(
        f"gs://{BUCKET_NAME}/evaluation-pipeline/original_datasets/sic_2k/sic_2k_test_data.parquet"
    )

    is_self_employed = test_df["sic2007_employee"] == "-9"

    test_df["full_entry"] = np.where(
        is_self_employed,
        test_df["sic2007_self_employed"],
        test_df["sic2007_employee"],
    )
    test_df["employment_status"] = np.where(
        is_self_employed, "self_employed", "employed"
    )

    test_df = test_df.rename(columns={"clerical_codes": CORRECT_CODE_COL})

else:
    DF_SIZE = "_100"
    test_df = pd.read_excel(
        f"gs://{BUCKET_NAME}/evaluation-pipeline/SAYT/SAYT matching.xlsx",
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
# LOOKUP_FILE_NAME = f"gs://{BUCKET_NAME}/evaluation-pipeline/SAYT/Lookup_IT3_Final.csv"
LOOKUP_FILE_NAME = f"gs://{BUCKET_NAME}/sic_knowledgebase/sic_kb_for_sayt.csv"

sayt_df = pd.read_csv(LOOKUP_FILE_NAME, dtype=str)
if LOOKUP_FILE_NAME.endswith("sic_kb_for_sayt.csv"):
    KB = "_sic_kb"
    display_text_col = "display_text"

elif LOOKUP_FILE_NAME.endswith("Lookup_IT3_Final.csv"):
    KB = "_lookup_it3"
    sayt_df["code"] = sayt_df["SIC07"].apply(
        lambda x: x if len(x) == SIC_CODE_LENGTH else f"0{x}"
    )
    sayt_df = sayt_df.rename(columns={"SIC_lookup": "search_text"})
    display_text_col = "search_text"
else:
    raise ValueError(
        f"LOOKUP_FILE_NAME {LOOKUP_FILE_NAME} does not match expected file names."
    )

sayt_corpus = build_sayt_corpus_from_df(
    df=sayt_df,
    search_text_col="search_text",
    display_text_col=display_text_col,
    code_col="code",
)[1]

SAVE_FOLDER = FOLDER_PREFIX + DF_SIZE + KB
BLOB_NAME = f"evaluation-pipeline/SAYT/weights_by_character/{SAVE_FOLDER}/"


# %%
if not os.path.exists(OUTPUT_DIR + SAVE_FOLDER):
    os.makedirs(OUTPUT_DIR + SAVE_FOLDER)
    print(f"Created folder: {OUTPUT_DIR + SAVE_FOLDER}")
# %%

characters_to_run = NUM_CHARACTERS_LIST.copy()
for characters in NUM_CHARACTERS_LIST.copy():

    main_file_name = (
        f"{OUTPUT_DIR}{SAVE_FOLDER}/weight_test_{characters}chars_n_p_s.json"
    )

    if os.path.exists(main_file_name):
        print(
            f"File {main_file_name} already exists, no need to run for {characters} characters."
        )
        characters_to_run.remove(characters)

for ngram in range(0, GRID_GRANULARITY + 1):
    for prefix in range(0, GRID_GRANULARITY + 1 - ngram):
        semantic = GRID_GRANULARITY - ngram - prefix

        characters_to_run2 = characters_to_run.copy()
        for characters in characters_to_run2.copy():

            sub_file_name = f"{OUTPUT_DIR}{SAVE_FOLDER}/w_{characters}_n{ngram}_p{prefix}_s{semantic}.json"
            if os.path.exists(sub_file_name):
                print(
                    f"File already exists, no need to run for {characters} characters."
                )
                characters_to_run2.remove(characters)

        if characters_to_run2 == []:
            print(
                f"File for ngram={ngram}, prefix={prefix}, semantic={semantic} already exists."
            )
            continue

        retrievers_list = []
        if ngram > 0:
            retrievers_list.append(NgramRetrieverSpec(weight=ngram))
        if prefix > 0:
            retrievers_list.append(PrefixRetrieverSpec(weight=prefix))
        if semantic > 0:
            retrievers_list.append(SemanticRetrieverSpec(weight=semantic))

        suggesters_three = {
            SUGGESTERS_NAME: build_lookup_suggester(
                sayt_corpus,
                retrievers=retrievers_list,
            ),
        }

        for characters in characters_to_run2:
            print(
                f"""Running evaluation for {characters} characters,
with ngram={ngram}, prefix={prefix}, semantic={semantic}."""
            )
            sub_file_name = f"{OUTPUT_DIR}{SAVE_FOLDER}/w_{characters}_n{ngram}_p{prefix}_s{semantic}.json"

            suggestions_df, avg_ms_dict = get_suggestions_by_chars(
                df=test_df,
                suggesters_dict=suggesters_three,
                correct_codes_col=CORRECT_CODE_COL,
                num_chars=[characters],
                suggestions_limit=MAX_SUGGESTIONS,
                hard_suggestions_limit=HARD_LIMIT,
            )

            suggestions_col_to_compare = (
                f"suggestions_{characters}chars_{SUGGESTERS_NAME}"
            )

            compare_performance_metrics = compute_performance_metrics_from_suggestions(
                df=suggestions_df,
                correct_codes_col=CORRECT_CODE_COL,
                suggestions_col=suggestions_col_to_compare,
                ave_time_per_query=avg_ms_dict[suggestions_col_to_compare],
                k_values=[characters],
            )

            data = {
                "Ngram_weight": ngram,
                "Prefix_weight": prefix,
                "Semantic_weight": semantic,
                "MRR": compare_performance_metrics.mrr,
                "avg_time": compare_performance_metrics.ave_time_per_query_ms,
                "mean_rank": compare_performance_metrics.mean_rank,
                "precision": compare_performance_metrics.precision_at_k[characters],
                "recall": compare_performance_metrics.recall_at_k[characters],
            }
            print(data)

            with open(sub_file_name, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

# %%
# combine separate test results into one file
remove_files = False  # set to True to remove the individual test files after combining
save_to_bucket = True  # set to True to save the combined file to the GCS bucket

for character_file in NUM_CHARACTERS_LIST:
    master_dict = {}
    files_to_delete = []
    final_file_name = f"weight_test_{character_file}chars_n_p_s.json"
    main_file_name = f"{OUTPUT_DIR}{SAVE_FOLDER}/{final_file_name}"

    if os.path.exists(main_file_name):
        print("Final file already exists.")
    else:
        for filename in sorted(os.listdir(OUTPUT_DIR + SAVE_FOLDER)):
            if filename.startswith(f"w_{character_file}_n") and filename.endswith(
                ".json"
            ):
                full_path = os.path.join(OUTPUT_DIR + SAVE_FOLDER, filename)
                key_name = filename[:-5]  # remove .json from the file name
                test_name = key_name.lstrip(f"w_{character_file}")
                with open(full_path, encoding="utf-8") as f:
                    master_dict[test_name] = json.load(f)
                files_to_delete.append(full_path)
        # Save locally
        with open(
            os.path.join(OUTPUT_DIR + SAVE_FOLDER, final_file_name),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(master_dict, f, indent=4)

        print(f"File {final_file_name} saved.")

        # Save to the bucket
        if save_to_bucket:
            bucket_path = "gs://" + BUCKET_NAME + "/" + BLOB_NAME + final_file_name
            _write_json(master_dict, bucket_path)

        # remove files
        if remove_files:
            for file_path in files_to_delete:
                os.remove(file_path)
            print("Source files removed.")
        else:
            print("Source files not removed.")

# %%
