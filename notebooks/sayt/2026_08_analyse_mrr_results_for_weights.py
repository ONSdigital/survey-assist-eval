"""Find best performing MRR and corresponding test."""

# pylint: disable=C0103

# %%
import json
import os

from dotenv import load_dotenv
from google.cloud import storage as gcs

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")
client = gcs.Client()
blob_name = "evaluation-pipeline/SAYT/weights_by_character/"
bucket = client.bucket(bucket_name)


# %%
def read_json_from_gcs(blob_name_path: str, file_name_weight: str) -> dict:
    """Read JSON data from GCS.

    Args:
        blob_name_path (str): The path to the blob in Goggle Cloud Storage.
        file_name_weight (str): The name of the JSON file to read.

    Returns:
        dict: The JSON data as a dictionary.
    """
    blob = bucket.blob(blob_name_path + file_name_weight)
    json_data = blob.download_as_text()
    return json.loads(json_data)


# %%
def find_best_performing_setup(data: dict):
    """Finds best performing setup measured by MRR.

    Args:
        data (dict): A dictionary containing the test results with MRR scores.

    Returns:
        max_score (float): the highest MRR score achieved.
        best_dict (dict): a dictionary with those entries that achieved highest MRR.
    """
    # find best score and those tests that achieved that score
    max_score = max(d["MRR"] for d in data.values())
    best_dics = {k: v for k, v in data.items() if v["MRR"] == max_score}
    return max_score, best_dics


# %%
def get_ranked_setups(data: dict):
    """Get all tests ordered descending by MRR score.

    Args:
        data (dict): A dictionary containing the test results with MRR scores.

    Return:
        dict: A dictionary of tests, ordered by their MRR scores.
    """
    # Sort by MRR descending
    sorted_items = sorted(data.items(), key=lambda x: x[1]["MRR"], reverse=True)

    rankings = {}

    for key, value in sorted_items:
        score = value["MRR"]
        rankings.setdefault(score, {})[key] = value

    return rankings


# %%
for i in range(4, 10):
    file_name = f"weight_{i}_test_n_p_s.json"

    if bucket_name:
        data_file = read_json_from_gcs(blob_name, file_name)

    else:
        weights_file = f"notebooks/sayt/weights_sum_10/{file_name}"

        with open(weights_file, encoding="utf-8") as f:
            data_file = json.load(f)

    mrr_score, best_dict = find_best_performing_setup(data_file)
    print(f"Best MRR for {i} characters: {mrr_score}")
    print(f"Best setup for {i} characters: {best_dict.keys()}\n")

# %%
weight = 5

file_name = f"weight_{weight}_test_n_p_s.json"
if bucket_name:
    data_file = read_json_from_gcs(blob_name, file_name)

else:
    weights_file = f"notebooks/sayt/weights_sum_10/{file_name}"

    with open(weights_file, encoding="utf-8") as f:
        data_file = json.load(f)

rankings_by_weight = get_ranked_setups(data_file)

for rank, (individual_score, setups) in enumerate(rankings_by_weight.items(), start=1):
    print(f"Rank {rank}: MRR={individual_score}")
    print(f"  {list(setups.keys())}\n")
    if rank == 5:  # noqa: PLR2004
        break
