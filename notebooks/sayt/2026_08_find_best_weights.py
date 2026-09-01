# %%
"""Find best performing MRR and corresponding test."""
# pylint: disable=C0103

# %%
import json
import os

import gcsfs
from dotenv import load_dotenv

# %%


# %%
def find_best_performing_setup(path: str):
    """Finds best performing setup measured by MRR.

    Args:
        path (str): path to the json file to be tested. Needs to contain MRR (Mean Reciprocal Rank).

    Returns:
        max_score (float): the highest MRR score achieved.
        best_dict (dict): a dictionary with those entries that achieved highest MRR.
    """
    with open(path, encoding="utf-8") as f:
        file = json.load(f)

    # find best score and those tests that achieved that score
    max_score = max(d["MRR"] for d in file.values())
    best_dics = {k: v for k, v in file.items() if v["MRR"] == max_score}
    return max_score, best_dics


# %%
# weights_5 = "notebooks/sayt/weights/weight_5_test_n_p_s.json"
# weights_6 = "notebooks/sayt/weights/weight_6_test_n_p_s.json"
# weights_7 = "notebooks/sayt/weights/weight_7_test_n_p_s.json"
weights_8 = "notebooks/sayt/weights/weight_8_test_n_p_s.json"
weights_9 = "notebooks/sayt/weights/weight_9_test_n_p_s.json"

# %%
# ms5, bd5 = find_best_performing_setup(weights_5)
# ms6, bd6 = find_best_performing_setup(weights_6)
# ms7, bd7 = find_best_performing_setup(weights_7)
ms8, bd8 = find_best_performing_setup(weights_8)
ms9, bd9 = find_best_performing_setup(weights_9)

# %%
# print("5 characters:", ms5, "\n", bd5.keys())
# print("6 characters:", ms6, "\n", bd6.keys())
# print("7 characters:", ms7, "\n", bd7.keys())
print("8 characters:", ms8, "\n", bd8.keys())
print("9 characters:", ms9, "\n", bd9.keys())

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")


# %%
characters = [8, 9]

# %%
fs = gcsfs.GCSFileSystem()

# %%
for character in characters:
    save_path = f"gs://{bucket_name}/evaluation-pipeline/SAYT/weights_by_character/weight_{character}_test_n_p_s.json"

    with open(
        f"notebooks/sayt/weights/weight_{character}_test_n_p_s.json", encoding="utf-8"
    ) as f:
        file = json.load(f)

    with fs.open(save_path, "w", encoding="utf-8") as f:
        json.dump(file, f, ensure_ascii=False, indent=4)

    print(f"Saved dictionary to: {save_path}")
