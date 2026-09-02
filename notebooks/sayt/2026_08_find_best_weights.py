# %%
"""Find best performing MRR and corresponding test."""
# pylint: disable=C0103

# %%
import json


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
def get_ranked_setups(path):
    """Get all tests orgered descending by MRR score.

    Args:
        path (str): path to the json file to be tested. Needs to contain MRR (Mean Reciprocal Rank).

    Return:
        dict: A dictionary of tests, ordered by their MRR scores.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Sort by MRR descending
    sorted_items = sorted(data.items(), key=lambda x: x[1]["MRR"], reverse=True)

    rankings = {}

    for key, value in sorted_items:
        score = value["MRR"]
        rankings.setdefault(score, {})[key] = value

    return rankings


# %%
for i in range(4, 10):
    weights_file = f"notebooks/sayt/weights_sum_10/weight_{i}_test_n_p_s.json"
    mrr_score, best_dict = find_best_performing_setup(weights_file)
    print(f"Best MRR for {i} characters: {mrr_score}")
    print(f"Best setup for {i} characters: {best_dict.keys()}\n")


# %%
weight = 5

rankings_by_weight = get_ranked_setups(
    f"notebooks/sayt/weights_sum_10/weight_{weight}_test_n_p_s.json"
)

for rank, (individual_score, setups) in enumerate(rankings_by_weight.items(), start=1):
    print(f"Rank {rank}: MRR={individual_score}")
    print(f"  {list(setups.keys())}\n")
    if rank == 5:  # noqa: PLR2004
        break
