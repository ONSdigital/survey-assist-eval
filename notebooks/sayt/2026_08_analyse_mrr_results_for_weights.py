# %%
"""Find best performing MRR and corresponding test."""

# pylint: disable=C0103

# %%
import json
import os

import plotly.graph_objects as go
from dotenv import load_dotenv

from src.survey_assist_eval.pipeline.shared_components import _read_json

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")
blob_name = "evaluation-pipeline/SAYT/weights_by_character/"


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
    """Get all tests orgered descending by MRR score.

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
def generate_ternary_plot(data: dict):
    """Generate a ternary plot of the n/p/s weight combinations.

    Args:
        data (dict): A dictionary containing the test results with MRR scores.

    Returns:
        fig: A Plotly figure object representing the ternary plot.
    """
    # Plot n/p/s weight combinations in a ternary diagram.
    points = []

    for setup_name, results in data.items():
        # get points
        points.append(
            {
                "setup": setup_name,
                "n": results["Ngram_weight"],
                "p": results["Prefix_weight"],
                "s": results["Semantic_weight"],
                "mrr": results["MRR"],
            }
        )

    if not points:
        raise ValueError("No records containing 'n', 'p', and 's' weights were found.")

    # ternary plot
    fig = go.Figure(
        go.Scatterternary(
            a=[point["n"] for point in points],
            b=[point["p"] for point in points],
            c=[point["s"] for point in points],
            mode="markers",
            text=[point["setup"] for point in points],
            customdata=[[point["mrr"]] for point in points],
            hovertemplate=(
                "Setup: %{text}<br>"
                "n: %{a}<br>"
                "p: %{b}<br>"
                "s: %{c}<br>"
                "MRR: %{customdata[0]}<extra></extra>"
            ),
            marker={
                "size": 10,
                "color": [point["mrr"] for point in points],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "MRR score"},
            },
        )
    )

    fig.update_layout(
        title="Weight Configurations",
        ternary={
            "sum": 1,
            "aaxis": {"title": "ngram"},
            "baxis": {"title": "prefix"},
            "caxis": {"title": "semantic"},
        },
    )

    # fig.show()
    return fig


# %%
for i in range(4, 10):

    file_name = f"weight_test_{i}chars_n_p_s.json"
    if bucket_name:
        print("read from storage")
        path = f"gs://{bucket_name}/{blob_name}{file_name}"
        data_file = _read_json(path)

    else:
        print("read from local file")
        weights_file = f"data/sayt/weights_grid_10_lookup_it3/{file_name}"

        with open(weights_file, encoding="utf-8") as f:
            data_file = json.load(f)

    mrr_score, best_dict = find_best_performing_setup(data_file)
    print(f"Best MRR for {i} characters: {mrr_score}")
    print(f"Best setup for {i} characters: {best_dict.keys()}\n")

# %%
character = 5

file_name = f"weight_test_{character}chars_n_p_s.json"
if bucket_name:
    print("read from storage")
    path = f"gs://{bucket_name}/{blob_name}{file_name}"
    data_file = _read_json(path)

else:
    print("read from local file")
    weights_file = f"data/sayt/weights_grid_10_lookup_it3/{file_name}"

    with open(weights_file, encoding="utf-8") as f:
        data_file = json.load(f)

rankings_by_weight = get_ranked_setups(data_file)

for rank, (individual_score, setups) in enumerate(rankings_by_weight.items(), start=1):
    print(f"Rank {rank}: MRR={individual_score}")
    print(f"  {list(setups.keys())}\n")
    if rank == 5:  # noqa: PLR2004
        break

# %%
# Access data for visualisation
character = 7

file_name = f"weight_test_{character}chars_n_p_s.json"
if bucket_name:
    print("read from storage")
    path = f"gs://{bucket_name}/{blob_name}{file_name}"
    data_file = _read_json(path)

else:
    print("read from local file")
    weights_file = f"data/sayt/weights_grid_10_lookup_it3/{file_name}"

    with open(weights_file, encoding="utf-8") as f:
        data_file = json.load(f)

# %%
generate_ternary_plot(data_file)

# %%
