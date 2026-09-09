"""Find best performing MRR and corresponding test."""

# pylint: disable=C0103

# %%
import json
import os

import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from src.survey_assist_eval.pipeline.shared_components import _read_json

# %%
TEST_FOLDER = "weights_grid_10_sic_kb"
LOCAL_DIR = f"data/sayt/{TEST_FOLDER}/"
USE_BUCKET = True

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")
BLOB_NAME = f"evaluation-pipeline/SAYT/weights_by_character/{TEST_FOLDER}/"


# %%
def get_data(
    characters: int,
    use_bucket: bool,
    bucket_path: str | None = None,
    local_path: str | None = None,
) -> dict:
    """Get data for visualisation of weight combinations.

    Args:
        characters (int): The number of characters to consider for the test.
        use_bucket (bool): Whether to read data from a cloud bucket or local file.
        bucket_path (str, optional): The path to the cloud bucket. Required if use_bucket==True.
        local_path (str, optional): The path to the local directory. Required if use_bucket==False.

    Returns:
        dict: A dictionary containing the test results with MRR scores.
    """
    file_name = f"weight_test_{characters}chars_n_p_s.json"
    if use_bucket:
        path = bucket_path + file_name
        data_file = _read_json(path)

    else:
        weights_file = f"{local_path}{file_name}"

        with open(weights_file, encoding="utf-8") as f:
            data_file = json.load(f)

    return data_file


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
def generate_ternary_plot(data: dict, character: int):
    """Generate a ternary plot of the n/p/s weight combinations.

    Args:
        data (dict): A dictionary containing the test results with MRR scores.
        character (int): The number of characters to consider for the test.

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
                "colorscale": "Jet",
                "showscale": True,
                "colorbar": {"title": "MRR score"},
                "reversescale": True,
            },
        )
    )

    fig.update_layout(
        title=f"Weight Configurations ({character} characters)",
        ternary={
            "sum": 1,
            "aaxis": {"title": ""},
            "baxis": {"title": ""},
            "caxis": {"title": ""},
        },
    )

    fig.add_annotation(
        text="ngram",
        x=0.25,
        y=0.5,
        xref="paper",
        yref="paper",
        textangle=-60,
        showarrow=False,
        font={"size": 14},
    )
    fig.add_annotation(
        text="prefix",
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        textangle=0,
        showarrow=False,
        font={"size": 14},
    )
    fig.add_annotation(
        text="semantic",
        x=0.75,
        y=0.5,
        xref="paper",
        yref="paper",
        textangle=60,
        showarrow=False,
        font={"size": 14},
    )

    # fig.show()
    return fig


# %%
# Best performing setup for each character count
for i in range(4, 10):
    data_weights = get_data(
        characters=i,
        use_bucket=USE_BUCKET,
        bucket_path=f"gs://{bucket_name}/{BLOB_NAME}",
        local_path=LOCAL_DIR,
    )

    mrr_score, best_dict = find_best_performing_setup(data_weights)
    print(f"Best MRR for {i} characters: {mrr_score}")
    print(f"Best setup for {i} characters: {best_dict.keys()}\n")

# %%
# Top 5 performing setups
char = 9

data_weights = get_data(
    characters=char,
    use_bucket=USE_BUCKET,
    bucket_path=f"gs://{bucket_name}/{BLOB_NAME}",
    local_path=LOCAL_DIR,
)

rankings_by_weight = get_ranked_setups(data_weights)

for rank, (individual_score, setups) in enumerate(rankings_by_weight.items(), start=1):
    print(f"Rank {rank}: MRR={individual_score}")
    print(f"  {list(setups.keys())}\n")
    if rank == 5:  # noqa: PLR2004
        break

# %%
# Access data for visualisation

characters_list = [6, 9]
for char in characters_list:
    data_weights = get_data(
        characters=char,
        use_bucket=USE_BUCKET,
        bucket_path=f"gs://{bucket_name}/{BLOB_NAME}",
        local_path=LOCAL_DIR,
    )
    plot = generate_ternary_plot(data_weights, char)
    plot.show()
    # plot.write_html(f"data/sayt/{TEST_FOLDER}/ternary_plot_{char}_chars.html")

# %%
colours = ["Blues"]
characters_list = [4]
for char in characters_list:
    for c in colours:
        data_weights = get_data(
            characters=char,
            use_bucket=USE_BUCKET,
            bucket_path=f"gs://{bucket_name}/{BLOB_NAME}",
            local_path=LOCAL_DIR,
        )
        df = pd.DataFrame.from_dict(data_weights, orient="index")
        heatmap_data = df.pivot_table(
            index="Ngram_weight", columns="Semantic_weight", values="MRR"
        )
        semantic_matrix = df.pivot_table(
            index="Ngram_weight", columns="Semantic_weight", values="Prefix_weight"
        )
        x_vals = [val / 10 for val in heatmap_data.columns]
        y_vals = [val / 10 for val in heatmap_data.index]
        c_scaled = [
            [val / 10 if pd.notna(val) else None for val in row]
            for row in semantic_matrix.values
        ]
        text_matrix = [
            [f"{val*100:.2f}" if pd.notna(val) and val != 0 else "" for val in row]
            for row in heatmap_data.values
        ]

        heat_fig = go.Figure(
            data=go.Heatmap(
                x=x_vals,
                y=y_vals,
                z=heatmap_data.values.tolist(),
                customdata=c_scaled,
                text=text_matrix,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorscale=c,
                colorbar={"title": "MRR"},
                hovertemplate=(
                    "Ngram Weight: %{y}<br>"
                    "Semantic Weight: %{x}<br>"
                    "Prefix Weight: %{customdata}<br>"
                    "MRR: %{z}<extra></extra>"
                ),
            )
        )
        heat_fig.update_layout(
            title=f"Weight Configurations ({char} characters, colour: {c})",
            xaxis_title="semantic",
            yaxis_title="ngram",
            plot_bgcolor="white",
        )
        heat_fig.show()
# %%
