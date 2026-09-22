"""Find best performing MRR and corresponding test."""

# pylint: disable=C0103

# %%
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

from src.survey_assist_eval.pipeline.shared_components import _read_json

# %%
TEST_FOLDER = "weights_grid_10_2k_sic_kb"
LOCAL_DIR = f"data/sayt/{TEST_FOLDER}/"
USE_BUCKET = True
SAVE_PLOT = True

os.makedirs(LOCAL_DIR, exist_ok=True)

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")
blob_name = f"evaluation-pipeline/SAYT/weights_by_character/{TEST_FOLDER}/"


# %%
def get_weight_by_char_dicts(
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
    max_score = max(d["mrr"] for d in data.values())
    best_dics = {k: v for k, v in data.items() if v["mrr"] == max_score}
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
    sorted_items = sorted(data.items(), key=lambda x: x[1]["mrr"], reverse=True)

    rankings = {}

    for key, value in sorted_items:
        score = value["mrr"]
        rankings.setdefault(score, {})[key] = value

    return rankings


# %%
def _pivot_weight_matrix(
    data: pd.DataFrame,
    value_col: str,
    weight_orders: tuple[list[str], list[str]],
    aggfunc: str = "first",
):
    ngram_weight_order, semantic_weight_order = weight_orders
    return data.pivot_table(
        index="Ngram_weight_label",
        columns="Semantic_weight_label",
        values=value_col,
        aggfunc=aggfunc,
    ).reindex(index=ngram_weight_order, columns=semantic_weight_order)


def _underline_max_labels(mrr_matrix: pd.DataFrame, label_matrix: pd.DataFrame):
    max_mrr = mrr_matrix.max().max()
    if pd.isna(max_mrr) or max_mrr == 0:
        return label_matrix

    max_cells = (mrr_matrix == max_mrr).stack()
    for ngram_weight, semantic_weight in max_cells[max_cells].index:
        label_matrix.loc[ngram_weight, semantic_weight] = (
            "<span style='text-decoration: underline; text-decoration-color: red;'>"
            f"{label_matrix.loc[ngram_weight, semantic_weight]}</span>"
        )
    return label_matrix


def _style_faceted_heatmap_axes(fig):
    xaxes = [axis for axis in fig.select_xaxes() if axis.anchor]
    yaxes = [axis for axis in fig.select_yaxes() if axis.anchor]
    yaxis_by_name = {axis.plotly_name.replace("axis", ""): axis for axis in yaxes}
    xaxis_by_name = {axis.plotly_name.replace("axis", ""): axis for axis in xaxes}
    bottom_domain = min(yaxis_by_name[axis.anchor].domain[0] for axis in xaxes)
    left_domain = min(xaxis_by_name[axis.anchor].domain[0] for axis in yaxes)

    for axis in xaxes:
        title = (
            "semantic"
            if yaxis_by_name[axis.anchor].domain[0] == bottom_domain
            else None
        )
        axis.update(
            title=title,
            tickangle=0,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        )
    for axis in yaxes:
        title = "ngram" if xaxis_by_name[axis.anchor].domain[0] == left_domain else None
        axis.update(
            title=title,
            scaleanchor=axis.anchor,
            scaleratio=1,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        )


def _build_faceted_heatmap_matrices(
    weight_results_df: pd.DataFrame,
    character_order: list[str],
    weight_orders: tuple[list[str], list[str]],
):
    mrr_matrices = []
    label_matrices = []
    prefix_matrices = []
    mean_rank_matrices = []
    precision_matrices = []
    recall_matrices = []

    for character in character_order:
        data = weight_results_df[weight_results_df["Characters"] == character]
        mrr_matrix = _pivot_weight_matrix(
            data,
            "MRR_percent",
            weight_orders,
        )
        label_matrix = _pivot_weight_matrix(
            data,
            "MRR_text",
            weight_orders,
        ).fillna("")

        mrr_matrices.append(mrr_matrix.to_numpy())
        label_matrices.append(
            _underline_max_labels(mrr_matrix, label_matrix).to_numpy()
        )
        prefix_matrices.append(
            _pivot_weight_matrix(
                data,
                "Prefix_weight",
                weight_orders,
            )
            .map(lambda value: f"{value:.1f}" if pd.notna(value) else "")
            .to_numpy()
        )
        mean_rank_matrices.append(
            _pivot_weight_matrix(
                data,
                "mean_rank",
                weight_orders,
            )
            .map(lambda value: f"{value:.1f}" if pd.notna(value) else "")
            .to_numpy()
        )
        precision_matrices.append(
            _pivot_weight_matrix(
                data,
                "precision_at_k",
                weight_orders,
            )
            .map(
                lambda d: (
                    {k: round(v, 2) if pd.notna(v) else "" for k, v in d.items()}
                    if isinstance(d, dict)
                    else np.nan
                )
            )
            .to_numpy()
        )
        recall_matrices.append(
            _pivot_weight_matrix(
                data,
                "recall_at_k",
                weight_orders,
            )
            .map(
                lambda d: (
                    {k: round(v, 2) if pd.notna(v) else "" for k, v in d.items()}
                    if isinstance(d, dict)
                    else np.nan
                )
            )
            .to_numpy()
        )

    matrices = {
        "MRR": mrr_matrices,
        "Label": label_matrices,
        "Prefix": prefix_matrices,
        "Mean_Rank": mean_rank_matrices,
        "Precision": prefix_matrices,
        "Recall": recall_matrices,
    }

    return matrices


def _create_faceted_imshow(
    mrr_matrices: list[np.ndarray],
    semantic_weight_order: list[str],
    ngram_weight_order: list[str],
    facet_col_wrap: int,
):
    return px.imshow(
        np.array(mrr_matrices),
        x=semantic_weight_order,
        y=ngram_weight_order,
        facet_col=0,
        facet_col_wrap=facet_col_wrap,
        color_continuous_scale="Blues",
        text_auto=False,
        aspect="equal",
        origin="lower",
        labels={
            "x": "semantic",
            "y": "ngram",
            "color": "MRR (%)",
            "facet_col": "Characters",
        },
    )


def _prepare_faceted_heatmap_data(character_weight_results: dict[int, dict]):
    weight_results_df = pd.concat(
        [
            pd.DataFrame.from_dict(data, orient="index").assign(
                Characters=f"{character} chars"
            )
            for character, data in character_weight_results.items()
        ],
        ignore_index=True,
    )
    weight_results_df = weight_results_df.assign(
        Ngram_weight=weight_results_df["Ngram_weight"] / 10,
        Semantic_weight=weight_results_df["Semantic_weight"] / 10,
        Prefix_weight=weight_results_df["Prefix_weight"] / 10,
        MRR_percent=weight_results_df["mrr"] * 100,
    )
    weight_results_df = weight_results_df.assign(
        Ngram_weight_label=weight_results_df["Ngram_weight"].map(
            lambda value: f"{value:.1f}"
        ),
        Semantic_weight_label=weight_results_df["Semantic_weight"].map(
            lambda value: f"{value:.1f}"
        ),
        MRR_text=weight_results_df["MRR_percent"].map(
            lambda value: f"{value:.0f}" if value != 0 else ""
        ),
    )
    return weight_results_df


def _add_faceted_heatmap_text(  # noqa: PLR0913, pylint: disable=R0913,R0917
    fig,
    character_order,
    label_matrices,
    prefix_matrices,
    mean_ranks_matrices,
    precision_matrices,
    recall_matrices,
):
    for (
        character,
        trace,
        labels,
        prefix_weights,
        mean_ranks,
        precisions,
        recalls,
    ) in zip(
        character_order,
        fig.data,
        label_matrices,
        prefix_matrices,
        mean_ranks_matrices,
        precision_matrices,
        recall_matrices,
        strict=True,
    ):
        hover_matrix = [
            [
                f"Prefix Weight: {pw}<br>Mean Rank: {mr}<br>Precision at: {pr}<br>Recall at: {re}"
                for pw, mr, pr, re in zip(row_pw, row_mr, row_pr, row_re, strict=False)
            ]
            for row_pw, row_mr, row_pr, row_re in zip(
                prefix_weights, mean_ranks, precisions, recalls, strict=False
            )
        ]
        trace.update(
            hovertext=hover_matrix,
            text=labels,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate=(
                f"Characters: {character}<br>"
                "Ngram Weight: %{y}<br>"
                "Semantic Weight: %{x}<br>"
                "%{hovertext}<br>"
                "MRR (%): %{z:.3f}<extra></extra>"
            ),
        )


def _rename_facet_titles(fig, character_order):
    for annotation in fig.layout.annotations:
        if annotation.text.startswith("Characters="):
            character_index = int(annotation.text.removeprefix("Characters="))
            annotation.update(text=character_order[character_index])


def generate_faceted_heatmap(character_weight_results: dict[int, dict]):
    """Generate faceted heatmaps of n/p/s weight combinations by character count.

    Args:
        character_weight_results (dict): Weight test results keyed by character count.

    Returns:
        fig: A Plotly figure object representing the faceted heatmaps.
    """
    weight_results_df = _prepare_faceted_heatmap_data(character_weight_results)

    ngram_weight_order = [
        f"{value:.1f}" for value in sorted(weight_results_df["Ngram_weight"].unique())
    ]
    semantic_weight_order = [
        f"{value:.1f}"
        for value in sorted(weight_results_df["Semantic_weight"].unique())
    ]
    character_order = [
        f"{character} chars" for character in sorted(character_weight_results)
    ]
    weight_orders = (ngram_weight_order, semantic_weight_order)
    facet_col_wrap = 3
    facet_rows = (len(character_order) + facet_col_wrap - 1) // facet_col_wrap

    matrices = _build_faceted_heatmap_matrices(
        weight_results_df,
        character_order,
        weight_orders,
    )
    fig = _create_faceted_imshow(
        matrices["MRR"],
        semantic_weight_order,
        ngram_weight_order,
        facet_col_wrap,
    )
    _add_faceted_heatmap_text(
        fig,
        character_order,
        matrices["Label"],
        matrices["Prefix"],
        matrices["Mean_Rank"],
        matrices["Precision"],
        matrices["Recall"],
    )
    _rename_facet_titles(fig, character_order)

    fig.update_layout(
        title="Weight Configurations by Character Count",
        width=(360 * facet_col_wrap) + 180,
        height=(360 * facet_rows) + 160,
        margin={"l": 80, "r": 120, "t": 90, "b": 70},
        plot_bgcolor="white",
        coloraxis_colorbar={"title": "MRR (%)"},
    )
    _style_faceted_heatmap_axes(fig)

    return fig


# %%
# Best performing setup for each character count
characters_list = list(range(4, 10))
for char in characters_list:
    data_weights = get_weight_by_char_dicts(
        characters=char,
        use_bucket=USE_BUCKET,
        bucket_path=f"gs://{bucket_name}/{blob_name}",
        local_path=LOCAL_DIR,
    )

    mrr_score, best_dict = find_best_performing_setup(data_weights)
    print(f"Best MRR for {char} characters: {mrr_score}")
    print(f"Best setup for {char} characters: {best_dict.keys()}\n")

# %%
# Top 5 performing setups for specific characters
char = 9

data_weights = get_weight_by_char_dicts(
    characters=char,
    use_bucket=USE_BUCKET,
    bucket_path=f"gs://{bucket_name}/{blob_name}",
    local_path=LOCAL_DIR,
)

rankings_by_weight = get_ranked_setups(data_weights)

for rank, (individual_score, setups) in enumerate(rankings_by_weight.items(), start=1):
    print(f"Rank {rank}: MRR={individual_score}")
    print(f"  {list(setups.keys())}\n")
    if rank == 5:  # noqa: PLR2004
        break
# %%
# create heatmaps for specific character
characters_list = list(range(4, 10))
data_by_character = {}
for char in characters_list:
    data_weights = get_weight_by_char_dicts(
        characters=char,
        use_bucket=USE_BUCKET,
        bucket_path=f"gs://{bucket_name}/{blob_name}",
        local_path=LOCAL_DIR,
    )
    data_by_character[char] = data_weights

# %%
faceted_plot = generate_faceted_heatmap(data_by_character)
if SAVE_PLOT:
    faceted_plot.write_html(f"data/sayt/{TEST_FOLDER}/heatmaps_by_character.html")
faceted_plot.show()


# %%
