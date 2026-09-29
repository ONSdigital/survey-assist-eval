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
def _metric_display_titles(score_metric: str, k: int | None = None) -> str:
    """Return a human-readable title for the score metric.

    Args:
        score_metric (str): Name of the score metric.
        k (int | None): Cutoff value for an @k metric, if applicable.

    Returns:
        str: The human-readable metric title.
    """
    if score_metric in {"precision_at_k", "recall_at_k"}:
        return f"{score_metric.split('_at_')[0].capitalize()}@{"k" if k is None else k}"

    if score_metric == "mrr":
        return "MRR (%)"

    return score_metric.replace("_", " ").capitalize()


def _pivot_weight_matrix(
    data: pd.DataFrame,
    value_col: str,
    weight_orders: tuple[list[str], list[str]],
    aggfunc: str = "first",
):
    """Pivot a weight-results dataframe into a matrix for plotting.

    Args:
        data (pd.DataFrame): Weight result rows.
        value_col (str): Column to place in the matrix.
        weight_orders (tuple[list[str], list[str]]): Desired n-gram and semantic weight order.
        aggfunc (str): Aggregation function used when pivoting.

    Returns:
        pd.DataFrame: The reshaped matrix aligned to the requested weight order.
    """
    ngram_weight_order, semantic_weight_order = weight_orders
    return data.pivot_table(
        index="Ngram_weight_label",
        columns="Semantic_weight_label",
        values=value_col,
        aggfunc=aggfunc,
    ).reindex(index=ngram_weight_order, columns=semantic_weight_order)


def _underline_max_min_labels(
    score_matrix: pd.DataFrame, label_matrix: pd.DataFrame, score_metric
):
    """Underline the best-performing cell(s) labels in a heatmap matrix.

    Args:
        score_matrix (pd.DataFrame): Matrix of metric scores.
        label_matrix (pd.DataFrame): Matrix of display text for each cell.
        score_metric (str): Metric name used to determine whether to take the max or min.

    Returns:
        pd.DataFrame: Label matrix with the winning cells underlined.
    """
    best_score = (
        score_matrix.min().min()
        if score_metric == "mean_rank"
        else score_matrix.max().max()
    )

    if pd.isna(best_score) or best_score == 0:
        return label_matrix

    max_min_cells = (score_matrix == best_score).stack()
    for ngram_weight, semantic_weight in max_min_cells[max_min_cells].index:
        label_matrix.loc[ngram_weight, semantic_weight] = (
            "<span style='text-decoration: underline; text-decoration-color: red;'>"
            f"{label_matrix.loc[ngram_weight, semantic_weight]}</span>"
        )
    return label_matrix


def _style_faceted_heatmap_axes(fig):
    """Apply axis titles and styling to a faceted heatmap.

    Args:
        fig: Plotly figure containing the faceted heatmaps.

    Returns:
        None: Updates the figure in place.
    """
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
    score_metric: str,
):
    """Build the score and label matrices for each character facet.

    Args:
        weight_results_df (pd.DataFrame): Prepared weight result data.
        character_order (list[str]): Character-count labels in facet order.
        weight_orders (tuple[list[str], list[str]]): N-gram and semantic weight orders.
        score_metric (str): Metric used for the heatmap values.

    Returns:
        dict: Matrices keyed by metric name for each character facet.
    """
    score_matrices = []
    label_matrices = []
    prefix_matrices = []
    mrr_matrices = []
    mean_rank_matrices = []
    precision_matrices = []
    recall_matrices = []

    for character in character_order:
        data = weight_results_df[weight_results_df["Characters"] == character]
        score_matrix = _pivot_weight_matrix(
            data,
            "metric_value",
            weight_orders,
        )
        label_matrix = _pivot_weight_matrix(
            data,
            "label_text",
            weight_orders,
        ).fillna("")

        score_matrices.append(score_matrix.to_numpy())
        label_matrices.append(
            _underline_max_min_labels(
                score_matrix, label_matrix, score_metric=score_metric
            ).to_numpy()
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

        mrr_matrices.append(
            _pivot_weight_matrix(
                data,
                "MRR_percent",
                weight_orders,
            )
            .map(lambda value: f"{value:.3f}" if pd.notna(value) else "")
            .to_numpy()
        )
        mean_rank_matrices.append(
            _pivot_weight_matrix(
                data,
                "mean_rank",
                weight_orders,
            )
            .map(lambda value: f"{value:.3f}" if pd.notna(value) else "")
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
    return {
        "score": score_matrices,
        "Label": label_matrices,
        "Prefix Weight": prefix_matrices,
        "mean_rank": mean_rank_matrices,
        "precision_at_k": precision_matrices,
        "recall_at_k": recall_matrices,
        "mrr": mrr_matrices,
    }


def _create_faceted_imshow(
    score_matrices: list[np.ndarray],
    semantic_weight_order: list[str],
    ngram_weight_order: list[str],
    facet_col_wrap: int,
    score_metric: str,
):
    """Create a faceted Plotly heatmap from score matrices.

    Args:
        score_matrices (list[np.ndarray]): Arrays of metric values for each facet.
        semantic_weight_order (list[str]): Semantic weight labels on the x-axis.
        ngram_weight_order (list[str]): N-gram weight labels on the y-axis.
        facet_col_wrap (int): Number of columns before wrapping facets.
        score_metric (str): Metric name used to pick the color scale.

    Returns:
        plotly.graph_objects.Figure: Faceted heatmap figure.
    """
    colour = "Blues_r" if score_metric == "mean_rank" else "Blues"

    return px.imshow(
        np.array(score_matrices),
        x=semantic_weight_order,
        y=ngram_weight_order,
        facet_col=0,
        facet_col_wrap=facet_col_wrap,
        color_continuous_scale=colour,
        text_auto=False,
        aspect="equal",
        origin="lower",
        labels={
            "x": "semantic",
            "y": "ngram",
            "facet_col": "Characters",
        },
    )


def _prepare_faceted_heatmap_data(
    character_weight_results: dict[int, dict], score_metric: str, k: int | None = None
):
    """Transform raw weight result data into a plotting dataframe.

    Args:
        character_weight_results (dict[int, dict]): Results keyed by character count.
        score_metric (str): Score metric to visualise.
        k (int | None): Rank cutoff used for precision and recall metrics.

    Returns:
        pd.DataFrame: Dataframe ready for facet heatmap plotting.
    """
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
    if score_metric == "mrr":
        weight_results_df = weight_results_df.assign(
            metric_value=weight_results_df["MRR_percent"],
        )
        weight_results_df = weight_results_df.assign(
            label_text=weight_results_df["metric_value"].map(
                lambda value: f"{value:.0f}" if value != 0 else ""
            ),
        )
    elif score_metric in ("precision_at_k", "recall_at_k"):

        weight_results_df = weight_results_df.assign(
            metric_value=weight_results_df[score_metric].apply(lambda x: x.get(str(k))),
        )
        weight_results_df = weight_results_df.assign(
            label_text=weight_results_df["metric_value"].map(
                lambda value: f"{value:.2f}" if value != 0 else ""
            ),
        )
    else:
        weight_results_df = weight_results_df.assign(
            metric_value=weight_results_df[score_metric],
        )
        weight_results_df = weight_results_df.assign(
            label_text=weight_results_df["metric_value"].map(
                lambda value: f"{value:.1f}" if value != 0 else ""
            ),
        )

    weight_results_df = weight_results_df.assign(
        Ngram_weight_label=weight_results_df["Ngram_weight"],
        Semantic_weight_label=weight_results_df["Semantic_weight"],
    )
    return weight_results_df


def _add_faceted_heatmap_text(
    fig,
    character_order,
    label_matrices,
    score_metric_title,
    metrics_matrices,
):
    """Attach hover text and labels to each heatmap facet.

    Args:
        fig: Plotly figure containing the heatmap traces.
        character_order: Facet labels by character count.
        label_matrices: Display labels for each heatmap cell.
        score_metric_title (str): Human-readable metric title.
        metrics_matrices: Additional metric matrices used for hover text.

    Returns:
        None: Updates the figure in place.
    """
    metric_names = list(metrics_matrices)
    for character, trace, labels, *metric_matrices in zip(
        character_order,
        fig.data,
        label_matrices,
        *metrics_matrices.values(),
        strict=True,
    ):
        hover_matrix = [
            [
                "".join(
                    f"{_metric_display_titles(metric)}: {value}<br>"
                    for metric, value in zip(metric_names, cell_values, strict=True)
                )
                for cell_values in zip(*rows, strict=True)
            ]
            for rows in zip(*metric_matrices, strict=True)
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
                "%{hovertext}"
                f"{score_metric_title}: " + "%{z:.3f}<extra></extra>"
            ),
        )


def _rename_facet_titles(fig, character_order):
    """Rename faceted heatmap titles to readable character-count labels.

    Args:
        fig: Plotly figure for the faceted heatmap.
        character_order: Ordered list of facet labels.

    Returns:
        None: Updates the figure annotations in place.
    """
    for annotation in fig.layout.annotations:
        if annotation.text.startswith("Characters="):
            character_index = int(annotation.text.removeprefix("Characters="))
            annotation.update(text=character_order[character_index])


def generate_faceted_heatmap(
    character_weight_results: dict[int, dict],
    score_metric: str = "mrr",
    k: int | None = None,
):
    """Generate faceted heatmaps of n/p/s weight combinations by character count.

    Args:
        character_weight_results (dict): Weight test results keyed by character count.
        score_metric (str): The metric used for assessing the performance.
        k (int | optional): rank k for recall and precision.

    Returns:
        fig: A Plotly figure object representing the faceted heatmaps.
    """
    weight_results_df = _prepare_faceted_heatmap_data(
        character_weight_results, score_metric=score_metric, k=k
    )

    ngram_weight_order = sorted(weight_results_df["Ngram_weight"].unique())
    semantic_weight_order = sorted(weight_results_df["Semantic_weight"].unique())
    character_order = [
        f"{character} chars" for character in sorted(character_weight_results)
    ]
    weight_orders = (ngram_weight_order, semantic_weight_order)
    facet_col_wrap = 3
    facet_rows = (len(character_order) + facet_col_wrap - 1) // facet_col_wrap

    matrices = _build_faceted_heatmap_matrices(
        weight_results_df, character_order, weight_orders, score_metric=score_metric
    )
    fig = _create_faceted_imshow(
        matrices["score"],
        semantic_weight_order,
        ngram_weight_order,
        facet_col_wrap,
        score_metric=score_metric,
    )

    excluded_hover_metrics = {"Label", "score"}
    if score_metric not in {"precision_at_k", "recall_at_k"}:
        excluded_hover_metrics.add(score_metric)

    hover_metrics_dict = {
        metric: result
        for metric, result in matrices.items()
        if metric not in excluded_hover_metrics
    }

    _add_faceted_heatmap_text(
        fig=fig,
        character_order=character_order,
        label_matrices=matrices["Label"],
        score_metric_title=_metric_display_titles(score_metric, k),
        metrics_matrices=hover_metrics_dict,
    )
    _rename_facet_titles(fig, character_order)

    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")

    fig.update_layout(
        title="Weight Configurations by Character Count",
        width=(360 * facet_col_wrap) + 180,
        height=(360 * facet_rows) + 160,
        margin={"l": 80, "r": 120, "t": 90, "b": 70},
        plot_bgcolor="white",
        coloraxis_colorbar={"title": _metric_display_titles(score_metric, k)},
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
score_metric_label = "precision_at_k"
k_value = 1

faceted_plot = generate_faceted_heatmap(
    character_weight_results=data_by_character,
    score_metric=score_metric_label,
    k=k_value,
)
if SAVE_PLOT:
    faceted_plot.write_html(
        f"data/sayt/{TEST_FOLDER}/heatmaps_by_character_{score_metric_label}.html"
    )
faceted_plot.show()


# %%
