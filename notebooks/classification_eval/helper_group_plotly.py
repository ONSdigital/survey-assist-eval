"""Helper function to create a grouped selector for Plotly figures."""

# pylint: disable=too-many-locals

from collections.abc import Callable
from copy import deepcopy

import pandas as pd
import plotly.graph_objects as go


def create_grouped_selector(
    input_df: pd.DataFrame,
    group_col: str,
    default_group: str,
    figure_builder: Callable,
    include_default_group: bool = False,
    **kwargs,
) -> go.Figure:
    """Create a figure with an always-visible selector for grouping values."""
    group_labels = input_df[group_col].dropna().drop_duplicates().tolist()
    if not group_labels:
        msg = f"No groups found in column {group_col}."
        raise ValueError(msg)

    if default_group not in group_labels:
        default_group = group_labels[0]

    ordered_groups = [
        default_group,
        *sorted(
            [
                group_label
                for group_label in group_labels
                if group_label != default_group
            ]
        ),
    ]

    grouped_figures = []
    for group_label in ordered_groups:
        group_fig = figure_builder(
            input_df[
                input_df[group_col].isin(
                    [group_label, default_group if include_default_group else None]
                )
            ],
            **kwargs,
        )
        grouped_figures.append(group_fig)

    selector_fig = go.Figure(grouped_figures[0])
    trace_counts = [len(grouped_fig.data) for grouped_fig in grouped_figures]
    selector_fig.update_traces(visible=True)

    for grouped_fig in grouped_figures[1:]:
        for trace in grouped_fig.data:
            trace_copy = deepcopy(trace)
            trace_copy.update(visible=False)
            selector_fig.add_trace(trace_copy)

    visibility_masks = []
    start_index = 0
    total_traces = sum(trace_counts)
    for trace_count in trace_counts:
        mask = [False] * total_traces
        for trace_index in range(start_index, start_index + trace_count):
            mask[trace_index] = True
        visibility_masks.append(mask)
        start_index += trace_count

    selector_fig.update_layout(
        title_text=grouped_figures[0].layout.title.text,
        font_size=10,
        height=600,
        width=1200,
        margin={"r": 180},
        annotations=[
            *grouped_figures[0].layout.annotations,
            {
                "x": 1.02,
                "y": 1.10,
                "xref": "paper",
                "yref": "paper",
                "text": "Section selection",
                "showarrow": False,
                "xanchor": "left",
                "yanchor": "bottom",
                "font": {"size": 12},
            },
        ],
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": group_label,
                        "method": "update",
                        "args": [
                            {"visible": visibility_masks[button_index]},
                            {
                                "title_text": grouped_figures[
                                    button_index
                                ].layout.title.text,
                            },
                        ],
                    }
                    for button_index, group_label in enumerate(ordered_groups)
                ],
                "direction": "down",
                "showactive": True,
                "active": 0,
                "x": 1.02,
                "xanchor": "left",
                "y": 1.08,
                "yanchor": "top",
            }
        ],
    )
    return selector_fig
