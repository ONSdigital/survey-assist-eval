"""Helper function to create a grouped selector for Plotly figures."""

from typing import Callable

import pandas as pd
import plotly.graph_objects as go


def create_grouped_selector(
    input_df: pd.DataFrame,
    group_col: str,
    default_group: str,
    figure_builder: Callable,
    **kwargs,
) -> tuple[go.Figure, str]:
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
            input_df[input_df[group_col] == group_label], **kwargs
        )
        grouped_figures.append(group_fig)

    selector_fig = go.Figure()
    for index, grouped_fig in enumerate(grouped_figures):
        trace = grouped_fig.data[0]
        trace.visible = index == 0
        selector_fig.add_trace(trace)

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
                            {
                                "visible": [
                                    current_index == button_index
                                    for current_index in range(len(ordered_groups))
                                ]
                            },
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
    return selector_fig, default_group
