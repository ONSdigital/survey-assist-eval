"""Plotting Helper Functions used in Notebooks."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.express import colors
from plotly.subplots import make_subplots

# ruff: noqa: PLR0913
# pylint: disable=R0914,R0917,R0913


def build_histogram(
    df: pd.DataFrame,
    x_col: str,
    layout: dict[str, Any] | None = None,
    **hist_kwargs,
) -> go.Figure:
    """Return a simple Plotly histogram figure.

    Args:
        df: The DataFrame containing the data to histogram.
        x_col: The column name in df to histogram.
        layout: Dictionary of layout options (title, xtitle, ytitle, template).
        **hist_kwargs: Additional go.Histogram kwargs (nbins, name, color, opacity, showlegend).

    Returns:
        A Plotly Figure containing a single histogram trace.
    """
    layout = layout or {}
    x = df[x_col].dropna()

    figure = go.Figure(
        data=go.Histogram(
            x=x,
            autobinx=not pd.api.types.is_numeric_dtype(x),
            **hist_kwargs,
        )
    )
    figure.update_layout(
        title_text=layout.get("title"),
        xaxis_title=layout.get(f"xtitle (n = {len(df)})", f"{x_col} (n = {len(df)})"),
        yaxis_title=layout.get("ytitle", None),
        template=layout.get("template"),
        margin={"t": 60, "b": 60, "l": 60, "r": 60},
    )
    return figure


def make_colour_map(
    category_values: list[str],
    palette_name: str | None = None,
    palette_values: list[str] | None = None,
) -> dict[str, str]:
    """Uses a named palette (e.g. "ons") or a custom list of colours.
    Custom colours take priority if provided.

    Args:
        category_values: Ordered list of category labels.
        palette_name: Name of predefined palette to use. If None, use default.
        palette_values: Optional list of colours to override palette.

    Returns:
        Dictionary mapping each category to a colour.

    Raises:
        ValueError: If there are not enough colours for the categories.
    """
    if not category_values:
        raise ValueError("category_values must contain at least one value.")

    # Select colour palette
    if palette_values is not None:
        colours = palette_values
    elif palette_name == "ons":
        colours = [
            "#206095",
            "#A8BD3A",
            "#871A5B",
            "#F66068",
            "#05341A",
            "#27A0CC",
            "#003C57",
            "#22D0B6",
            "#746CB1",
        ]
    elif palette_name == "ons_plus":
        colours = [
            "#206095",
            "#A8BD3A",
            "#871A5B",
            "#F66068",
            "#05341A",
            "#27A0CC",
            "#003C57",
            "#22D0B6",
            "#746CB1",
            "#E69F00",
            "#D55E00",
            "#F0E442",
            "#CC79A7",
            "#4A4DD1",
            "#009E73",
            "#8C510A",
            "#BF812D",
            "#C51B7D",
        ]
    else:
        colours = colors.qualitative.Light24

    # Validate length
    if len(colours) < len(category_values):
        raise ValueError(
            f"Palette has {len(colours)} colours but "
            f"{len(category_values)} categories are required."
        )

    # Create mapping
    return dict(zip(category_values, colours, strict=False))


def get_trace_colour_map(fig: go.Figure) -> dict[str, str]:
    """Return a mapping of trace names to colours.

    Extracts colours from each trace in the figure, checking marker
    colour first and falling back to line colour if needed.

    Only includes traces with a valid name and a single colour value.
    """
    colour_map: dict[str, str] = {}

    for trace in fig.data:
        if trace.name is None or trace.name in colour_map:
            continue

        colour = None

        # Prefer marker colour if present
        if getattr(trace, "marker", None) is not None:
            colour = getattr(trace.marker, "color", None)

        # Fallback to line colour (for line plots)
        if colour is None and getattr(trace, "line", None) is not None:
            colour = getattr(trace.line, "color", None)

        if isinstance(colour, str):
            colour_map[trace.name] = colour

    return colour_map


def _apply_trace_colors(trace: Any, color: str | None) -> None:
    """Apply a colour to a trace.

    Sets the colour on marker and line attributes if they exist.
    """
    if getattr(trace, "marker", None) is not None:
        trace.marker.color = color
    if getattr(trace, "line", None) is not None:
        trace.line.color = color


def _get_group_colors(
    group_names: list[str], reference_group: str | None, palette: str | None = None
) -> dict[str, str]:
    """Create a colour map for groups.

    Uses the selected palette and sets the default group colour to grey.
    """
    colors_map = make_colour_map(group_names, palette_name=palette)
    if reference_group is not None and reference_group in colors_map:
        colors_map[reference_group] = "#A09FA0"
    return colors_map


def _build_visibility_buttons(
    group_labels: list[str],
    reference_group: str | None,
    trace_register: dict[str, list[int]],
    fig_data_len: int,
) -> list[dict[str, Any]]:
    """Create dropdown buttons to control trace visibility.

    Builds a list of Plotly dropdown buttons that toggle groups of traces
    based on their indices in `trace_register`.

    Args:
        group_labels: List of group names.
        reference_group: Group added to all selections.
        trace_register: Mapping of group names to trace indices.
        fig_data_len: Total number of traces in the figure.

    Returns:
        List of dropdown button definitions for Plotly.
    """
    buttons = [
        {
            "label": "All",
            "method": "update",
            "args": [{"visible": [True] * fig_data_len}],
        }
    ]

    visibility_groups = [False] * fig_data_len

    if reference_group is not None:
        for idx in trace_register[reference_group]:
            visibility_groups[idx] = True

    for legend_name in group_labels:
        visibility = visibility_groups.copy()

        for idx in trace_register[legend_name]:
            visibility[idx] = True

        buttons.append(
            {
                "label": str(legend_name),
                "method": "update",
                "args": [
                    {"visible": visibility},
                ],
            }
        )

    return buttons


def _setup_colour_map(
    ordered_groups: list[str],
    reference_group: str | None,
    group_colour_map: dict[str, str] | None,
    colour_palette: str | None = None,
) -> dict[str, str]:
    """Set up or validate the colour map for groups."""
    if group_colour_map is None:
        group_colour_map = _get_group_colors(
            ordered_groups, reference_group, palette=colour_palette
        )

    if reference_group and reference_group not in group_colour_map:
        group_colour_map[reference_group] = "#A09FA0"

    return group_colour_map


def build_filterable_plot(
    input_df: pd.DataFrame,
    group_col: str,
    figure_builder: Callable,
    reference_group: None | str = "Total",
    groups_order: list[str] | None = None,
    group_colour_map: dict[str, str] | None = None,
    colour_palette: str | None = None,
    **kwargs,
) -> go.Figure:
    """Create a filterable figure by grouping data and toggling traces.

    Builds a figure for each group using `figure_builder` and combines the
    traces into a single figure. A dropdown menu is added to show or hide
    groups by updating trace visibility.

    Args:
        input_df: Data to plot.
        group_col: Column defining groups.
        figure_builder: Function that creates a Plotly figure from a DataFrame.
        reference_group: Reference group to be included on all plots. If None,
            no reference group is shown.
        groups_order: Optional order for groups in the dropdown.
        group_colour_map: Optional mapping of group names to colours.
        colour_palette: Name of predefined palette to use. Only
            used if group_colour_map is None.
        **kwargs: Additional arguments passed to `figure_builder`.

    Returns:
        A Plotly figure with grouped traces and a dropdown selector to
        control visibility.
    """
    group_labels = input_df[group_col].dropna().drop_duplicates().tolist()
    if not group_labels:
        msg = f"No groups found in column {group_col}."
        raise ValueError(msg)

    if groups_order is not None:
        ordered_groups = [
            g for g in groups_order if g in group_labels and g != reference_group
        ]
    else:
        ordered_groups = [g for g in group_labels if g != reference_group]

    if reference_group is not None:
        ordered_groups = [reference_group, *ordered_groups]

    if "Total" not in ordered_groups:
        ordered_groups = ["Total", *ordered_groups]

    group_colour_map = _setup_colour_map(
        ordered_groups,
        reference_group,
        group_colour_map,
        colour_palette,
    )

    grouped_traces = []
    trace_register: dict[str, list[int]] = {}
    trace_idx = 0

    for group_label in ordered_groups:
        group_kwargs = kwargs.copy()
        group_kwargs["name"] = str(group_label)

        if group_label == "Total":
            group_fig = figure_builder(input_df, **group_kwargs)
        else:
            group_fig = figure_builder(
                input_df[input_df[group_col].isin([group_label])],
                **group_kwargs,
            )

        for trace in group_fig.data:
            _apply_trace_colors(trace, group_colour_map[group_label])
            if trace.name not in trace_register:
                trace_register[trace.name] = []
            trace_register[trace.name].append(trace_idx)
            trace_idx += 1

        grouped_traces.extend(group_fig.data)

    selector_fig = go.Figure(grouped_traces)

    selector_fig.layout = group_fig.layout

    selector_fig.update_layout(
        legend_title=group_col,
    )

    buttons = _build_visibility_buttons(
        ordered_groups, reference_group, trace_register, len(grouped_traces)
    )

    selector_fig.update_layout(
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1, "y": 1}]
    )
    return selector_fig


def build_filterable_dashboard(
    filterable_plots: list[go.Figure],
    reference_group: None | str = "Total",
    group_colour_map: dict[str, str] | None = None,
) -> go.Figure:
    """Combine filterable plots into a single dashboard with shared controls.

    Creates a row of subplots and groups traces by name so they can be
    shown or hidden together using a dropdown menu.

    Args:
        filterable_plots: List of Plotly figures to include.
        reference_group: Reference group to be included on all plots. If None,
            no reference group is shown.
        group_colour_map: Optional mapping of group names to colours.
        include_default_group: If True, include default_group in all plots.

    Returns:
        A Plotly figure with a dropdown selector controlling visibility
        of grouped traces across all subplots.
    """
    n_plots = len(filterable_plots)

    fig = make_subplots(
        rows=1,
        cols=n_plots,
        subplot_titles=[plot.layout.title["text"] for plot in filterable_plots],
        shared_yaxes=True,
    )

    group_labels = list(
        {
            trace.name
            for plot in filterable_plots
            for trace in plot.data
            if trace.name is not None
        }
    )

    group_colour_map = _setup_colour_map(
        group_labels, reference_group, group_colour_map
    )

    trace_register: dict[str, list[int]] = {}

    trace_idx = 0

    for subplot_col, plot in enumerate(filterable_plots, start=1):
        subplot_plot = deepcopy(plot)
        for trace in subplot_plot.data:
            trace.legendgroup = trace.name
            trace.showlegend = subplot_col == 1

            if trace.name not in trace_register:
                trace_register[trace.name] = []
            trace_register[trace.name].append(trace_idx)
            trace_idx += 1

            color = group_colour_map.get(trace.name)
            _apply_trace_colors(trace, color)

        fig.add_traces(subplot_plot.data, rows=1, cols=subplot_col)

        fig.update_xaxes(
            title_text=(
                plot.layout.xaxis.title.text if plot.layout.xaxis.title else None
            ),
            row=1,
            col=subplot_col,
        )

        if subplot_col == 1:
            fig.update_yaxes(
                title_text=(
                    plot.layout.yaxis.title.text if plot.layout.yaxis.title else None
                ),
                row=1,
                col=1,
            )

    buttons = _build_visibility_buttons(
        group_labels,
        reference_group,
        trace_register,
        len(fig.data),
    )

    fig.update_layout(
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1, "y": 1}]
    )

    return fig
