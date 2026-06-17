from collections.abc import Callable
from copy import deepcopy

import pandas as pd
import plotly.graph_objects as go
from plotly.express import colors
from plotly.subplots import make_subplots


def build_histogram(  # noqa
    df: pd.DataFrame,
    x_col: str,
    nbins: int = 20,
    name: str | None = None,
    title: str | None = None,
    xtitle: str | None = None,
    ytitle: str = "Count",
    color: str | None = None,
    opacity: float = 0.75,
    showlegend: bool = False,
    template: str | None = None,
) -> go.Figure:
    """Return a simple Plotly histogram figure.

    Args:
        df: The DataFrame containing the data to histogram.
        x_col: The column name in df to histogram.
        nbins: Number of histogram bins for numeric data.
        title: Optional figure title.
        xtitle: Optional x-axis title.
        ytitle: Optional y-axis title.
        color: Optional bar color.
        opacity: Histogram trace opacity.
        showlegend: Whether to show the trace legend.
        template: Optional Plotly layout template name.

    Returns:
        A Plotly Figure containing a single histogram trace.
    """
    x = df[x_col].dropna()

    figure = go.Figure(
        data=go.Histogram(
            x=x,
            nbinsx=nbins,
            name=name,
            marker_color=color,
            opacity=opacity,
            showlegend=showlegend,
            autobinx=not pd.api.types.is_numeric_dtype(x),
        )
    )
    figure.update_layout(
        title_text=title,
        xaxis_title=xtitle or "",
        yaxis_title=ytitle,
        template=template,
        margin={"t": 60, "b": 60, "l": 60, "r": 60},
    )
    return figure


def make_colour_map(
    filter_values: list[str], color_palette: list[str] | None = None
) -> dict[str, str]:
    """Create a color map for the given filter values.

    Args:
        filter_values: List of unique filter values.
        color_palette: Optional list of colors to use. If None, a default palette is used.

    Returns:
        A dictionary mapping filter values to colors.
    """
    if color_palette is None:
        color_palette = colors.qualitative.Light24

    return dict(zip(filter_values, color_palette[: len(filter_values)], strict=False))


def build_filterable_plot(  # noqa
    input_df: pd.DataFrame,
    group_col: str,
    default_group: str,
    figure_builder: Callable,
    groups_order: list[str] | None = None,
    include_default_group: bool = False,
    **kwargs,
) -> go.Figure:
    """Create a figure with an always-visible selector for grouping values.

    Args:
        input_df: The input DataFrame containing the data to plot.
        group_col: The column name in input_df to group by.
        default_group: The group value that should be visible by default.
        figure_builder: A callable that takes a DataFrame and returns a Plotly Figure.
        groups_order: Explicit display order for groups in the selector.
        include_default_group: Whether to include the default group in all other groups' plots.
        **kwargs: Additional keyword arguments to pass to the figure_builder.

    Returns:
        A Plotly Figure containing one trace per group and a dropdown selector to choose visibility.
    """
    group_labels = input_df[group_col].dropna().drop_duplicates().tolist()
    if not group_labels:
        msg = f"No groups found in column {group_col}."
        raise ValueError(msg)

    if default_group != "All" and default_group not in group_labels:
        default_group = group_labels[0]

    if groups_order is not None:
        ordered_groups = [default_group] + [
            g for g in groups_order if g in group_labels and g != default_group
        ]
    else:
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

    grouped_traces = []
    for group_label in ordered_groups:
        kwargs["name"] = str(group_label)
        if include_default_group and default_group == group_label == "All":
            default_kwargs = kwargs.copy()
            default_kwargs["color"] = "lightgray"
            group_fig = figure_builder(input_df, **default_kwargs)
        else:
            group_fig = figure_builder(
                input_df[
                    input_df[group_col].isin(
                        [group_label, default_group if include_default_group else None]
                    )
                ],
                **kwargs,
            )
        grouped_traces.extend(group_fig.data)

    selector_fig = go.Figure(grouped_traces)

    selector_fig.layout = group_fig.layout

    selector_fig.update_layout(
        legend_title=group_col,
    )

    if include_default_group:
        visible_all = [True] * len(grouped_traces)
        buttons = [
            {
                "label": default_group,
                "method": "update",
                "args": [{"visible": visible_all}],
            }
        ]
    else:
        buttons = []

    background_offset = 1 if include_default_group else 0
    for trace_index, group_value in enumerate(group_labels):
        visibility = [True] * background_offset + [False] * len(group_labels)
        visibility[background_offset + trace_index] = True
        buttons.append(
            {
                "label": str(group_value),
                "method": "update",
                "args": [
                    {"visible": visibility},
                ],
            }
        )

    selector_fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "active": 0,
                "x": 1.3,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
                "font": {"size": 10},
                "pad": {"t": 5, "r": 5, "b": 5, "l": 5},
            }
        ]
    )
    return selector_fig


def build_filterable_dashboard(  # noqa
    filterable_plots: list[go.Figure],
    default_group: str = "All",
) -> go.Figure:
    """Build a dashboard of subplots with shared legend and y-axis.

    Args:
        filterable_plots: A list of Plotly Figure objects, each representing a
            subplot in the dashboard.
        default_group: The legend group name that should be visible by default.
            If the group exists, its color is forced to light gray.

    Returns:
        A Plotly Figure containing a row of subplots with shared y-axis
        settings, x-axis titles copied from each input figure, and dropdown
        buttons to toggle visibility by legend group.
    """
    n_plots = len(filterable_plots)

    fig = make_subplots(
        rows=1,
        cols=n_plots,
        subplot_titles=[plot.layout.title["text"] for plot in filterable_plots],
        shared_yaxes=True,
    )

    legend_names = list(
        {
            trace.name
            for plot in filterable_plots
            for trace in plot.data
            if trace.name is not None
        }
    )
    group_colors = make_colour_map(legend_names)

    if default_group in legend_names:
        group_colors[default_group] = "lightgray"

    trace_register = {}

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

            color = group_colors.get(trace.name)
            if getattr(trace, "marker", None) is not None:
                trace.marker.color = color
            if getattr(trace, "line", None) is not None:
                trace.line.color = color

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

    if default_group in legend_names:
        visible_all = [True] * len(fig.data)
        buttons = [
            {
                "label": default_group,
                "method": "update",
                "args": [{"visible": visible_all}],
            }
        ]
        visibility_groups = [False] * len(fig.data)
        for idx in trace_register[default_group]:
            visibility_groups[idx] = True
    else:
        buttons = []
        visibility_groups = [False] * len(fig.data)

    for legend_name in legend_names:
        if legend_name == default_group:
            continue
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

    fig.update_layout(
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0, "y": 1.15}]
    )

    return fig
