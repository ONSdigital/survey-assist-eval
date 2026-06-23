"""Standard plotting functions."""

from typing import Any

import pandas as pd
import plotly.graph_objects as go


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
        xaxis_title=layout.get("xtitle", x_col),
        yaxis_title=layout.get("ytitle", None),
        template=layout.get("template"),
        margin={"t": 60, "b": 60, "l": 60, "r": 60},
    )
    return figure
