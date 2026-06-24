"""Plotting Helper Functions used in plot functions."""

import plotly.graph_objects as go
from plotly.express import colors


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
