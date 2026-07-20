"""Functions for detecting examples and leading wording in open questions."""

import pandas as pd
from pydantic import BaseModel

from survey_assist_eval.evaluation.open_questions.metric_utils import (
    add_metrics_columns,
)


class ExampleLeadingQuestionMetrics(BaseModel):
    """Container for all example and leading question evaluation metrics."""

    n_count: int
    pct_with_explicit_example_marker: float
    pct_with_examples: float
    pct_with_closed_category_option: float
    pct_with_closed_category_without_examples: float

    def report_metrics(self) -> str:
        """Pretty print the example and leading question evaluation metrics."""
        lines = [
            "\nExample and leading question metrics:",
            f" Number of follow-up questions: {self.n_count:.0f}",
            " Percentage with explicit example markers: "
            f"{self.pct_with_explicit_example_marker:.2f}%",
            f" Percentage with examples: {self.pct_with_examples:.2f}%",
            " Percentage with closed category options: "
            f"{self.pct_with_closed_category_option:.2f}%",
            " Percentage with closed category options without examples: "
            f"{self.pct_with_closed_category_without_examples:.2f}%",
        ]
        return "\n".join(lines)


def compute_example_and_leading_metrics(
    df: pd.DataFrame,
    *,
    text_column: str,
) -> ExampleLeadingQuestionMetrics:
    """Evaluate example and leading question quality for generated follow-up questions.

    Args:
        df: DataFrame containing generated questions.
        text_column: Column containing the text responses.

    Returns:
        ExampleLeadingQuestion Metrics: Structured summary of metrics.
    """
    df = add_example_and_leading_columns(df, text_column=text_column, prefix="eval_")

    metrics = summarise_example_and_leading_columns(df, prefix="eval_")

    return metrics


def _normalise_text(text: object) -> str | None:
    """Normalise text for matching, returning None for unsupported inputs."""
    if not isinstance(text, str):
        return None

    normalised = text.strip().lower()
    return normalised or None


def has_explicit_example_marker(text: str) -> bool:
    """Check whether text contains explicit example markers.

    Args:
        text: Question text to evaluate.

    Returns:
        True if explicit markers such as "for example" are present,
        otherwise False.
    """
    markers = [
        "for example",
        "e.g.",
        "e.g",
        " eg ",
        " eg. ",
        "i.e.",
        "i.e",
        " ie ",
        " ie. ",
        "such as",
        "for instance",
    ]

    normalised_text = _normalise_text(text)
    if normalised_text is None:
        return False

    return any(marker in normalised_text for marker in markers)


def has_example_introduction_phrase(text: str) -> bool:
    """Check whether text contains phrases that introduce examples,
    categories, or included items.

    Args:
        text: Question text to evaluate.

    Returns:
        True if example-introduction phrases such as "including" or
        "among other things" are present, otherwise False.
    """
    phrases = [
        "including",
        "includes",
        "inclusive of",
        "among other things",
        "in particular",
    ]

    normalised_text = _normalise_text(text)
    if normalised_text is None:
        return False

    return any(phrase in normalised_text for phrase in phrases)


def has_definition_example_wording(text: str) -> bool:
    """Check whether text contains definition-style example wording.

    Args:
        text: Question text to evaluate.

    Returns:
        True if wording such as "which means" or "namely" is present,
        otherwise False.
    """
    phrases = [
        "meaning",
        "which means",
        "namely",
        "that is",
    ]

    normalised_text = _normalise_text(text)
    if normalised_text is None:
        return False

    return any(phrase in normalised_text for phrase in phrases)


def has_examples(text: str) -> bool:
    """Check whether a question contains any example-style wording.

    Args:
        text: Question text to evaluate.

    Returns:
        True if any example detector matches, otherwise False.
    """
    return any(
        [
            has_explicit_example_marker(text),
            has_example_introduction_phrase(text),
            has_definition_example_wording(text),
        ]
    )


def has_closed_category_options(text: str) -> bool:
    """Check whether text provides predefined response categories.

    Args:
        text: Question text to evaluate.

    Returns:
        True if closed-category wording is detected,
        otherwise False.
    """
    normalised_text = _normalise_text(text)
    if normalised_text is None:
        return False

    return any(
        [
            " or " in normalised_text,
            ":" in normalised_text
            and (" or " in normalised_text or "," in normalised_text),
            "/" in normalised_text,
            "which of the following" in normalised_text,
            "which of these" in normalised_text,
            "select one" in normalised_text,
            "select the one" in normalised_text,
            "choose one" in normalised_text,
            "choose the one" in normalised_text,
            "pick one" in normalised_text,
            "pick the one" in normalised_text,
        ]
    )


def has_closed_category_without_examples(text: str) -> bool:
    """Check whether text provides closed-category options without examples.

    Args:
        text: Question text to evaluate.

    Returns:
        True if closed-category wording is detected and no examples are present,
        otherwise False.
    """
    return has_closed_category_options(text) and not has_examples(text)


def get_example_and_leading_metrics(text: str) -> dict[str, int | float | list[int]]:
    """Return example and leading question metrics for one string.

    Args:
        text: Text to analyse.

    Returns:
        A dict containing simple language metrics.
    """
    return {
        "has_explicit_example_marker": has_explicit_example_marker(text),
        "has_examples": has_examples(text),
        "has_closed_category_option": has_closed_category_options(text),
        "has_closed_category_without_examples": has_closed_category_without_examples(
            text
        ),
    }


def add_example_and_leading_columns(
    df: pd.DataFrame,
    text_column: str,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add example and leading question metric columns derived from a text column.

    Args:
        df: DataFrame with text data.
        text_column: Column containing text.
        prefix: Prefix for new columns. Defaults to "{text_column}_".

    Returns:
        DataFrame with added example and leading question metric columns.
    """
    return add_metrics_columns(
        df,
        text_column=text_column,
        metrics_func=get_example_and_leading_metrics,
        prefix=prefix,
    )


def summarise_example_and_leading_columns(
    df: pd.DataFrame,
    *,
    prefix: str,
) -> ExampleLeadingQuestionMetrics:
    """Summarise precomputed example and leading question metric columns
    into a structured summary.

    Args:
        df: DataFrame containing precomputed example and leading question metric columns.
        prefix: Prefix used for the metric columns
            (e.g. "<prefix>has_examples").

    Returns:
        ExampleLeadingQuestionMetrics: Structured summary of metrics.

    """
    has_explicit_example_marker_col = df[f"{prefix}has_explicit_example_marker"]
    has_examples_col = df[f"{prefix}has_examples"]
    has_closed_category_option_col = df[f"{prefix}has_closed_category_option"]
    has_closed_category_without_examples_col = df[
        f"{prefix}has_closed_category_without_examples"
    ]

    summary = {
        "n_count": len(df),
        "pct_with_explicit_example_marker": has_explicit_example_marker_col.mean()
        * 100,
        "pct_with_examples": has_examples_col.mean() * 100,
        "pct_with_closed_category_option": has_closed_category_option_col.mean() * 100,
        "pct_with_closed_category_without_examples": has_closed_category_without_examples_col.mean()
        * 100,
    }

    return ExampleLeadingQuestionMetrics(**summary)
