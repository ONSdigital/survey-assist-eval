"""Tests for simple language functions."""

# pylint: disable=redefined-outer-name

import pandas as pd
import pytest
from textstat import textstat

from survey_assist_eval.evaluation.open_questions.simple_language_functions import (
    SimpleLanguageMetrics,
    add_simple_language_columns,
    compute_simple_language_metrics,
    extract_acronyms,
    get_avg_syllables_per_word,
    get_simple_language_metrics,
    get_syllable_count_per_word,
    summarise_simple_language_columns,
)

# ============================================================================
# Test Data - Shared between tests
# ============================================================================


NON_STRING_INPUTS = [
    pytest.param(None, id="none_input"),
    pytest.param(123, id="integer_input"),
    pytest.param(12.5, id="float_input"),
    pytest.param([], id="list_input"),
    pytest.param({}, id="dict_input"),
    pytest.param(True, id="bool_input"),
]

EMPTY_TEXT_INPUTS = [
    pytest.param("", id="empty_string"),
    pytest.param("   ", id="whitespace_only"),
]


@pytest.fixture
def simple_language_input_df():
    """Return test data for simple language wrapper function tests.

    Covers diverse text patterns for wrapper testing.
    """
    return pd.DataFrame(
        {
            "follow_up_question": [
                "Would you like to participate?",
                "ONS is a UK government agency.",
                "Does the NHS provide good service?",
                "",
                None,
            ],
            "respondent_id": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def expected_simple_language_df():
    """Return expected output after adding simple language columns."""
    return pd.DataFrame(
        {
            "follow_up_question": [
                "Would you like to participate?",
                "ONS is a UK government agency.",
                "Does the NHS provide good service?",
                "",
                None,
            ],
            "respondent_id": [1, 2, 3, 4, 5],
            "follow_up_question_n_acronyms": [0, 2, 1, 0, 0],
            "follow_up_question_avg_syllables_per_word": [
                1.6,
                1.66667,
                1.33333,
                0.0,
                0.0,
            ],
            "follow_up_question_syllable_counts": [
                [1, 1, 1, 1, 4],
                [1, 1, 1, 1, 3, 3],
                [1, 1, 1, 2, 1, 2],
                [],
                [],
            ],
        }
    )


# ============================================================================
# Test extract_acronyms function
# ============================================================================


def test_extract_acronyms_simple_acronyms_basic_and_digits():
    """Extracts uppercase acronyms and those with digits using the simple pattern."""
    text = "ONS, NLP and ISO9001 with G7 and DD10."
    assert extract_acronyms(text) == [
        "ONS",
        "NLP",
        "ISO9001",
        "G7",
        "DD10",
    ], "Expected basic acronyms and digit-containing tokens to be extracted."


def test_extract_acronyms_simple_ignores_invalid_patterns():
    """Ignores lowercase, mixed case and single-letter tokens."""
    text = "A b Ab OnS nlp are not valid acronyms."
    assert (
        extract_acronyms(text) == []
    ), "Expected invalid acronym patterns to be ignored."


def test_extract_acronyms_extended_includes_dotted_and_ampersand():
    """Extended mode captures dotted and ampersand acronyms."""
    text = (
        "A ONS works with U.S.A or U.S. on R&D and M&S&Z projects "
        "and check simple G7."
    )
    result = extract_acronyms(text, extended=True)
    assert set(result) == {
        "ONS",
        "U.S.A",
        "U.S.",
        "R&D",
        "M&S&Z",
        "G7",
    }, "Expected extended acronyms with punctuation and ampersands to be detected."


def test_extract_acronyms_extended_excludes_invalid_variants():
    """Extended mode does not match malformed dotted or spaced ampersand patterns."""
    text = "U.S is incomplete and R & D has spaces and US.A is only US."
    result = extract_acronyms(text, extended=True)
    assert "U.S" not in result, "Expected incomplete dotted acronyms to be excluded."
    assert "R & D" not in result, "Expected spaced acronyms to be excluded."
    assert "US.A" not in result, "Expected malformed dotted initials to be excluded."
    assert "US" in result, "Expected a valid uppercase token to remain in the result."


def test_extract_acronyms_extended_ignores_invalid_patterns():
    """Ignores lowercase, mixed case and single-letter tokens."""
    text = "A b Ab OnS nlp 9 are not valid acronyms."
    assert (
        extract_acronyms(text) == []
    ), "Expected extended mode to ignore invalid acronym patterns."


def test_extract_acronyms_boundary_and_punctuation_handling():
    """Correctly identifies acronyms at boundaries and next to punctuation."""
    text = "(ONS), NLP; ISO9001."
    assert extract_acronyms(text) == [
        "ONS",
        "NLP",
        "ISO9001",
    ], "Expected acronyms near punctuation and boundaries to be matched."


def test_extract_acronyms_duplicates_and_order_preserved():
    """Preserves duplicate acronyms and maintains original order."""
    text = "ONS works with ONS and NLP"
    assert extract_acronyms(text) == [
        "ONS",
        "ONS",
        "NLP",
    ], "Expected duplicate acronyms and original order to be preserved."


def test_extract_acronyms_long_token_not_split():
    """Does not split long uppercase tokens into partial matches."""
    text = "ABCDEF, A&B&C&D&E, A.B.C.D.E"
    result = extract_acronyms(text, extended=True)
    assert set(result) == {
        "ABCDEF",
        "A&B&C&D&E",
        "A.B.C.D.E",
    }, "Expected long uppercase tokens to remain intact rather than be split."


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUTS)
def test_extract_acronyms_empty_text(text):
    """Returns [] for empty text inputs."""
    assert extract_acronyms(text) == [], f"Expected [] for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUTS)
def test_extract_acronyms_non_string_inputs(text):
    """Returns [] for non-string inputs."""
    assert extract_acronyms(text) == [], f"Expected [] for non-string input: {text!r}"


# ============================================================================
# Test get_syllable_count_per_word function
# ============================================================================


def test_get_syllable_count_per_word_single_word():
    """Verify syllable count for a single word."""
    text = "hello"
    result = get_syllable_count_per_word(text)

    assert result == [2], f"Expected [2] for {text!r}, got {result}"


def test_get_syllable_count_per_word_multiple_words():
    """Validate syllable counts across multiple words."""
    text = "hello world"
    result = get_syllable_count_per_word(text)

    assert result == [2, 1], f"Expected [2, 1] for {text!r}, got {result}"


def test_get_syllable_count_per_word_multi_syllable_words():
    """Ensure words with multiple syllables are counted correctly."""
    text = "question analysis whole"
    result = get_syllable_count_per_word(text)

    assert result == [2, 4, 1], f"Expected [2, 4, 1] for {text!r}, got {result}"


def test_get_syllable_count_per_word_ignores_extra_whitespace():
    """Confirm that extra whitespace between words is handled correctly."""
    text = "hello   world"
    result = get_syllable_count_per_word(text)

    assert result == [2, 1], f"Expected [2, 1] for {text!r}, got {result}"


def test_get_syllable_count_per_word_punctuation():
    """Verify syllable counts are returned for words with punctuation."""
    text = "Hello, world!"
    result = get_syllable_count_per_word(text)

    assert result == [2, 1], f"Expected [2, 1] for {text!r}, got {result}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUTS)
def test_get_syllable_count_per_word_empty_text(text):
    """Returns [] for empty text inputs."""
    assert (
        get_syllable_count_per_word(text) == []
    ), f"Expected [] for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUTS)
def test_get_syllable_count_per_word_non_string_inputs(text):
    """Returns [] for non-string inputs."""
    assert (
        get_syllable_count_per_word(text) == []
    ), f"Expected [] for non-string input: {text!r}"


# ============================================================================
# Test get_avg_syllables_per_word function
# ============================================================================


def test_get_avg_syllables_per_word_returns_expected_average():
    """Check the average syllables per word calculation is as expected."""
    texts = [
        "Simple sentence",
        "This is a simple sentence",
        "More complex wording with variability",
        "A",
    ]

    expected = [2, 1.4, 2.4, 1]

    for i, text in enumerate(texts):
        assert (
            get_avg_syllables_per_word(text) == expected[i]
        ), f"Expected syllable average {expected[i]} for {text!r}."


def test_get_avg_syllables_per_word_matches_textstat_exactly():
    """Matches textstat's average syllables per word calculation."""
    texts = ["Simple sentence", "More complex wording with variability", "A"]

    for text in texts:
        assert get_avg_syllables_per_word(text) == textstat.avg_syllables_per_word(
            text
        ), f"Expected result to match textstat for {text!r}."


def test_get_avg_syllables_per_word_returns_float_type():
    """Returns a float value for valid text input."""
    text = "Short text"
    result = get_avg_syllables_per_word(text)

    assert isinstance(result, float), "Expected a float result for valid input text."


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUTS)
def test_get_avg_syllables_per_word_empty_text(text):
    """Returns 0 for empty text inputs."""
    assert (
        get_avg_syllables_per_word(text) == 0.0
    ), f"Expected 0 for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUTS)
def test_get_avg_syllables_per_word_non_string_inputs(text):
    """Returns 0 for non-string inputs."""
    assert (
        get_avg_syllables_per_word(text) == 0.0
    ), f"Expected 0 for non-string input: {text!r}"


# ============================================================================
# Test get_simple_language_metrics function
# ============================================================================

EXPECTED_FALSE_SIMPLE_LANGUAGE_METRICS = {
    "n_acronyms": 0,
    "avg_syllables_per_word": 0,
    "syllable_counts": [],
}


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param(
            "Would you like to participate?",
            {
                "n_acronyms": 0,
                "avg_syllables_per_word": 1.6,
                "syllable_counts": [1, 1, 1, 1, 4],
            },
            id="no_acronyms_one_four_syllable_word",
        ),
        pytest.param(
            "ONS is a UK government agency.",
            {
                "n_acronyms": 2,
                "avg_syllables_per_word": pytest.approx(1.666666666),
                "syllable_counts": [1, 1, 1, 1, 3, 3],
            },
            id="two_acronyms_two_three_syllable_words",
        ),
        pytest.param(
            "Does the NHS provide good service?",
            {
                "n_acronyms": 1,
                "avg_syllables_per_word": pytest.approx(1.33333333),
                "syllable_counts": [1, 1, 1, 2, 1, 2],
            },
            id="one_acronym_two_two_syllable_words",
        ),
    ],
)
def test_get_simple_language_metrics_returns_expected_metrics(text, expected):
    """Returns expected simple language metrics."""
    assert (
        get_simple_language_metrics(text) == expected
    ), f"Expected simple language metrics for {text!r} to be {expected}"


@pytest.mark.parametrize("text", EMPTY_TEXT_INPUTS)
def test_get_simple_language_metrics_empty_text(text):
    """Returns falsey simple language metrics for empty text inputs."""
    assert (
        get_simple_language_metrics(text) == EXPECTED_FALSE_SIMPLE_LANGUAGE_METRICS
    ), f"Expected falsey simple language metrics for empty text input: {text!r}"


@pytest.mark.parametrize("text", NON_STRING_INPUTS)
def test_get_simple_language_metrics_non_string(text):
    """Returns falsey simple language metrics for non-string inputs."""
    assert (
        get_simple_language_metrics(text) == EXPECTED_FALSE_SIMPLE_LANGUAGE_METRICS
    ), (
        f"Expected falsey simple language metrics for non-string input: {text!r} "
        f"(type: {type(text).__name__})"
    )


def test_get_simple_language_metrics_returns_expected_keys():
    """Returns all expected simple language metric keys."""
    expected_keys = {
        "n_acronyms",
        "avg_syllables_per_word",
        "syllable_counts",
    }

    assert set(get_simple_language_metrics("What is your name?")) == (
        expected_keys
    ), "Expected get_simple_language_metrics to return all expected keys"


# ============================================================================
# Test add_simple_language_columns function
# ============================================================================


def test_add_simple_language_columns_returns_expected_dataframe(
    simple_language_input_df,
    expected_simple_language_df,
):
    """Adds expected simple language metric columns and values."""
    result = add_simple_language_columns(
        simple_language_input_df,
        text_column="follow_up_question",
    )

    pd.testing.assert_frame_equal(result, expected_simple_language_df)


# ============================================================================
# Test summarise_simple_language_columns function
# ============================================================================


# ============================================================================
# Test summarise_simple_language_columns function
# ============================================================================


def test_summarise_simple_language_columns_returns_expected_summary(
    expected_simple_language_df,
):
    """Returns expected summary statistics from precomputed simple language columns."""
    result = summarise_simple_language_columns(
        expected_simple_language_df,
        prefix="follow_up_question_",
    )

    assert result["n_count"] == 5, "Expected n_count to equal the number of rows"

    assert result["pct_with_acronyms"] == pytest.approx(40.0, rel=1e-2), (
        "Expected pct_with_acronyms to equal the percentage of rows with "
        "one or more acronyms"
    )

    assert result["mean_avg_syllables_per_word"] == pytest.approx(0.92, rel=1e-2), (
        "Expected mean_avg_syllables_per_word to equal the mean of the "
        "average syllables per word column"
    )

    assert result["pct_with_word_over_syllables_threshold"] == pytest.approx(
        20.0,
        rel=1e-2,
    ), (
        "Expected pct_with_word_over_syllables_threshold to equal the percentage "
        "of rows with at least one word over the default syllables threshold"
    )


def test_summarise_simple_language_columns_uses_prefix():
    """Uses the supplied prefix to locate simple language metric columns."""
    df = pd.DataFrame(
        {
            "test_n_acronyms": [0, 1],
            "test_avg_syllables_per_word": [1.0, 2.0],
            "test_syllable_counts": [[1, 1], [2, 4]],
        }
    )

    result = summarise_simple_language_columns(
        df,
        prefix="test_",
    )

    assert result["n_count"] == 2, (
        "Expected summarise_simple_language_columns to use the supplied "
        "prefix when locating metric columns"
    )

    assert result["pct_with_acronyms"] == 50.0, (
        "Expected pct_with_acronyms to be calculated from the prefixed "
        "acronym column"
    )

    assert result["mean_avg_syllables_per_word"] == pytest.approx(1.5, rel=1e-2), (
        "Expected mean_avg_syllables_per_word to be calculated from the "
        "prefixed average syllables per word column"
    )

    assert result["pct_with_word_over_syllables_threshold"] == 50.0, (
        "Expected pct_with_word_over_syllables_threshold to be calculated "
        "from the prefixed syllable counts column"
    )


def test_summarise_simple_language_columns_uses_custom_syllables_threshold(
    expected_simple_language_df,
):
    """Uses the supplied syllables threshold when summarising syllable counts."""
    result = summarise_simple_language_columns(
        expected_simple_language_df,
        prefix="follow_up_question_",
        syllables_threshold=2,
    )

    assert result["pct_with_word_over_syllables_threshold"] == pytest.approx(
        40.0,
        rel=1e-2,
    ), (
        "Expected pct_with_word_over_syllables_threshold to use the supplied "
        "syllables threshold"
    )


def test_summarise_simple_language_columns_handles_empty_syllable_counts(
    expected_simple_language_df,
):
    """Treats empty syllable count lists as having no words over the threshold."""
    result = summarise_simple_language_columns(
        expected_simple_language_df,
        prefix="follow_up_question_",
    )

    assert result["pct_with_word_over_syllables_threshold"] == pytest.approx(
        20.0,
        rel=1e-2,
    ), (
        "Expected empty syllable count lists to be treated as having no words "
        "over the syllables threshold"
    )


def test_summarise_simple_language_columns_returns_expected_index(
    expected_simple_language_df,
):
    """Returns a Series with all expected simple language summary metric names."""
    result = summarise_simple_language_columns(
        expected_simple_language_df,
        prefix="follow_up_question_",
    )

    expected_index = {
        "n_count",
        "pct_with_acronyms",
        "mean_avg_syllables_per_word",
        "pct_with_word_over_syllables_threshold",
    }

    assert set(result.index) == expected_index, (
        "Expected summarise_simple_language_columns to return all expected "
        "summary metric names"
    )


@pytest.mark.parametrize(
    "missing_column",
    [
        pytest.param(
            "follow_up_question_n_acronyms",
            id="missing_n_acronyms",
        ),
        pytest.param(
            "follow_up_question_avg_syllables_per_word",
            id="missing_avg_syllables_per_word",
        ),
        pytest.param(
            "follow_up_question_syllable_counts",
            id="missing_syllable_counts",
        ),
    ],
)
def test_summarise_simple_language_columns_missing_column_raises_key_error(
    expected_simple_language_df,
    missing_column,
):
    """Raises KeyError when a required simple language metric column is missing."""
    df = expected_simple_language_df.drop(columns=missing_column)

    with pytest.raises(
        KeyError,
        match=missing_column,
    ):
        summarise_simple_language_columns(
            df,
            prefix="follow_up_question_",
        )


# ============================================================================
# Test SimpleLanguageMetrics function
# ============================================================================


def test_simple_language_metrics_stores_values():
    """Stores the supplied simple language metric values."""
    metrics = SimpleLanguageMetrics(
        n_count=5,
        pct_with_acronyms=40.0,
        mean_avg_syllables_per_word=0.92,
        pct_with_word_over_syllables_threshold=20.0,
    )

    assert metrics.n_count == 5, "Expected n_count to be stored"
    assert metrics.pct_with_acronyms == 40.0, "Expected pct_with_acronyms to be stored"
    assert (
        metrics.mean_avg_syllables_per_word == 0.92
    ), "Expected mean_avg_syllables_per_word to be stored"
    assert (
        metrics.pct_with_word_over_syllables_threshold == 20.0
    ), "Expected pct_with_word_over_syllables_threshold to be stored"


def test_simple_language_metrics_report_metrics_returns_expected_text():
    """Returns formatted simple language metrics as text."""
    metrics = SimpleLanguageMetrics(
        n_count=5,
        pct_with_acronyms=40.0,
        mean_avg_syllables_per_word=0.92,
        pct_with_word_over_syllables_threshold=20.0,
    )

    result = metrics.report_metrics()

    expected = "\n".join(
        [
            "\nSimple language metrics:",
            " Number of follow-up questions: 5",
            " Percentage with acronyms: 40.00%",
            " Mean average syllables per word: 0.92",
            " Percentage with words over syllables threshold: 20.00%",
        ]
    )

    assert (
        result == expected
    ), "Expected report_metrics to return correctly formatted metric text"


# ============================================================================
# Test compute_simple_language_metrics function
# ============================================================================


def test_compute_simple_language_metrics_returns_metrics_model(
    simple_language_input_df,
):
    """Returns a SimpleLanguageMetrics model."""
    result = compute_simple_language_metrics(
        simple_language_input_df,
        text_column="follow_up_question",
        prefix="follow_up_question_",
    )

    assert isinstance(result, SimpleLanguageMetrics), (
        "Expected compute_simple_language_metrics to return a "
        "SimpleLanguageMetrics instance"
    )


def test_compute_simple_language_metrics_returns_expected_values(
    simple_language_input_df,
):
    """Returns expected simple language summary values."""
    result = compute_simple_language_metrics(
        simple_language_input_df,
        text_column="follow_up_question",
        prefix="follow_up_question_",
    )

    assert (
        result.n_count == 5
    ), "Expected n_count to equal the number of rows in the input DataFrame"

    assert result.pct_with_acronyms == pytest.approx(40.0, rel=1e-2), (
        "Expected pct_with_acronyms to equal the percentage of rows with "
        "one or more acronyms"
    )

    assert result.mean_avg_syllables_per_word == pytest.approx(0.92, rel=1e-2), (
        "Expected mean_avg_syllables_per_word to equal the mean of the "
        "average syllables per word column"
    )

    assert result.pct_with_word_over_syllables_threshold == pytest.approx(
        20.0,
        rel=1e-2,
    ), (
        "Expected pct_with_word_over_syllables_threshold to equal the percentage "
        "of rows with at least one word over the default syllables threshold"
    )
