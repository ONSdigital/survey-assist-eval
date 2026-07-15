"""Tests for interrogative detection functions in question_structure_functions.py."""

import pytest

from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    count_wh_interrogatives,
    has_interrogative_not_at_start,
    has_interrogative_start,
)

# ============================================================================
# Test Data - Shared between tests
# ============================================================================

# False cases for interrogative anywhere
INTERROGATIVE_ANYWHERE_FALSE_CASES = [
    pytest.param("Is this correct", id="is_statement_style"),
    pytest.param("This is a normal sentence", id="neutral_statement"),
    pytest.param("Describe the process clearly", id="imperative_describe"),
    pytest.param("Whatever you decide is fine", id="partial_match_whatever"),
    pytest.param("The whole idea is simple", id="partial_match_whole"),
    pytest.param("", id="empty_string"),
    pytest.param("   ", id="whitespace_only"),
]

# WH-interrogative and auxiliary starts
INTERROGATIVE_START_TRUE_CASES = [
    # WH words
    pytest.param("What is this", id="what_interrogative"),
    pytest.param("Why does this happen", id="why_interrogative"),
    pytest.param("How does it work", id="how_interrogative"),
    pytest.param("When is this due", id="when_interrogative"),
    pytest.param("Where is this located", id="where_interrogative"),
    pytest.param("Who is responsible", id="who_interrogative"),
    pytest.param("Whom does this affect", id="whom_interrogative"),
    pytest.param("Whose idea was this", id="whose_interrogative"),
    pytest.param("Which option is best", id="which_interrogative"),
    # Auxiliary verbs
    pytest.param("Is this correct", id="is_interrogative"),
    pytest.param("Are we ready", id="are_interrogative"),
    pytest.param("Do you understand", id="do_interrogative"),
    pytest.param("Does this work", id="does_interrogative"),
    pytest.param("Did you check", id="did_interrogative"),
    pytest.param("Can we proceed", id="can_interrogative"),
    pytest.param("Could you explain", id="could_interrogative"),
    pytest.param("Would this help", id="would_interrogative"),
    pytest.param("Should we continue", id="should_interrogative"),
    pytest.param("Will this change", id="will_interrogative"),
    pytest.param("Have you checked", id="have_interrogative"),
    pytest.param("Has this been done", id="has_interrogative"),
    pytest.param("Had this occurred", id="had_interrogative"),
]

# Interrogative not at start
INTERROGATIVE_NOT_AT_START_TRUE_CASES = [
    pytest.param("Tell me what this is", id="tell_what"),
    pytest.param("I wonder why that happens", id="wonder_why"),
    pytest.param("Explain how this works", id="explain_how"),
    pytest.param("I need to know when this happened", id="know_when"),
    pytest.param("Tell me where this is located", id="tell_where"),
    pytest.param("I want to know who is responsible", id="know_who"),
    pytest.param("Tell me whom this concerns", id="tell_whom"),
    pytest.param("I wonder whose idea this was", id="wonder_whose"),
    pytest.param("Tell me which option is best", id="tell_which"),
]

# No interrogative
ABSENT_INTERROGATIVE_CASES = [
    pytest.param("This is a normal sentence", id="neutral_statement"),
    pytest.param("The system works as expected", id="neutral_system"),
    pytest.param("We completed the analysis", id="neutral_analysis"),
    pytest.param("Describe the process clearly", id="imperative_describe"),
    pytest.param("Give an overview of the method", id="imperative_give"),
    pytest.param("Share your feedback", id="imperative_share"),
    pytest.param("Whatever you decide is fine", id="partial_match_whatever"),
    pytest.param("The whole idea is simple", id="partial_match_whole"),
    pytest.param("Someone handled this already", id="partial_match_someone"),
    pytest.param("It works well in practice", id="declarative_works"),
    pytest.param("He explained the results clearly", id="declarative_explained"),
]

# Edge cases
EDGE_CASES = [
    pytest.param(None, id="none"),
    pytest.param(123, id="integer"),
    pytest.param(12.5, id="float"),
    pytest.param(True, id="boolean"),
    pytest.param([], id="empty_list"),
    pytest.param({}, id="empty_dict"),
    pytest.param((), id="empty_tuple"),
    pytest.param("", id="empty_string"),
    pytest.param("   ", id="whitespace_only"),
]

# Punctuation variations
PUNCTUATION_WITH_INTERROGATIVE_START = [
    pytest.param("What is this?", id="question_mark_start"),
    pytest.param("Why does this happen!", id="exclamation_start"),
]

PUNCTUATION_WITH_INTERROGATIVE_NOT_START = [
    pytest.param("Tell me what this is?", id="question_mark_not_start"),
    pytest.param("I wonder why!", id="exclamation_not_start"),
]

# Whitespace variations
WHITESPACE_VARIATIONS_START = [
    pytest.param("What is this", id="no_extra_whitespace"),
    pytest.param("   What is this", id="leading_spaces"),
    pytest.param("\tWhat is this", id="leading_tab"),
]

WHITESPACE_VARIATIONS_NOT_START = [
    pytest.param("Tell me\twhat this is", id="tab_within_sentence"),
]

# Mixed case variations
MIXED_CASE_START = [
    pytest.param("WHAT IS THIS", id="all_uppercase"),
    pytest.param("what is this", id="all_lowercase"),
    pytest.param("What Is This", id="title_case"),
    pytest.param("WhAt Is ThIs", id="random_case"),
]

MIXED_CASE_NOT_START = [
    pytest.param("Tell me WHAT this is", id="mixed_case_not_start"),
]


# ============================================================================
# TestCountWhInterrogatives
# ============================================================================


class TestCountWhInterrogatives:
    """Tests for count_wh_interrogatives function."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            pytest.param("What is this", 1, id="single_what"),
            pytest.param("Tell me why this happens", 1, id="single_why"),
            pytest.param("I want to know how it works", 1, id="single_how"),
            pytest.param("Explain when this occurred", 1, id="single_when"),
            pytest.param("Tell me where this is", 1, id="single_where"),
            pytest.param("I wonder who is responsible", 1, id="single_who"),
            pytest.param("Tell me whom this concerns", 1, id="single_whom"),
            pytest.param("I wonder whose idea this was", 1, id="single_whose"),
            pytest.param("Tell me which option is best", 1, id="single_which"),
            pytest.param("What and why did this happen", 2, id="two_what_and_why"),
            pytest.param("Who, what and where", 3, id="three_who_what_where"),
            pytest.param("How and when should we do this", 2, id="two_how_and_when"),
            pytest.param(
                "Why, how, and when does this work", 3, id="three_why_how_when"
            ),
            pytest.param("Which option and whose idea was it", 2, id="two_which_whose"),
        ],
    )
    def test_multiple_interrogatives(self, text, expected):
        """Counts WH-interrogative words correctly for single and multiple cases."""
        assert (
            count_wh_interrogatives(text) == expected
        ), f"Expected {expected} interrogatives in: {text!r}"

    @pytest.mark.parametrize("text", INTERROGATIVE_ANYWHERE_FALSE_CASES)
    def test_false_cases(self, text):
        """Returns 0 when no WH-interrogative words are present."""
        assert (
            count_wh_interrogatives(text) == 0
        ), f"Expected 0 for non-interrogative text: {text!r}"

    @pytest.mark.parametrize("text", ABSENT_INTERROGATIVE_CASES)
    def test_absent_cases(self, text):
        """Returns 0 when no interrogative signals are present in the text."""
        assert (
            count_wh_interrogatives(text) == 0
        ), f"Expected 0 for absent interrogative: {text!r}"

    @pytest.mark.parametrize("text", MIXED_CASE_START)
    def test_case_insensitive(self, text):
        """Counts interrogatives regardless of casing."""
        assert (
            count_wh_interrogatives(text) == 1
        ), f"Should count interrogative regardless of case: {text!r}"

    @pytest.mark.parametrize("text", PUNCTUATION_WITH_INTERROGATIVE_START)
    def test_with_punctuation(self, text):
        """Counts interrogatives with punctuation marks."""
        assert (
            count_wh_interrogatives(text) == 1
        ), f"Should count interrogative with punctuation: {text!r}"

    def test_edge_cases(self):
        """Returns 0 for non-string, empty, or otherwise invalid inputs."""
        for text in EDGE_CASES:
            assert (
                count_wh_interrogatives(text) == 0
            ), f"Expected 0 for edge case: {text!r} (type: {type(text).__name__})"


# ============================================================================
# TestHasInterrogativeStart
# ============================================================================


class TestHasInterrogativeStart:
    """Tests for has_interrogative_start function."""

    @pytest.mark.parametrize("text", INTERROGATIVE_START_TRUE_CASES)
    def test_true_cases(self, text):
        """Returns True when text starts with interrogative or auxiliary."""
        assert (
            has_interrogative_start(text) is True
        ), f"Expected True for interrogative at start: {text!r}"

    @pytest.mark.parametrize("text", INTERROGATIVE_NOT_AT_START_TRUE_CASES)
    def test_false_for_middle(self, text):
        """Returns False when interrogative appears later."""
        assert (
            has_interrogative_start(text) is False
        ), f"Expected False for interrogative not at start: {text!r}"

    @pytest.mark.parametrize("text", ABSENT_INTERROGATIVE_CASES)
    def test_false_absent(self, text):
        """Returns False when no interrogative signal is present."""
        assert (
            has_interrogative_start(text) is False
        ), f"Expected False for absent interrogative: {text!r}"

    def test_case_insensitive(self):
        """Detects interrogatives regardless of casing."""
        assert (
            has_interrogative_start("WHAT is this") is True
        ), "Should detect uppercase interrogative at start"
        assert (
            has_interrogative_start("is this correct") is True
        ), "Should detect lowercase interrogative at start"

    def test_leading_whitespace(self):
        """Ignores leading whitespace."""
        assert (
            has_interrogative_start("   What is this") is True
        ), "Should detect interrogative after leading whitespace"

    @pytest.mark.parametrize("text", WHITESPACE_VARIATIONS_START)
    def test_whitespace_variations(self, text):
        """Handles various whitespace patterns."""
        assert (
            has_interrogative_start(text) is True
        ), f"Should handle whitespace variations: {text!r}"

    @pytest.mark.parametrize("text", PUNCTUATION_WITH_INTERROGATIVE_START)
    def test_punctuation_variations(self, text):
        """Handles punctuation correctly."""
        assert (
            has_interrogative_start(text) is True
        ), f"Should handle punctuation: {text!r}"

    @pytest.mark.parametrize("text", MIXED_CASE_START)
    def test_mixed_case_variations(self, text):
        """Detects interrogatives with various casing patterns."""
        assert (
            has_interrogative_start(text) is True
        ), f"Should detect mixed case: {text!r}"

    def test_edge_cases(self):
        """Returns False for all edge cases."""
        for text in EDGE_CASES:
            assert (
                has_interrogative_start(text) is False
            ), f"Expected False for edge case: {text!r} (type: {type(text).__name__})"


# ============================================================================
# TestHasInterrogativeNotAtStart
# ============================================================================


class TestHasInterrogativeNotAtStart:
    """Tests for has_interrogative_not_at_start function."""

    @pytest.mark.parametrize("text", INTERROGATIVE_NOT_AT_START_TRUE_CASES)
    def test_true_cases(self, text):
        """Returns True when WH-word appears later in the sentence."""
        assert (
            has_interrogative_not_at_start(text) is True
        ), f"Expected True for interrogative not at start: {text!r}"

    @pytest.mark.parametrize("text", INTERROGATIVE_START_TRUE_CASES)
    def test_false_for_start(self, text):
        """Returns False when interrogative is at the start."""
        assert (
            has_interrogative_not_at_start(text) is False
        ), f"Expected False for interrogative at start: {text!r}"

    @pytest.mark.parametrize("text", ABSENT_INTERROGATIVE_CASES)
    def test_false_absent(self, text):
        """Returns False when no interrogative signal is present."""
        assert (
            has_interrogative_not_at_start(text) is False
        ), f"Expected False for absent interrogative: {text!r}"

    def test_case_insensitive(self):
        """Detects WH-words regardless of casing."""
        assert (
            has_interrogative_not_at_start("Tell me WHAT this is") is True
        ), "Should detect uppercase interrogative not at start"
        assert (
            has_interrogative_not_at_start("WHY is that") is False
        ), "Should return False when uppercase interrogative is at start"

    def test_leading_whitespace(self):
        """Ignores leading whitespace."""
        assert (
            has_interrogative_not_at_start("   Tell me what this is") is True
        ), "Should detect interrogative not at start (after whitespace)"
        assert (
            has_interrogative_not_at_start("   What is this") is False
        ), "Should return False for interrogative after leading whitespace"

    @pytest.mark.parametrize("text", WHITESPACE_VARIATIONS_NOT_START)
    def test_whitespace_variations(self, text):
        """Handles various whitespace patterns."""
        assert (
            has_interrogative_not_at_start(text) is True
        ), f"Should handle whitespace variations: {text!r}"

    def test_punctuation(self):
        """Handles punctuation correctly."""
        assert (
            has_interrogative_not_at_start("Tell me, what is this?") is True
        ), "Should detect interrogative not at start with punctuation"
        assert (
            has_interrogative_not_at_start("What? Really.") is False
        ), "Should return False for interrogative at start with punctuation"

    @pytest.mark.parametrize("text", PUNCTUATION_WITH_INTERROGATIVE_NOT_START)
    def test_punctuation_variations(self, text):
        """Handles various punctuation patterns not at start."""
        assert (
            has_interrogative_not_at_start(text) is True
        ), f"Should handle punctuation: {text!r}"

    @pytest.mark.parametrize("text", MIXED_CASE_NOT_START)
    def test_mixed_case_variations(self, text):
        """Detects interrogatives with various casing patterns."""
        assert (
            has_interrogative_not_at_start(text) is True
        ), f"Should detect mixed case: {text!r}"

    def test_edge_cases(self):
        """Returns False for all edge cases."""
        for text in EDGE_CASES:
            assert (
                has_interrogative_not_at_start(text) is False
            ), f"Expected False for edge case: {text!r} (type: {type(text).__name__})"
