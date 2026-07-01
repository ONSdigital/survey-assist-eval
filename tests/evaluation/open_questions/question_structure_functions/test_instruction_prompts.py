"""Tests for instruction prompt detection functions in question_structure_functions.py."""

import pytest

from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    count_instruction_prompts,
    has_instruction_prompt_not_at_start,
    has_instruction_prompt_start,
)

# ============================================================================
# Test Data - Shared between tests
# ============================================================================

# Instruction prompts at the start
INSTRUCTION_START_TRUE_CASES = [
    pytest.param("Tell me what happened", id="imperative_tell"),
    pytest.param("Describe your role", id="imperative_describe"),
    pytest.param("Explain this clearly", id="imperative_explain"),
    pytest.param("Please describe your role", id="please_describe"),
    pytest.param("Please explain the process", id="please_explain"),
    pytest.param("Please tell me more", id="please_tell"),
    pytest.param("Please share your feedback", id="please_share"),
    pytest.param("Share your feedback", id="imperative_share"),
    pytest.param("Give details of the issue", id="imperative_give"),
]

# Instruction prompts NOT at the start
INSTRUCTION_NOT_AT_START_TRUE_CASES = [
    pytest.param("Can you tell me what happened", id="can_you_tell"),
    pytest.param("I want you to explain this", id="i_want_explain"),
    pytest.param("Could you please describe your role", id="could_please_describe"),
    pytest.param("Can you please explain this process", id="can_please_explain"),
    pytest.param("I need you to please tell me more", id="i_need_tell"),
    pytest.param("Could you please share the details", id="could_please_share"),
    pytest.param("We need you to give details of this", id="we_need_give"),
]

# No instruction prompts
ABSENT_INSTRUCTION_CASES = [
    pytest.param("This is a normal sentence", id="neutral_statement"),
    pytest.param("The system works as expected", id="neutral_system"),
    pytest.param("We completed the analysis", id="neutral_analysis"),
    pytest.param("What is this", id="interrogative_what"),
    pytest.param("Why does this happen", id="interrogative_why"),
    pytest.param("How does it work", id="interrogative_how"),
    pytest.param("This is a shareholder report", id="partial_match_share"),
    pytest.param("Descriptive text only", id="partial_match_describe"),
    pytest.param("He explained the results clearly", id="declarative_explained"),
    pytest.param("It works well in practice", id="declarative_works"),
]

# Edge cases and non-string inputs
EDGE_CASES = [
    pytest.param("", id="empty_string"),
    pytest.param("   ", id="whitespace_only"),
    pytest.param(123, id="integer"),
    pytest.param(12.5, id="float"),
    pytest.param(True, id="boolean"),
    pytest.param(None, id="none"),
    pytest.param([], id="empty_list"),
    pytest.param({}, id="empty_dict"),
    pytest.param((), id="empty_tuple"),
]

# Punctuation variations
PUNCTUATION_WITH_INSTRUCTION_START = [
    pytest.param("Tell me what happened!", id="exclamation_start"),
    pytest.param("Describe your role.", id="period_start"),
    pytest.param("Explain this clearly?", id="question_mark_start"),
]

PUNCTUATION_WITH_INSTRUCTION_NOT_START = [
    pytest.param("Can you tell me what happened!", id="exclamation_not_start"),
    pytest.param("I need you to describe this.", id="period_not_start"),
]

# Whitespace variations
WHITESPACE_VARIATIONS_START = [
    pytest.param("Tell me what happened", id="no_extra_whitespace"),
    pytest.param("   Tell me what happened", id="leading_spaces"),
    pytest.param("\tTell me what happened", id="leading_tab"),
]

WHITESPACE_VARIATIONS_NOT_START = [
    pytest.param("Can you\ttell me more", id="tab_within_sentence"),
]

# Contractions (only where contraction is part of supporting text, not core instruction)
CONTRACTIONS_START = [
    pytest.param("Tell me what's happening", id="contraction_in_middle"),
    pytest.param("Explain what's going on", id="contraction_explain"),
]

CONTRACTIONS_NOT_START = [
    pytest.param("Can you tell me what's happening", id="contraction_in_not_start"),
]

# Mixed case variations
MIXED_CASE_START = [
    pytest.param("TELL ME WHAT HAPPENED", id="all_uppercase"),
    pytest.param("tell me what happened", id="all_lowercase"),
    pytest.param("Tell Me What Happened", id="title_case"),
    pytest.param("TeLL mE wHaT hApPeNeD", id="random_case"),
]

MIXED_CASE_NOT_START = [
    pytest.param("Can YOU tell me more", id="mixed_case_not_start"),
]


# ============================================================================
# TestCountInstructionPrompts
# ============================================================================


class TestCountInstructionPrompts:
    """Tests for count_instruction_prompts function."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            pytest.param("Please describe your role", 1, id="single_describe"),
            pytest.param("Explain the process", 1, id="single_explain"),
            pytest.param("Can you share your feedback", 1, id="single_share"),
            pytest.param("I need you to give details of this", 1, id="single_give"),
            pytest.param("Please tell me more", 1, id="single_tell"),
            pytest.param(
                "Could you please explain this", 1, id="single_please_explain"
            ),
            pytest.param("Tell me what happened", 1, id="single_imperative_tell"),
            pytest.param("Tell me and explain this", 2, id="two_tell_and_explain"),
            pytest.param(
                "Please describe and explain your role",
                2,
                id="two_describe_and_explain",
            ),
            pytest.param("Tell me, describe, and explain", 3, id="three_instructions"),
            pytest.param(
                "Can you please describe the bird and describe yourself",
                2,
                id="two_describe_repeated",
            ),
        ],
    )
    def test_multiple_instructions(self, text, expected):
        """Counts multiple instruction prompts in a single string."""
        assert (
            count_instruction_prompts(text) == expected
        ), f"Expected {expected} instruction prompts in: {text!r}"

    def test_count_zero(self):
        """Counts zero instructions in non-instruction text."""
        assert count_instruction_prompts("This is a normal sentence") == 0

    @pytest.mark.parametrize("text", PUNCTUATION_WITH_INSTRUCTION_START)
    def test_count_with_punctuation_start(self, text):
        """Counts instructions with various punctuation marks at start."""
        assert (
            count_instruction_prompts(text) == 1
        ), f"Should count instruction with punctuation: {text!r}"

    @pytest.mark.parametrize("text", PUNCTUATION_WITH_INSTRUCTION_NOT_START)
    def test_count_with_punctuation_not_start(self, text):
        """Counts instructions with punctuation when not at start."""
        assert (
            count_instruction_prompts(text) == 1
        ), f"Should count instruction with punctuation: {text!r}"

    @pytest.mark.parametrize("text", MIXED_CASE_START)
    def test_count_case_insensitive(self, text):
        """Counts instructions regardless of casing."""
        assert (
            count_instruction_prompts(text) == 1
        ), f"Should count instruction regardless of case: {text!r}"

    @pytest.mark.parametrize("text", CONTRACTIONS_START)
    def test_count_with_contractions(self, text):
        """Counts instructions that contain contractions."""
        assert (
            count_instruction_prompts(text) >= 1
        ), f"Should count instruction with contraction: {text!r}"

    def test_count_edge_cases(self):
        """Returns 0 for non-string and empty inputs."""
        for text in EDGE_CASES:
            # Extract the actual value from pytest.param
            value = text.values[0] if hasattr(text, "values") else text
            assert (
                count_instruction_prompts(value) == 0
            ), f"Expected 0 for edge case: {value!r} (type: {type(value).__name__})"


# ============================================================================
# TestHasInstructionPromptStart
# ============================================================================


class TestHasInstructionPromptStart:
    """Tests for has_instruction_prompt_start function."""

    def test_true_cases(self):
        """Returns True when instruction prompt is at the start."""
        for text in INSTRUCTION_START_TRUE_CASES:
            # Extract the actual value from pytest.param
            value = text.values[0] if hasattr(text, "values") else text
            assert (
                has_instruction_prompt_start(value) is True
            ), f"Expected True for instruction at start: {value!r}"

    def test_false_for_middle(self):
        """Returns False when instruction appears later in text."""
        for text in INSTRUCTION_NOT_AT_START_TRUE_CASES:
            # Extract the actual value from pytest.param
            value = text.values[0] if hasattr(text, "values") else text
            assert (
                has_instruction_prompt_start(value) is False
            ), f"Expected False for instruction not at start: {value!r}"

    def test_false_absent_instruction(self):
        """Returns False when no instruction prompt is present."""
        for text in ABSENT_INSTRUCTION_CASES:
            # Extract the actual value from pytest.param
            value = text.values[0] if hasattr(text, "values") else text
            assert (
                has_instruction_prompt_start(value) is False
            ), f"Expected False for non-instruction text: {value!r}"

    def test_case_insensitive(self):
        """Detects instruction prompts regardless of casing."""
        assert (
            has_instruction_prompt_start("PLEASE DESCRIBE your role") is True
        ), "Should detect uppercase instruction at start"
        assert (
            has_instruction_prompt_start("tell me more") is True
        ), "Should detect lowercase instruction at start"

    def test_leading_whitespace(self):
        """Ignores leading whitespace."""
        assert (
            has_instruction_prompt_start("   Tell me what happened") is True
        ), "Should detect instruction after leading whitespace"

    @pytest.mark.parametrize("text", WHITESPACE_VARIATIONS_START)
    def test_whitespace_variations(self, text):
        """Handles various whitespace patterns."""
        # Extract the actual value from pytest.param
        value = text.values[0] if hasattr(text, "values") else text
        assert (
            has_instruction_prompt_start(value) is True
        ), f"Should handle whitespace variations: {value!r}"

    @pytest.mark.parametrize("text", PUNCTUATION_WITH_INSTRUCTION_START)
    def test_punctuation_variations(self, text):
        """Handles instructions with different punctuation marks."""
        # Extract the actual value from pytest.param
        value = text.values[0] if hasattr(text, "values") else text
        assert (
            has_instruction_prompt_start(value) is True
        ), f"Should handle punctuation: {value!r}"

    @pytest.mark.parametrize("text", MIXED_CASE_START)
    def test_mixed_case_variations(self, text):
        """Detects instructions with various casing patterns."""
        # Extract the actual value from pytest.param
        value = text.values[0] if hasattr(text, "values") else text
        assert (
            has_instruction_prompt_start(value) is True
        ), f"Should detect mixed case: {value!r}"

    @pytest.mark.parametrize("text", CONTRACTIONS_START)
    def test_contractions(self, text):
        """Handles instructions with contractions."""
        # Extract the actual value from pytest.param
        value = text.values[0] if hasattr(text, "values") else text
        assert (
            has_instruction_prompt_start(value) is True
        ), f"Should handle contractions: {value!r}"

    def test_edge_cases(self):
        """Returns False for edge cases."""
        for text in EDGE_CASES:
            # Extract the actual value from pytest.param
            value = text.values[0] if hasattr(text, "values") else text
            assert (
                has_instruction_prompt_start(value) is False
            ), f"Expected False for edge case: {value!r} (type: {type(value).__name__})"


# ============================================================================
# TestHasInstructionPromptNotAtStart
# ============================================================================


class TestHasInstructionPromptNotAtStart:
    """Tests for has_instruction_prompt_not_at_start function."""

    def test_true_cases(self):
        """Returns True when instruction prompt appears later."""
        for text in INSTRUCTION_NOT_AT_START_TRUE_CASES:
            # Extract the actual value from pytest.param
            value = text.values[0] if hasattr(text, "values") else text
            assert (
                has_instruction_prompt_not_at_start(value) is True
            ), f"Expected True for instruction not at start: {value!r}"

    def test_false_for_start(self):
        """Returns False when instruction is at the start."""
        for text in INSTRUCTION_START_TRUE_CASES:
            # Extract the actual value from pytest.param
            value = text.values[0] if hasattr(text, "values") else text
            assert (
                has_instruction_prompt_not_at_start(value) is False
            ), f"Expected False for instruction at start: {value!r}"

    def test_false_absent_instruction(self):
        """Returns False when no instruction prompt is present."""
        for text in ABSENT_INSTRUCTION_CASES:
            # Extract the actual value from pytest.param
            value = text.values[0] if hasattr(text, "values") else text
            assert (
                has_instruction_prompt_not_at_start(value) is False
            ), f"Expected False for non-instruction text: {value!r}"

    def test_case_insensitive(self):
        """Detects instruction prompts regardless of casing."""
        assert (
            has_instruction_prompt_not_at_start("Can you TELL ME more") is True
        ), "Should detect uppercase instruction not at start"
        assert (
            has_instruction_prompt_not_at_start("PLEASE EXPLAIN this") is False
        ), "Should return False when uppercase instruction is at start"

    def test_leading_whitespace(self):
        """Handles leading whitespace correctly."""
        assert (
            has_instruction_prompt_not_at_start("   Tell me more") is False
        ), "Should return False for instruction after leading whitespace"
        assert (
            has_instruction_prompt_not_at_start("   Can you tell me more") is True
        ), "Should detect instruction not at start (after whitespace)"

    @pytest.mark.parametrize("text", WHITESPACE_VARIATIONS_NOT_START)
    def test_whitespace_variations(self, text):
        """Handles various whitespace patterns."""
        # Extract the actual value from pytest.param
        value = text.values[0] if hasattr(text, "values") else text
        assert (
            has_instruction_prompt_not_at_start(value) is True
        ), f"Should handle whitespace variations: {value!r}"

    def test_punctuation(self):
        """Handles punctuation correctly."""
        assert (
            has_instruction_prompt_not_at_start("Can you, tell me what happened?")
            is True
        ), "Should detect instruction not at start with punctuation"
        assert (
            has_instruction_prompt_not_at_start("Tell me, what happened?") is False
        ), "Should return False for instruction at start followed by punctuation"

    @pytest.mark.parametrize("text", PUNCTUATION_WITH_INSTRUCTION_NOT_START)
    def test_punctuation_variations(self, text):
        """Handles various punctuation patterns not at start."""
        # Extract the actual value from pytest.param
        value = text.values[0] if hasattr(text, "values") else text
        assert (
            has_instruction_prompt_not_at_start(value) is True
        ), f"Should handle punctuation: {value!r}"

    @pytest.mark.parametrize("text", MIXED_CASE_NOT_START)
    def test_mixed_case_variations(self, text):
        """Detects instructions with various casing patterns."""
        # Extract the actual value from pytest.param
        value = text.values[0] if hasattr(text, "values") else text
        assert (
            has_instruction_prompt_not_at_start(value) is True
        ), f"Should detect mixed case: {value!r}"

    @pytest.mark.parametrize("text", CONTRACTIONS_NOT_START)
    def test_contractions(self, text):
        """Handles instructions with contractions not at start."""
        # Extract the actual value from pytest.param
        value = text.values[0] if hasattr(text, "values") else text
        assert (
            has_instruction_prompt_not_at_start(value) is True
        ), f"Should handle contractions: {value!r}"

    def test_edge_cases(self):
        """Returns False for edge cases."""
        for text in EDGE_CASES:
            # Extract the actual value from pytest.param
            value = text.values[0] if hasattr(text, "values") else text
            assert (
                has_instruction_prompt_not_at_start(value) is False
            ), f"Expected False for edge case: {value!r} (type: {type(value).__name__})"
