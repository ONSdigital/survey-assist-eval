"""Tests for question structure functions."""
import pytest

from survey_assist_eval.evaluation.open_questions.question_structure_functions import (
    has_question_mark,
    has_interrogative_not_at_start,
    has_interrogative_start,
    has_interrogative_anywhere,
    has_instruction_prompt_anywhere,
    has_instruction_prompt_start,
    has_instruction_prompt_not_at_start,
    count_question_signals
)


INSTRUCTION_PROMPT_START_CASES = [
    "Tell me what happened",
    "Describe your role",
    "Explain this clearly",
    "Please describe your role",
    "Please explain the process",
    "Please tell me more",
    "Please share the details",
    "Share your feedback",
    "Give details of the issue",
]

INSTRUCTION_PROMPT_NOT_AT_START_CASES = [
    "Can you tell me what happened",
    "Can you describe your role",
    "I want you to explain this",
    "Could you please describe your role",
    "Can you please explain this process",
    "I need you to please tell me more",
    "Could you please share the details",
    "Can you share your feedback",
    "We need you to give details of this",
]



def test_has_question_mark_returns_true_when_present():
    """Returns True when the text contains a question mark."""
    text = "Is this a question?"
    assert has_question_mark(text) is True


def test_has_question_mark_returns_false_when_absent():
    """Returns False when the text does not contain a question mark."""
    text = "This is a statement."
    assert has_question_mark(text) is False


def test_has_question_mark_multiple_question_marks():
    """Returns True when multiple question marks are present."""
    text = "What?? Really??"
    assert has_question_mark(text) is True


def test_has_question_mark_empty_string():
    """Returns False when input text is empty."""
    assert has_question_mark("") is False


def test_has_question_mark_none_input():
    """Returns False when input is None."""
    assert has_question_mark(None) is False


def test_has_question_mark_non_string_input():
    """Returns False when input is not a string."""
    assert has_question_mark(123) is False


def test_has_question_mark_whitespace_only():
    """Returns False when input contains only whitespace."""
    assert has_question_mark("   ") is False


def test_has_question_mark_question_mark_with_spaces():
    """Returns True when question mark is surrounded by whitespace."""
    assert has_question_mark(" ? ") is True


class TestInterrogatives:
    """Tests for interrogative detection functions."""

    INTERROGATIVE_ANYWHERE_TRUE_CASES = [
        "What is this",
        "Tell me why this happens",
        "I want to know how it works",
        "Explain when this occurred",
        "Tell me where this is",
        "I wonder who is responsible",
        "Tell me whom this concerns",
        "I wonder whose idea this was",
        "Tell me which option is best",
    ]

    INTERROGATIVE_ANYWHERE_FALSE_CASES = [
        "This is a normal sentence",
        "Describe the process clearly",
        "Whatever you decide is fine",
        "The whole idea is simple",
        "",
        "   ",
    ]

    INTERROGATIVE_START_TRUE_CASES = [
        # WH words
        "What is this",
        "Why does this happen",
        "How does it work",
        "When is this due",
        "Where is this located",
        "Who is responsible",
        "Whom does this affect",
        "Whose idea was this",
        "Which option is best",

        # auxiliary verbs
        "Is this correct",
        "Are we ready",
        "Do you understand",
        "Does this work",
        "Did you check",
        "Can we proceed",
        "Could you explain",
        "Would this help",
        "Should we continue",
        "Will this change",
        "Have you checked",
        "Has this been done",
        "Had this occurred",
    ]

    INTERROGATIVE_NOT_AT_START_TRUE_CASES = [
        "Tell me what this is",
        "I wonder why that happens",
        "Explain how this works",
        "I need to know when this happened",
        "Tell me where this is located",
        "I want to know who is responsible",
        "Tell me whom this concerns",
        "I wonder whose idea this was",
        "Tell me which option is best",
    ]

    ABSENT_INTERROGATIVE_CASES = [
        # Plain neutral sentences
        "This is a normal sentence",
        "The system works as expected",
        "We completed the analysis",
        
        # Instruction / statement (no WH, no auxiliary start)
        "Describe the process clearly",
        "Give an overview of the method",
        "Share your feedback",
        
        # Partial match traps (important!)
        "Whatever you decide is fine",
        "The whole idea is simple",
        "Someone handled this already",
        
        # Declarative sentences with verbs
        "It works well in practice",
        "He explained the results clearly",
        
        ]


    EDGE_CASES = [
        None,
        123,
        12.5,
        True,
        [],
        {},
        (),
        "",
        "   ",
    ]



    # ===== ANYWHERE =====

    @pytest.mark.parametrize("text", INTERROGATIVE_ANYWHERE_TRUE_CASES)
    def test_has_interrogative_anywhere_true(self, text):
        """Returns True when WH-word appears anywhere."""
        assert has_interrogative_anywhere(text) is True
        
    
    @pytest.mark.parametrize("text", INTERROGATIVE_ANYWHERE_FALSE_CASES)
    def test_has_interrogative_anywhere_false(self, text):
        """Returns False when no WH-word is present."""
        assert has_interrogative_anywhere(text) is False


    @pytest.mark.parametrize("text", ABSENT_INTERROGATIVE_CASES)
    def test_has_interrogative_anywhere_false(self, text):
        """Returns False when no interrogative signal is present."""
        assert has_interrogative_anywhere(text) is False

    @pytest.mark.parametrize("text", EDGE_CASES)
    def test_has_interrogative_anywhere_EDGE_CASES(self, text):
        """Returns False for all edge cases."""
        assert has_interrogative_anywhere(text) is False

    # ===== START =====

    @pytest.mark.parametrize("text", INTERROGATIVE_START_TRUE_CASES)
    def test_has_interrogative_start_returns_true(self, text):
        """Returns True when text starts with interrogative or auxiliary."""
        assert has_interrogative_start(text) is True

    def test_has_interrogative_start_case_insensitive(self):
        """Detects interrogatives regardless of casing."""
        assert has_interrogative_start("WHAT is this") is True
        assert has_interrogative_start("is this correct") is True

    def test_has_interrogative_start_with_leading_whitespace(self):
        """Ignores leading whitespace."""
        assert has_interrogative_start("   What is this") is True

    @pytest.mark.parametrize("text", INTERROGATIVE_NOT_AT_START_TRUE_CASES)
    def test_has_interrogative_start_returns_false_for_middle(self, text):
        """Returns False when interrogative appears later."""
        assert has_interrogative_start(text) is False

    def test_has_interrogative_start_punctuation(self):
        """Handles punctuation correctly."""
        assert has_interrogative_start("What? Really.") is True

    @pytest.mark.parametrize("text", ABSENT_INTERROGATIVE_CASES)
    def test_has_interrogative_start_false(self, text):
        """Returns False when no interrogative signal is present."""
        assert has_interrogative_start(text) is False

    @pytest.mark.parametrize("text", EDGE_CASES)
    def test_has_interrogative_start_EDGE_CASES(self, text):
        """Returns False for all edge cases."""
        assert has_interrogative_start(text) is False

    # ===== NOT AT START =====

    @pytest.mark.parametrize("text", INTERROGATIVE_NOT_AT_START_TRUE_CASES)
    def test_has_interrogative_not_at_start_returns_true(self, text):
        """Returns True when WH-word appears later in the sentence."""
        assert has_interrogative_not_at_start(text) is True

    @pytest.mark.parametrize("text", INTERROGATIVE_START_TRUE_CASES)
    def test_has_interrogative_not_at_start_returns_false_for_start(self, text):
        """Returns False when interrogative is at the start."""
        assert has_interrogative_not_at_start(text) is False

    def test_has_interrogative_not_at_start_case_insensitive(self):
        """Detects WH-words regardless of casing."""
        assert has_interrogative_not_at_start("Tell me WHAT this is") is True
        assert has_interrogative_not_at_start("WHY is that") is False

    def test_has_interrogative_not_at_start_with_leading_whitespace(self):
        """Ignores leading whitespace."""
        assert has_interrogative_not_at_start("   Tell me what this is") is True
        assert has_interrogative_not_at_start("   What is this") is False


    def test_has_interrogative_not_at_start_punctuation(self):
        """Handles punctuation correctly."""
        assert has_interrogative_not_at_start("Tell me, what is this?") is True
        assert has_interrogative_not_at_start("What? Really.") is False

    @pytest.mark.parametrize("text", ABSENT_INTERROGATIVE_CASES)
    def test_has_interrogative_not_at_start_false(self, text):
        """Returns False when no interrogative signal is present."""
        assert has_interrogative_not_at_start(text) is False

    @pytest.mark.parametrize("text", EDGE_CASES)
    def test_has_interrogative_not_at_start_EDGE_CASES(self, text):
        """Returns False for all edge cases."""
        assert has_interrogative_not_at_start(text) is False



def test_has_instruction_prompt_not_at_start_returns_false_when_at_start():
    """Returns False when instruction prompt is the first phrase."""
    assert has_instruction_prompt_not_at_start("Tell me what happened") is False
    assert has_instruction_prompt_not_at_start("Explain this clearly") is False
    assert has_instruction_prompt_not_at_start("Please describe your role") is False


class TestInstructionPrompts:
    """Tests for instruction prompt detection functions."""

    INSTRUCTION_ANYWHERE_TRUE_CASES = [
        "Tell me what happened",
        "Please describe your role",
        "Explain the process",
        "Can you share your feedback",
        "I need you to give details of this",
        "Please tell me more",
        "Could you please explain this",
    ]

    INSTRUCTION_START_TRUE_CASES = [
        "Tell me what happened",
        "Describe your role",
        "Explain this clearly",
        "Please describe your role",
        "Please explain the process",
        "Please tell me more",
        "Please share your feedback",
        "Share your feedback",
        "Give details of the issue",
    ]

    INSTRUCTION_NOT_AT_START_TRUE_CASES = [
        "Can you tell me what happened",
        "I want you to explain this",
        "Could you please describe your role",
        "Can you please explain this process",
        "I need you to please tell me more",
        "Could you please share the details",
        "We need you to give details of this",
    ]

    ABSENT_INSTRUCTION_CASES = [
        # Neutral statements
        "This is a normal sentence",
        "The system works as expected",
        "We completed the analysis",

        # Interrogatives (but not instructions)
        "What is this",
        "Why does this happen",
        "How does it work",

        # Partial match traps
        "This is a shareholder report",
        "Descriptive text only",

        # Declarative sentences
        "He explained the results clearly",
        "It works well in practice",
    ]

    EDGE_CASES = [
        None,
        123,
        12.5,
        True,
        [],
        {},
        (),
        "",
        "   ",
    ]

    # ===== ANYWHERE =====

    @pytest.mark.parametrize("text", INSTRUCTION_ANYWHERE_TRUE_CASES)
    def test_has_instruction_prompt_anywhere_true(self, text):
        """Returns True when instruction prompt appears anywhere."""
        assert has_instruction_prompt_anywhere(text) is True

    @pytest.mark.parametrize("text", ABSENT_INSTRUCTION_CASES)
    def test_has_instruction_prompt_anywhere_false(self, text):
        """Returns False when no instruction prompt is present."""
        assert has_instruction_prompt_anywhere(text) is False

    @pytest.mark.parametrize("text", EDGE_CASES)
    def test_has_instruction_prompt_anywhere_edge_cases(self, text):
        """Returns False for edge cases."""
        assert has_instruction_prompt_anywhere(text) is False

    # ===== START =====

    @pytest.mark.parametrize("text", INSTRUCTION_START_TRUE_CASES)
    def test_has_instruction_prompt_start_true(self, text):
        """Returns True when instruction prompt is at the start."""
        assert has_instruction_prompt_start(text) is True

    @pytest.mark.parametrize("text", INSTRUCTION_NOT_AT_START_TRUE_CASES)
    def test_has_instruction_prompt_start_false_for_middle(self, text):
        """Returns False when instruction appears later in text."""
        assert has_instruction_prompt_start(text) is False

    @pytest.mark.parametrize("text", ABSENT_INSTRUCTION_CASES)
    def test_has_instruction_prompt_start_false(self, text):
        """Returns False when no instruction prompt is present."""
        assert has_instruction_prompt_start(text) is False

    def test_has_instruction_prompt_start_case_insensitive(self):
        """Detects instruction prompts regardless of casing."""
        assert has_instruction_prompt_start("PLEASE DESCRIBE your role") is True
        assert has_instruction_prompt_start("tell me more") is True

    def test_has_instruction_prompt_start_with_leading_whitespace(self):
        """Ignores leading whitespace."""
        assert has_instruction_prompt_start("   Tell me what happened") is True

    @pytest.mark.parametrize("text", EDGE_CASES)
    def test_has_instruction_prompt_start_edge_cases(self, text):
        """Returns False for edge cases."""
        assert has_instruction_prompt_start(text) is False

    # ===== NOT AT START =====

    @pytest.mark.parametrize("text", INSTRUCTION_NOT_AT_START_TRUE_CASES)
    def test_has_instruction_prompt_not_at_start_true(self, text):
        """Returns True when instruction prompt appears later."""
        assert has_instruction_prompt_not_at_start(text) is True

    @pytest.mark.parametrize("text", INSTRUCTION_START_TRUE_CASES)
    def test_has_instruction_prompt_not_at_start_false_for_start(self, text):
        """Returns False when instruction is at the start."""
        assert has_instruction_prompt_not_at_start(text) is False

    @pytest.mark.parametrize("text", ABSENT_INSTRUCTION_CASES)
    def test_has_instruction_prompt_not_at_start_false(self, text):
        """Returns False when no instruction prompt is present."""
        assert has_instruction_prompt_not_at_start(text) is False

    def test_has_instruction_prompt_not_at_start_case_insensitive(self):
        """Detects instruction prompts regardless of casing."""
        assert has_instruction_prompt_not_at_start("Can you TELL ME more") is True
        assert has_instruction_prompt_not_at_start("PLEASE EXPLAIN this") is False

    def test_has_instruction_prompt_not_at_start_with_leading_whitespace(self):
        """Handles leading whitespace correctly."""
        assert has_instruction_prompt_not_at_start("   Tell me more") is False
        assert has_instruction_prompt_not_at_start("   Can you tell me more") is True

    def test_has_instruction_prompt_not_at_start_punctuation(self):
        """Handles punctuation correctly."""
        assert has_instruction_prompt_not_at_start("Can you, tell me what happened?") is True
        assert has_instruction_prompt_not_at_start("Tell me, what happened?") is False

    @pytest.mark.parametrize("text", EDGE_CASES)
    def test_has_instruction_prompt_not_at_start_edge_cases(self, text):
        """Returns False for edge cases."""
        assert has_instruction_prompt_not_at_start(text) is False


def test_count_question_signals_no_signals():
    """No question signals should return 0."""
    text = "This is a statement."
    assert count_question_signals(text) == 0


def test_count_question_signals_question_mark_only():
    """Only a question mark should return 1."""
    text = "This is a question?"
    assert count_question_signals(text) == 1


def test_count_question_signals_interrogative_start_only():
    """Interrogative at start only."""
    text = "What is your name"
    assert count_question_signals(text) == 1


def test_count_question_signals_interrogative_not_at_start_only():
    """Interrogative not at start."""
    text = "I wonder what this is"
    assert count_question_signals(text) == 1


def test_count_question_signals_instruction_prompt_start_only():
    """Instruction prompt at start."""
    text = "Tell me your name"
    assert count_question_signals(text) == 1


def test_count_question_signals_instruction_prompt_not_at_start_only():
    """Instruction prompt not at start."""
    text = "I want you to tell me your name"
    assert count_question_signals(text) == 1


def test_count_question_signals_all_signals_present():
    """Returns the number of distinct signal types present."""
    text = (
        "What do you think? I wonder what this is. "
        "Tell me something. I want you to tell me more."
    )
    assert count_question_signals(text) == 3


def test_count_question_signals_multiple_occurrences_same_signal():
    """
    Multiple occurrences of the same signal should still count as 1
    because signals are distinct, not cumulative.
    """
    text = "What is this? What is that? What is anything?"
    assert count_question_signals(text) == 2


def test_count_question_signals_empty_string():
    """Empty input should return 0."""
    assert count_question_signals("") == 0


def test_count_question_signals_whitespace_string():
    """Whitespace-only input should return 0."""
    assert count_question_signals("   ") == 0


def test_count_question_signals_mixed_case_handling():
    """Case should not affect detection."""
    text = "WHAT is this?"
    assert count_question_signals(text) >= 1


def test_count_question_signals_punctuation_without_question():
    """Other punctuation should not count."""
    text = "This is surprising!"
    assert count_question_signals(text) == 0
