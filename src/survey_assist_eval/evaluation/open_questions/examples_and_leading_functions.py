"""Functions for detecting examples and leading wording in open questions."""

import re


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
        "i.e.",
        "i.e",
        "such as",
        "like",
    ]

    text = text.lower()

    return any(marker in text for marker in markers)


def has_including_example_phrase(text: str) -> bool:
    """Check whether text contains including-style example phrases.

    Args:
        text: Question text to evaluate.

    Returns:
        True if phrases such as "including" are present,
        otherwise False.
    """
    phrases = [
        "including",
        "includes",
        "for instance",
    ]

    text = text.lower()

    return any(phrase in text for phrase in phrases)


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

    text = text.lower()

    return any(phrase in text for phrase in phrases)


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
            has_including_example_phrase(text),
            has_definition_example_wording(text),
        ]
    )


def has_follow_on_examples(text: str) -> bool:
    """Check whether the final non-question sentence contains an example.

    Args:
        text: Question text to evaluate.

    Returns:
        True if the final non-question sentence contains an example,
        otherwise False.
    """
    sentences = re.findall(r"[^.!?]+[.!?]?", text)

    non_question_sentences = [
        sentence for sentence in sentences if not sentence.strip().endswith("?")
    ]

    if not non_question_sentences:
        return False

    return has_examples(non_question_sentences[-1])


def has_closed_category_options(text: str) -> bool:
    """Check whether text provides predefined response categories.

    Args:
        text: Question text to evaluate.

    Returns:
        True if closed-category wording is detected,
        otherwise False.
    """
    text = text.lower()

    return any(
        [
            " either " in text and " or " in text,
            " or " in text,
            ":" in text and (" or " in text or "," in text),
            "/" in text,
            "which of the following" in text,
            "which of these" in text,
            "select one" in text,
            "select the one" in text,
            "choose one" in text,
            "choose the one" in text,
            "pick one" in text,
            "pick the one" in text,
        ]
    )
