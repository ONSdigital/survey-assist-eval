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
    patterns = [
        r"\bfor example\b",
        r"\be\.g\.?\b",
        r"\bi\.e\.?\b",
        r"\bsuch as\b",
        r"(?:,|\()\s*like\b",
    ]

    text = text.lower()

    return any(re.search(pattern, text) for pattern in patterns)


def has_including_example_phrase(text: str) -> bool:
    """Check whether text contains including-style example phrases.

    Args:
        text: Question text to evaluate.

    Returns:
        True if phrases such as "including" are present,
        otherwise False.
    """
    patterns = [
        r"\bincluding\b",
        r"\bfor instance\b",
        r"\b(?:do|does|did)\b[^?]*\binclude\b",
    ]

    text = text.lower()

    return any(re.search(pattern, text) for pattern in patterns)


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
    question_text, separator, follow_on_text = text.rpartition("?")

    if not separator or not question_text.strip():
        return False

    return has_examples(follow_on_text.strip())


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


def has_closed_category_without_examples(text: str) -> bool:
    """Check whether text provides closed-category options without examples.

    Args:
        text: Question text to evaluate.

    Returns:
        True if closed-category wording is detected and no examples are present,
        otherwise False.
    """
    return has_closed_category_options(text) and not has_examples(text)
