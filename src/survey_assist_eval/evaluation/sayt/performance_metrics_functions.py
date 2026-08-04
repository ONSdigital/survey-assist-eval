"""Performance metrics functions for SAYT evaluation."""


def compute_precision_at_k(
    retrieved_codes: list[str], correct_code: str, k: int
) -> float:
    """Compute Precision@K for a single query.

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_code: The correct code for the query.
        k: The cutoff rank at which to compute precision.

    Returns:
        float: Precision@K value.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    # Get the top-k retrieved items
    top_k_retrieved = retrieved_codes[:k]

    # Count how many of the top-k retrieved items are relevant
    relevant_count = sum(1 for item in top_k_retrieved if item == correct_code)

    # Calculate Precision@K
    precision_at_k = relevant_count / k

    return precision_at_k


def compute_recall_at_k(retrieved_codes: list[str], correct_code: str, k: int) -> float:
    """Compute Recall@K for a single query.

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_code: The correct code for the query.
        k: The cutoff rank at which to compute recall.

    Returns:
        float: Recall@K value.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    # Get the top-k retrieved items
    top_k_retrieved = retrieved_codes[:k]

    # Check whether the correct code appears in the top-k retrieved codes
    relevant_count = 1 if correct_code in top_k_retrieved else 0

    # Calculate Recall@K
    recall_at_k = float(relevant_count)

    return recall_at_k


def compute_mrr(retrieved_codes: list[str], correct_code: str) -> float:
    """Compute Mean Reciprocal Rank (MRR) for a single query.

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_code: The correct code for the query.

    Returns:
        float: MRR value.
    """
    for rank, item in enumerate(retrieved_codes, start=1):
        if item == correct_code:
            return 1 / rank
    return 0.0


def compute_mean_rank(retrieved_codes: list[str], correct_code: str) -> float:
    """Compute Mean Rank for a single query.

    Args:
        retrieved_codes: List of codes retrieved by the system (ordered by relevance).
        correct_code: The correct code for the query.

    Returns:
        float: Mean Rank value.
    """
    for rank, item in enumerate(retrieved_codes, start=1):
        if item == correct_code:
            return float(rank)
    return 0.0
