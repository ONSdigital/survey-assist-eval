"""Module for common constant definitions.

This module contains constants used across the industrial classification utilities.
"""


def get_default_config() -> dict:
    """Returns the configuration dictionary for the LLM.

    Returns:
        dict: A dictionary containing configuration details for the embedding model
        and lookup file paths.
    """
    return {
        "embedding": {
            "embedding_model_name": "all-MiniLM-L6-v2",  # text-embedding-004
            "db_dir": "data/vector_store",
            "k_matches": 20,
        },
        "llm": {
            "llm_model_name": "gemini-2.5-flash",
            "model_location": "europe-west2",
            "candidates_limit": 10,
        },
        "sic_data": {
            "code_digits": 5,
            "sic_index": (
                "industrial_classification_utils.data.sic_index",
                # "extended_SIC_index.xlsx",
                "uksic2007indexeswithaddendumdecember2022.xlsx",
            ),
            "sic_structure": (
                "industrial_classification_utils.data.sic_index",
                "publisheduksicsummaryofstructureworksheet.xlsx",
            ),
            "sic_condensed": (
                "industrial_classification_utils.data.example",
                "sic_2d_condensed.txt",
            ),
        },
    }
