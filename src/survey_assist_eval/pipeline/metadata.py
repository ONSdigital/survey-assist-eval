"""Module for common metadata settings."""

import os
from argparse import Namespace
from datetime import UTC, datetime
from typing import Any

from dotenv import load_dotenv

load_dotenv()
eval_bucket = os.getenv("EVALUATION_BUCKET_NAME", "")
SIC_EMBED_SOURCE_FILE = f"gs://{eval_bucket}/sic_knowledgebase/sic_kb_for_classifai.csv"
SOC_EMBED_SOURCE_FILE = f"gs://{eval_bucket}/soc_knowledgebase/soc_kb_for_classifai.csv"


def _get_default_metadata() -> dict:
    """Returns the configuration dictionary for the LLM.

    Returns:
        dict: A dictionary containing configuration details for the embedding model
        and lookup file paths.
    """
    return {
        "embedding_model_name": "all-MiniLM-L6-v2",  # text-embedding-004
        "embedding_db_dir": "data/vector_store",
        "embedding_k_matches": 20,
        "llm_model_name": "gemini-2.5-flash",
        "llm_model_location": "europe-west2",
        "llm_candidates_limit": 10,
        "sic_code_digits": 5,
        "sic_embed_source_file": SIC_EMBED_SOURCE_FILE,
        "sic_index_file": (
            "industrial_classification_utils.data.sic_index",
            # "extended_SIC_index.xlsx",
            "uksic2007indexeswithaddendumdecember2022.xlsx",
        ),
        "sic_structure_file": (
            "industrial_classification_utils.data.sic_index",
            "publisheduksicsummaryofstructureworksheet.xlsx",
        ),
        "soc_embed_source_file": SOC_EMBED_SOURCE_FILE,
        "soc_index_file": (
            "occupational_classification_utils.data.soc_index",
            "soc2020volume2thecodingindexexcel16102024.xlsx",
        ),
        "soc_structure_file": (
            "occupational_classification_utils.data.soc_index",
            "soc2020volume1structureanddescriptionofunitgroupsexcel16102024.xlsx",
        ),
        "batch_size": 100,
        "batch_size_async": 10,
    }


def update_metadata_with_args_and_defaults(
    parsed_args: Namespace,
    in_metadata: dict | None,
) -> dict[str, Any]:
    """Updates a metadata dict with CLI args and stage-specific defaults.

    This is intended to be used by pipeline stage scripts to keep metadata
    consistent across stages.

    Behavior:
    - Always ensures `batch_size` in metadata matches `parsed_args.batch_size`.
    - Optionally sets/updates `original_dataset_name` from `parsed_args.input_file`
      (recommended only for stage 1).
    - Applies provided `defaults` only when metadata keys are missing.
    - Optionally sets `batch_size_async` capped by `max_async_batch_size`.

    Args:
        parsed_args: The command-line arguments parsed by `parse_args()`.
        in_metadata: The initial metadata dictionary loaded from input JSON.

    Returns:
        Updated metadata dictionary.
    """
    updated_metadata: dict[str, Any] = in_metadata.copy() if in_metadata else {}

    if (parsed_args.output_shortname == "STG1") and not parsed_args.second_run:
        if (
            "original_dataset_name" in updated_metadata
            and updated_metadata.get("original_dataset_name") != parsed_args.input_file
        ):
            print(
                "Warning: The original dataset name in the input metadata "
                f"({updated_metadata.get('original_dataset_name')}) does not match the input "
                f"file specified in the arguments ({parsed_args.input_file}). "
                "The metadata will be updated with the input file name."
            )
        updated_metadata["original_dataset_name"] = parsed_args.input_file

    defaults = _get_default_metadata()

    if "batch_size" in updated_metadata and parsed_args.batch_size is not None:
        if updated_metadata.get("batch_size") != parsed_args.batch_size:
            print(
                f"""Warning: The batch size in the input metadata ({
                updated_metadata.get('batch_size')
                }) does not match the batch size specified in the arguments ({
                parsed_args.batch_size
                }). The metadata will be updated with the batch size."""
            )
            updated_metadata["batch_size"] = parsed_args.batch_size
    elif "batch_size" not in updated_metadata:
        updated_metadata["batch_size"] = (
            defaults["batch_size"]
            if parsed_args.batch_size is None
            else parsed_args.batch_size
        )

    if "batch_size_async" not in updated_metadata:
        updated_metadata["batch_size_async"] = min(
            updated_metadata["batch_size"], defaults["batch_size_async"]
        )

    for key in set(defaults.keys()).difference(updated_metadata.keys()):
        updated_metadata[key] = defaults[key]

    updated_metadata[f"{parsed_args.output_shortname}_input_file"] = (
        parsed_args.input_file
    )
    updated_metadata[f"{parsed_args.output_shortname}_start_timestamp"] = datetime.now(
        UTC
    ).timestamp()
    updated_metadata[f"{parsed_args.output_shortname}_start_time_readable"] = (
        datetime.now(UTC).strftime("%Y/%m/%d_%H:%M:%S")
    )

    return updated_metadata
