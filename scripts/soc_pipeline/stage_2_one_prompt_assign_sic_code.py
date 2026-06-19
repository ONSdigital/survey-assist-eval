#!/usr/bin/env python3
"""This script generates a SOC classification based on respondent's data
using a Retrieval Augmented Generation (RAG) approach and persists the results.
It reloads the output from the previous stage as a DataFrame object,
performs classification for each row using an LLM, writes the results back into
the DataFrame, and then saves the results to CSV, parquet, and JSON metadata
files in a user-specified output folder.

See README_evaluation_pipeline.md for more details on how to run.
"""

# pylint: disable=duplicate-code, redefined-outer-name

import asyncio
from argparse import Namespace
from typing import Any, cast

import numpy as np
import pandas as pd
from occupational_classification_utils.llm.llm import ClassificationLLM
from tqdm import tqdm

from survey_assist_eval.pipeline.shared_components import (
    parse_args,
    persist_results,
    set_up_initial_state,
)

#####################################################
# Constants:
CODE_DIGITS = 5
CANDIDATES_LIMIT = 10
MAX_CONCURRENT_TASKS = 10

MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
EDUCATION_COL = "level_of_education"
SEMANTIC_SEARCH_COL = "semantic_search_results"
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


async def get_rag_response_batch_async(
    batch: pd.DataFrame, c_llm: ClassificationLLM
) -> list[dict[str, Any]]:
    """Processes a batch of rows concurrently to generate RAG SOC results.

    Args:
        batch: Batch of rows. Must include the columns defined by
            `JOB_TITLE_COL`, `JOB_DESCRIPTION_COL`, `MERGED_INDUSTRY_DESC_COL`, `EDUCATION_COL`, and
            `SEMANTIC_SEARCH_COL`.
        c_llm: Initialised LLM wrapper used to run the RAG prompt.

    Returns:
        A list of dictionaries (one per row) that can be written directly into
        the output columns (`unambiguously_codable`, `initial_code`,
        `alt_soc_candidates`, `followup_question`).
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    async def _run_row(row: pd.Series):
        async with semaphore:
            return await c_llm.top_one_soc_code(
                respondent_data={
                    "Job title": row[JOB_TITLE_COL],
                    "Job description": row[JOB_DESCRIPTION_COL],
                    "Company main activity": row[MERGED_INDUSTRY_DESC_COL],
                    "Level of education": row.get(EDUCATION_COL, "unknown"),
                },
                code_digits=CODE_DIGITS,
                candidates_limit=CANDIDATES_LIMIT,
                semantic_search_results=row.get(SEMANTIC_SEARCH_COL, []),
            )

    # Create tasks for each row; semaphore enforces max concurrent calls.
    tasks = [asyncio.create_task(_run_row(row)) for _, row in batch.iterrows()]

    responses = await asyncio.gather(*tasks)

    return [
        {
            "initial_code": resp.soc_code or "",
            "code_title": resp.soc_descriptive or "",
            "likelihood": resp.likelihood or 0.0,
            "reasoning": resp.reasoning or "",
        }
        for resp in responses
    ]


async def main_async(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    start_batch_id: int,
    args: Namespace,
    c_llm: ClassificationLLM,
) -> None:
    """Runs async RAG SOC allocation and persists checkpoints.

    Args:
        df: The input DataFrame containing the survey responses and semantic search results.
        metadata: The metadata dictionary loaded from the input JSON file.
        start_batch_id: The batch ID to start from, determined by the checkpointing logic.
        args: The command-line arguments parsed by `parse_args()`.
        c_llm: An initialised instance of the ClassificationLLM class.
    """
    print("Running RAG SOC allocation...")

    batch_size = metadata["batch_size"]
    start_row = start_batch_id * batch_size

    for completed_batches, batch_start in tqdm(
        enumerate(
            range(start_row, len(df), batch_size),
            start=(start_batch_id + 1),
        )
    ):
        batch = cast(pd.DataFrame, df.iloc[batch_start : (batch_start + batch_size)])
        results = await get_rag_response_batch_async(batch, c_llm)

        # Write results directly into output columns (no extra apply helpers)
        for col in ["initial_code", "code_title", "likelihood", "reasoning"]:
            if col not in df.columns:
                df[col] = np.nan  # create column if it doesn't exist
            df.loc[batch.index, col] = pd.Series(
                [r[col] for r in results],
                index=batch.index,
            )

        persist_results(
            df=df,
            metadata=metadata,
            output_folder=args.output_folder,
            output_shortname=args.output_shortname,
            is_final=False,
            completed_batches=completed_batches,
        )

    print("RAG SOC allocation is complete")
    print("persisting results...")
    persist_results(
        df=df,
        metadata=metadata,
        output_folder=args.output_folder,
        output_shortname=args.output_shortname,
        is_final=True,
    )
    print("Done!")


if __name__ == "__main__":
    args = parse_args("STG2")
    df, metadata, start_batch_id = set_up_initial_state(args)

    uni_chat = ClassificationLLM(
        model_name=metadata["llm_model_name"],
        # model_location=metadata["llm_model_location"],
        verbose=False,
    )

    asyncio.run(
        main_async(
            df=df,
            metadata=metadata,
            start_batch_id=start_batch_id,
            args=args,
            c_llm=uni_chat,
        )
    )
