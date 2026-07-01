#!/usr/bin/env python3
"""This script generates a SOC classification based on respondent's data
using a Retrieval Augmented Generation (RAG) approach and persists the results.
It reloads the output from the previous stage as a DataFrame object,
performs classification for each row using an LLM, writes the results back into
the DataFrame, and then saves the results to CSV, parquet, and JSON metadata
files in a user-specified output folder.

See README_evaluation_pipeline.md for more details on how to run.
"""

# pylint: disable=duplicate-code, redefined-outer-name, protected-access

import asyncio
from argparse import Namespace
from typing import Any

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
LIKELIHOOD_THRESHOLD = 0.9

MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
EDUCATION_COL = "level_of_education"
SEMANTIC_SEARCH_COL = ["semantic_search_results", "second_semantic_search_results"]

FOLLOWUP_QUESTION_COL = "followup_question"
FOLLOWUP_ANSWER_COL = "followup_answer"

OUTPUT_COLS_INITIAL = {
    "code_col": "initial_code",
    "code_title_col": "initial_code_title",
    "likelihood_col": "initial_likelihood",
    "reasoning_col": "initial_reasoning",
    "codable_col": "unambiguously_codable",
    "alt_candidates_col": "alt_soc_candidates",
}

OUTPUT_COLS_FINAL = {
    "code_col": "final_code",
    "code_title_col": "final_code_title",
    "likelihood_col": "final_likelihood",
    "reasoning_col": "final_reasoning",
    "codable_col": "unambiguously_codable_final",
    "alt_candidates_col": "alt_soc_candidates_final",
}
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


def _prep_columns(df: pd.DataFrame, second_run_flag: bool) -> pd.DataFrame:
    """Prepares the DataFrame for RAG SOC allocation by ensuring that the necessary
    output columns exist. If the columns do not exist, they are created with default values.

    Args:
        df (pd.DataFrame): The input DataFrame containing the survey responses and semantic
            search results.
        second_run_flag (bool): Flag indicating whether this is a second run (final codes) or not.

    Returns:
        pd.DataFrame: The modified DataFrame with the necessary output columns.
    """
    out_col_names = OUTPUT_COLS_FINAL if second_run_flag else OUTPUT_COLS_INITIAL

    if all(col in df.columns for col in out_col_names.values()):
        return df  # All necessary columns already exist, no need to modify

    if out_col_names["codable_col"] not in df.columns:
        df[out_col_names["codable_col"]] = False
    if out_col_names["likelihood_col"] not in df.columns:
        df[out_col_names["likelihood_col"]] = 0.0

    for col in ["code_col", "code_title_col", "reasoning_col", "alt_candidates_col"]:
        if out_col_names[col] not in df.columns:
            df[out_col_names[col]] = ""

    if second_run_flag:
        msk = df[OUTPUT_COLS_INITIAL["codable_col"]]
        for col in out_col_names:
            df.loc[msk, OUTPUT_COLS_FINAL[col]] = df.loc[msk, OUTPUT_COLS_INITIAL[col]]

    return df


async def get_rag_response_batch_async(
    batch: pd.DataFrame,
    c_llm: ClassificationLLM,
    out_col_names: dict[str, str],
    semantic_search_col: str,
) -> list[dict[str, Any]]:
    """Processes a batch of rows concurrently to generate RAG SOC results.

    Args:
        batch: Batch of rows. Must include the columns defined by
            `JOB_TITLE_COL`, `JOB_DESCRIPTION_COL`, `MERGED_INDUSTRY_DESC_COL`, `EDUCATION_COL`, and
            `semantic_search_col`.
        c_llm: Initialised LLM wrapper used to run the RAG prompt.
        out_col_names: Dictionary mapping output column names for the results.
        semantic_search_col: Name of the column containing semantic search results.

    Returns:
        A list of dictionaries (one per row) that can be written directly into
        the output columns (`unambiguously_codable`, `initial_code`,
        `alt_soc_candidates`, `followup_question`).
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    async def _run_row(row: pd.Series):
        async with semaphore:
            respondent_data = {
                "Job title": row[JOB_TITLE_COL],
                "Job description": row[JOB_DESCRIPTION_COL],
                "Company main activity": row[MERGED_INDUSTRY_DESC_COL],
            }
            if row.get(EDUCATION_COL, None) not in {None, "unknown", "", "-8", "-9"}:
                respondent_data["Level of education"] = row[EDUCATION_COL]
            if row.get(FOLLOWUP_QUESTION_COL, None) not in {None, "", "-8", "-9"}:
                respondent_data["Followup question"] = row[FOLLOWUP_QUESTION_COL]
            if row.get(FOLLOWUP_ANSWER_COL, None) not in {None, "", "-8", "-9"}:
                respondent_data["Followup answer"] = row[FOLLOWUP_ANSWER_COL]

            candidates = c_llm._prompt_candidate_list(row.get(semantic_search_col, []))

            response = await c_llm.top_one_soc_code(
                respondent_data=respondent_data,
                code_digits=CODE_DIGITS,
                candidates_limit=CANDIDATES_LIMIT,
                semantic_search_results=row.get(semantic_search_col, []),
            )

            return respondent_data, candidates, response

    # Create tasks for each row; semaphore enforces max concurrent calls.
    tasks = [asyncio.create_task(_run_row(row)) for _, row in batch.iterrows()]

    responses = await asyncio.gather(*tasks)

    return [
        {
            out_col_names["code_col"]: resp.soc_code or "",
            out_col_names["code_title_col"]: resp.soc_title or "",
            out_col_names["likelihood_col"]: resp.likelihood_score or 0.0,
            out_col_names["reasoning_col"]: resp.reasoning or "",
            out_col_names["codable_col"]: (resp.likelihood_score or 0.0)
            >= LIKELIHOOD_THRESHOLD,
            out_col_names["alt_candidates_col"]: candidates,
            "respondent_data": respondent_data,
        }
        for respondent_data, candidates, resp in responses
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
    out_col_names = OUTPUT_COLS_FINAL if args.second_run else OUTPUT_COLS_INITIAL
    semantic_search_col = (
        SEMANTIC_SEARCH_COL[1] if args.second_run else SEMANTIC_SEARCH_COL[0]
    )

    print(
        f"Running RAG SOC allocation ({'final' if args.second_run else 'initial'})..."
    )
    df_uncodable = df if not args.second_run else df[~df["unambiguously_codable"]]

    for batch_id, batch in tqdm(
        enumerate(
            np.split(
                df_uncodable,
                np.arange(
                    start_batch_id * metadata["batch_size"],
                    len(df_uncodable),
                    metadata["batch_size"],
                ),
            )
        )
    ):
        if batch_id == 0:
            continue  # the zero batch will contain all of the processed rows so far => skip

        results = await get_rag_response_batch_async(
            batch,
            c_llm,
            out_col_names=out_col_names,
            semantic_search_col=semantic_search_col,
        )

        # Write results directly into output columns (no extra apply helpers)
        for col in [*out_col_names.values(), "respondent_data"]:
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
            completed_batches=batch_id + start_batch_id,
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

    df = _prep_columns(df, args.second_run)

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
