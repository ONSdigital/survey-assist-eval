#!/usr/bin/env python3
"""This script performs semantic search on a dataset using a local vector store
and persists the results. It reads in a CSV/PARQUET file as a DataFrame object,
uses :class:`industrial_classification_utils.embed.embedding.EmbeddingHandler`
to obtain semantic search results for each row, creates a new column in the
DataFrame with this information, and then saves the results to CSV, parquet,
and JSON metadata files in a user-specified output folder.

See README_evaluation_pipeline.md for more details on how to run.
"""

# pylint: disable=duplicate-code, redefined-outer-name

from re import sub as regex_sub

import numpy as np
import pandas as pd
from industrial_classification_utils.embed.embedding import EmbeddingHandler
from tqdm import tqdm

from survey_assist_eval.data_cleaning.constants import LEVEL_OF_EDUCATION
from survey_assist_eval.pipeline.shared_components import (
    parse_args,
    persist_results,
    set_up_initial_state,
)

#####################################################
# Default values and constants:
JOB_TITLE_COL = "soc2020_job_title"
JOB_DESCRIPTION_COL = "soc2020_job_description"
INDUSTRY_DESCR_COL = "sic2007_employee"
SELF_EMPLOYED_DESC_COL = "sic2007_self_employed"
EDUCATION_COL = "level_of_education"

MERGED_INDUSTRY_DESC_COL = "merged_industry_desc"
OUTPUT_COL_INITIAL = "semantic_search_results"

# Constants for second run (final codes):
FOLLOWUP_ANSWER_COL = "followup_answer"
OUTPUT_COL_FINAL = "second_semantic_search_results"
#####################################################

# Enable progress bar for semantic-search
tqdm.pandas()


def clean_text(text: str) -> str:
    """Cleans a text string by removing newlines, converting arbitrary
    whitespace to a single space, removing -9's and standardizing case.

    Args:
        text (str): The input string to clean.

    Returns:
        str: The cleaned string.
    """
    if not isinstance(text, str) or text == "-9":
        text = ""
    text = text.replace("\n", " ")
    text = regex_sub(r"\s+", " ", text)
    text = text.lower().strip().capitalize()
    return text


def make_merged_industry_desc(row: pd.Series) -> str:
    """Merges the main industry description column with the self-employed description column.

    Args:
        row (pd.Series): A row from the input DataFrame containing industry description,
                         self employed description.

    Returns:
        description (str): The merged descriptions.
    """
    ind_desc = (
        row[INDUSTRY_DESCR_COL] if isinstance(row[INDUSTRY_DESCR_COL], str) else ""
    )
    self_emp_desc = (
        row[SELF_EMPLOYED_DESC_COL]
        if isinstance(row[SELF_EMPLOYED_DESC_COL], str)
        else ""
    )

    return f"{ind_desc}{self_emp_desc}"


def expand_level_of_education(row: pd.Series) -> str:
    """Expands the level of education to the qualification description.

    Args:
        row (pd.Series): A row from the input DataFrame containing level of education.

    Returns:
        description (str): The expanded descriptions.

    """
    education_coded = str(row["level_of_education"])
    if education_coded in LEVEL_OF_EDUCATION:
        return LEVEL_OF_EDUCATION[education_coded]
    return education_coded


def _make_embedding_handler(in_metadata: dict) -> EmbeddingHandler:
    """Create an :class:`EmbeddingHandler` using settings from metadata where possible."""
    new_embedding_handler = EmbeddingHandler(
        embedding_model_name=in_metadata["embedding_model_name"],
        db_dir=in_metadata["embedding_db_dir"],
        k_matches=in_metadata["embedding_k_matches"],
        index_source_file=in_metadata["soc_embed_source_file"],
    )

    return new_embedding_handler


def _get_semantic_search_results(
    row: pd.Series, one_embedding_handler: EmbeddingHandler, second_run_flag: bool
) -> list[dict]:
    """Performs a semantic search using the provided row data.

    Intended for use as a `.apply()` operation to create a new column in a
    pd.DataFrame.

    Returns:
        list[dict]: List of dictionaries with `title`, `code`, `distance`.
        one_embedding_handler (EmbeddingHandler): An instance of the EmbeddingHandler class
            to perform the search.
        second_run_flag (bool): Flag indicating whether this is the second run (final codes)
            or not, which determines which column to use for the search query.

    """
    search_terms = [row[JOB_TITLE_COL]]
    if second_run_flag:
        search_terms.append(row[FOLLOWUP_ANSWER_COL])

    search_terms += [row[JOB_DESCRIPTION_COL], row[MERGED_INDUSTRY_DESC_COL]]

    results = one_embedding_handler.search_index_multi(
        search_terms,
    )

    reduced_results = [r.model_dump() for r in results.results]
    return reduced_results


if __name__ == "__main__":
    args = parse_args("STG1")

    df, metadata, start_batch_id = set_up_initial_state(args)

    embedding_handler = _make_embedding_handler(metadata)

    # Clean the Survey Response columns:
    df[JOB_DESCRIPTION_COL] = df[JOB_DESCRIPTION_COL].apply(clean_text)
    df[JOB_TITLE_COL] = df[JOB_TITLE_COL].apply(clean_text)
    df[INDUSTRY_DESCR_COL] = df[INDUSTRY_DESCR_COL].apply(clean_text)
    df[SELF_EMPLOYED_DESC_COL] = df[SELF_EMPLOYED_DESC_COL].apply(clean_text)
    df[MERGED_INDUSTRY_DESC_COL] = df.apply(make_merged_industry_desc, axis=1)
    df[EDUCATION_COL] = df.apply(expand_level_of_education, axis=1)
    df[MERGED_INDUSTRY_DESC_COL] = df[MERGED_INDUSTRY_DESC_COL].apply(clean_text)
    if args.second_run:
        df[FOLLOWUP_ANSWER_COL] = df[FOLLOWUP_ANSWER_COL].apply(clean_text)
    print("Input loaded")

    OUTPUT_COL = OUTPUT_COL_FINAL if args.second_run else OUTPUT_COL_INITIAL
    if OUTPUT_COL not in df:
        df[OUTPUT_COL] = np.empty((len(df), 0)).tolist()

    print("running semantic search...")
    for batch_id, batch in tqdm(
        enumerate(
            np.split(
                df,
                np.arange(
                    start_batch_id * metadata["batch_size"],
                    len(df),
                    metadata["batch_size"],
                ),
            )
        )
    ):
        # A quirk of the np.split approach is that the first batch will contain all
        # of the processed rows so far, so can be skipped
        if batch_id == 0:
            pass
        else:
            df.loc[batch.index, OUTPUT_COL] = batch.apply(
                lambda row: _get_semantic_search_results(
                    row, embedding_handler, args.second_run
                ),
                axis=1,
            )
            persist_results(
                df=df,
                metadata=metadata,
                output_folder=args.output_folder,
                output_shortname=args.output_shortname,
                is_final=False,
                completed_batches=(batch_id + start_batch_id),
            )

    print("semantic search complete")
    print("persisting results...")
    persist_results(
        df=df,
        metadata=metadata,
        output_folder=args.output_folder,
        output_shortname=args.output_shortname,
        is_final=True,
    )
    print("Done!")
