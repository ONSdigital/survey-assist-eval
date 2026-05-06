#!/usr/bin/env python3
# pylint: disable=duplicate-code, R0913
"""This script modifies industry description with followup questions and follow up answers,
and persists the results. It reads reloads the output from the previous stage as a DataFrame
object, modifies each row with selected method, overwrites 'merged_industry_desc' column in
the DataFrame with this information, and then saves the results to CSV, parquet, and
JSON metadata files in a user-specified output folder.

See README_evaluation_pipeline.md for more details on how to run.
"""
import numpy as np
import pandas as pd
from industrial_classification_utils.synthetic_responses.synthetic_response_utils import (
    SyntheticResponder,
)
from tqdm import tqdm

from survey_assist_eval.pipeline.shared_components import (
    parse_args,
    persist_results,
    set_up_initial_state,
)

#####################################################
# Constants:
EXTENSION_METHOD = "concatenate"

FOLLOWUP_QUESTION = "followup_question"
FOLLOWUP_ANSWER = "followup_answer"
JOB_DESCRIPTION_COL = "soc2020_job_description"

OUTPUT_COL = "extended_job_desccription"
#####################################################

# Enable progress bar
tqdm.pandas()


def get_rephrased_desc(  # noqa: PLR0913
    row: pd.Series,
    *,
    in_col: str = JOB_DESCRIPTION_COL,
    question_col: str = FOLLOWUP_QUESTION,
    answer_col: str = FOLLOWUP_ANSWER,
    one_sr: SyntheticResponder | None = None,
    method: str = "concatenate",
) -> str:
    """Rephrase variable description with follow up question and follow up answer as a label.

    Args:
        row (pd.Series): A row from the input DataFrame containing the survey responses,
                         and the followup question.
        in_col (str): the column name of the input variable description to be modified.
        question_col (str): the column name of the followup question.
        answer_col (str): the column name of the followup answer.
        one_sr (SyntheticResponder | None): An instance of the SyntheticResponder class,
            used to rephrase the industry description if method is set to 'rephrase'.
        method (str): method to use for rephrasing. Options are 'concatenate' and 'rephrase'.

    Returns: response (str)
    """
    if row[answer_col] == "":
        return row[in_col]

    if method == "concatenate":
        return (
            row[in_col]
            + """
- Followup Question: """
            + row[question_col]
            + """
- Followup Answer: """
            + row[answer_col]
        )

    if method != "rephrase":
        raise ValueError(
            f"Invalid method: {method}. Supported methods are 'concatenate' and 'rephrase'."
        )

    if one_sr is None:
        raise ValueError("one_sr must be provided when method is set to 'rephrase'.")

    return one_sr.rephrase_question_and_id(
        row[in_col],
        row[question_col],
        row[answer_col],
    )[0]


if __name__ == "__main__":
    args = parse_args("STG5")

    df, metadata, start_batch_id = set_up_initial_state(args)

    metadata["extension_method"] = EXTENSION_METHOD

    sr = (
        SyntheticResponder(
            persona=None,
            get_question_function=None,
            model_name=metadata["llm_model_name"],
        )
        if metadata["extension_method"] == "rephrase"
        else None
    )

    print(
        "rephrasing variable description with follow up questions and followup answers..."
    )
    if OUTPUT_COL not in df.columns:
        df[OUTPUT_COL] = ""

    # rephrase new job description
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
        if batch_id == 0:
            pass
        else:
            df.loc[batch.index, OUTPUT_COL] = batch.apply(
                lambda row: get_rephrased_desc(
                    row, one_sr=sr, method=metadata["extension_method"]
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

    print("industry description modification is complete")
    print("persisting results...")
    persist_results(
        df=df,
        metadata=metadata,
        output_folder=args.output_folder,
        output_shortname=args.output_shortname,
        is_final=True,
    )
    print("Done!")
