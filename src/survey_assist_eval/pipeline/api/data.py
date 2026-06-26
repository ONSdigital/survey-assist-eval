"""API evaluation data utilities."""

# turning off black for this file: set max line length to match PEP8 (79 chars)
# for improved readability
# fmt: off

import os
import re
from typing import Any

import numpy as np
import pandas as pd
from survey_assist_utils.logging import get_logger

RANDOM_SEED = 42
REQUIRED_COLUMNS = {
    "unique_id": "unique_id",
    "soc2020_job_title": "job_title",
    "soc2020_job_description": "job_description",
    "sic2007_self_employed": "self_employed",
    "sic2007_employee": "employee",
    "clerical_codes": "clerical_codes",
}
TEST_INPUT_COLUMNS = [
    "unique_id",
    "job_title",
    "job_description",
    "org_description",
    "clerical_codes",
    "api_payload",
]


def _build_org_description(*parts: str) -> str:
    """Concatenate strings to build an organisation description.

    Missing (-8) and Unknown (-9) values are removed from the concatenated
    string. If any input part is not a string it is ignored.

    Args:
        *parts: Strings to concatenate into an organisation description.

    Returns:
        str: Organisation description.
    """
    pattern = re.compile(r"-[89]")  # replace handles missing/unknown vals
    # pylint cant type infer varargs tuples correctly
    return "".join(
        pattern.sub("", p.strip())  # pytlint: disable=E1101
        for p in parts if isinstance(p, str)
    )


def get_and_prepare_test_data(
    filepath: str, log_level: str = "INFO"
) -> pd.DataFrame:
    """Load and prepare test data for API evaluation.

    Args:
        filepath: Path to the test data file in parquet format.
        log_level: Log level for logging within this function, must be one of
            "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".

    Returns:
        pd.DataFrame: Prepared test data for API evaluation.
    """
    logger = get_logger(__name__, level=log_level)

    # read test data
    logger.debug(f"Loading test data from {filepath}...")
    df = pd.read_parquet(filepath, columns=list(REQUIRED_COLUMNS.keys()))

    # generate a random sample only if env vars are set
    # pylint is wrong, it should be capitalised as it's an env var
    RANDOM_SAMPLE_SIZE = os.getenv(    # pylint: disable=C0103
        "API_EVAL_RANDOM_SAMPLE_SIZE", None
    )
    if RANDOM_SAMPLE_SIZE is not None:
        logger.info(
            f"Attempting to generate a random sample of size "
            f"{RANDOM_SAMPLE_SIZE} with seed {RANDOM_SEED}..."
        )
        try:
            random_sample = int(RANDOM_SAMPLE_SIZE)
        except ValueError as e:
            raise ValueError(
                f"Invalid API_EVAL_RANDOM_SAMPLE_SIZE: {RANDOM_SAMPLE_SIZE}. "
                "Must be an integer"
            ) from e
        size_of_df = len(df.index)
        if random_sample > size_of_df:
            raise ValueError(
                f"API_EVAL_RANDOM_SAMPLE_SIZE: {random_sample} is greater than"
                f" the total size of the test dataset: {size_of_df}. Can not "
                "sample more rows than exist in the dataset."
            )
        if random_sample <= 0:
            raise ValueError(
                f"API_EVAL_RANDOM_SAMPLE_SIZE: {random_sample} is not a valid "
                "positive integer. Must be greater than 0."
            )
        df = df.sample(
            n=random_sample,
            random_state=RANDOM_SEED
        ).reset_index(drop=True)
        logger.info(
            f"Random sample of size {random_sample} generated successfully."
        )
    else:
        logger.info(
            "No random sample generated, using full dataset for evaluation."
        )

    # perform common data preparation steps
    df.rename(columns=REQUIRED_COLUMNS, inplace=True)
    df["org_description"] = df.apply(
        lambda row: _build_org_description(
            row["self_employed"], row["employee"]
        ),
        axis=1,
    )
    # convert from list to set (correcting parquet serialisation)
    df["clerical_codes"] = df["clerical_codes"].apply(
        lambda codes: set(codes) if isinstance(codes, np.ndarray) else set()
    )
    # remove -8 values from job_description before passing to API
    df["job_description"] = df["job_description"].apply(
        lambda desc: desc.replace("-8", "") if isinstance(desc, str) else ""
    )

    # contruct the API payload for each record
    df["api_payload"] = df.apply(
        lambda row: {
            "job_title": row["job_title"],
            "job_description": row["job_description"],
            "org_description": row["org_description"],
        },
        axis=1,
    )

    # return only the columns needed for API evaluation
    return df[TEST_INPUT_COLUMNS]


def prep_data_for_lookup(
    df: pd.DataFrame
) -> tuple[list[str], list[dict[str, str]]]:
    """Prepare data for lookup evaluation.

    Args:
        df: DataFrame containing the input test data.

    Returns:
        tuple: List of unique IDs for the lookup_calls and a list of
            dictionaries containing the API payloads for each record.
    """
    lookup_ids = df["unique_id"].tolist()
    lookup_payloads = df["api_payload"].tolist()
    return lookup_ids, lookup_payloads


def record_lookup_results(
    df: pd.DataFrame,
    lookup_ids: list[str],
    lookup_responses: list[dict[str, Any]]
) -> pd.DataFrame:
    """Record the results of the lookup evaluation.

    Notes:
        - This function calls dict keys directly to raise a KeyError if the
            expected keys are not present in the lookup_responses.

    Args:
        df: DataFrame containing the input test data.
        lookup_ids: List of unique IDs for the lookup calls.
        lookup_responses: List of dictionaries containing the API responses
            for each record.

    Returns:
        pd.DataFrame: DataFrame with the recorded lookup results.
    """
    if len(lookup_ids) != len(lookup_responses):
        raise ValueError(
            "Mismatch between number of lookup IDs and lookup responses. "
            "Expected a response for each ID."
        )
    output_df = df.copy()

    # build and unpack key results from the lookup responses
    classified_by_lookup: list[bool | object] = []
    lookup_error = []
    lookup_code = []
    lookup_description = []
    for resp in lookup_responses:
        # case 1: valid response with classification
        if resp != {} and resp is not None:
            classified_by_lookup.append(True)
            lookup_error.append(False)
            lookup_code.append(resp["code"])
            lookup_description.append(resp["description"])
        # case 2: valid response with no classification
        elif resp is None:
            classified_by_lookup.append(False)
            lookup_error.append(False)
            lookup_code.append(pd.NA)
            lookup_description.append(pd.NA)
        # case 3: invalid response (i.e. API error)
        else:
            classified_by_lookup.append(pd.NA)
            lookup_error.append(True)
            lookup_code.append(pd.NA)
            lookup_description.append(pd.NA)

    lookup_df = pd.DataFrame(
        {
            "unique_id": lookup_ids,
            "classified_by_lookup": classified_by_lookup,
            "lookup_code": lookup_code,
            "lookup_description": lookup_description,
            "lookup_error": lookup_error,
        }
    )
    if lookup_df["unique_id"].nunique() != output_df["unique_id"].nunique():
        raise ValueError(
            "Mismatch between number of lookup results and input "
            "data. Expected a lookup result for each input record."
        )

    # join the lookup results back to the original DataFrame on the unique ID
    # no need to fillna values as lookup results are complete for all records
    output_df = output_df.merge(lookup_df, on="unique_id", how="left")
    return output_df


def prep_data_for_classify(
    df: pd.DataFrame,
) -> tuple[list[str], list[dict[str, str]]]:
    """Prepare data for classify evaluation.

    Args:
        df: DataFrame containing the input test data.

    Returns:
        tuple: List of unique IDs for the classify calls and a list of
            dictionaries containing the API payloads for each record.
    """
    if "classified_by_lookup" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'classified_by_lookup' to prep for "
            "classify evaluation."
        )
    # select not classified by lookup and no error during the lookup call
    # (i.e. treat lookup errors as do not continue to classify to prevent
    # misclassification).
    classify_calls = df[
        (~df["classified_by_lookup"]) & (~df["lookup_error"])
    ].copy()
    classify_ids = classify_calls["unique_id"].tolist()
    classify_payloads = classify_calls["api_payload"].tolist()
    return classify_ids, classify_payloads


def record_classify_results(
    df: pd.DataFrame,
    classify_ids: list[str],
    classify_responses: list[dict[str, Any]]
) -> pd.DataFrame:
    """Record the results of the classify evaluation.

    Notes:
        - This function calls dict keys directly to raise a KeyError if the
            expected keys are not present in the classify_responses.

    Args:
        df: DataFrame containing the input test data.
        classify_ids: List of unique IDs for the classify calls.
        classify_responses: List of dictionaries containing the API responses
            for each record.

    Returns:
        pd.DataFrame: DataFrame with the recorded classify results.
    """
    if len(classify_ids) != len(classify_responses):
        raise ValueError(
            "Mismatch between number of classify IDs and classify responses. "
            "Expected a response for each ID."
        )
    output_df = df.copy()

    # build and unpack key results from the classify responses
    classified_by_classify = []
    classify_code = []
    classify_description = []
    classify_followup = []
    classify_candidates = []
    classify_error = []
    for resp in classify_responses:
        # case 1: valid reponse
        if resp != {}:
            result = resp["result"]
            classified_by_classify.append(result["classified"])
            classify_code.append(result["code"])
            classify_description.append(result["description"])
            classify_followup.append(result["followup"])
            classify_candidates.append(result["candidates"])
        # case 2: invalid response = {} (i.e. API error)
        else:
            classified_by_classify.append(pd.NA)
            classify_code.append(pd.NA)
            classify_description.append(pd.NA)
            classify_followup.append(pd.NA)
            classify_candidates.append(pd.NA)
            classify_error.append(True)

    # merge the classify results back to the original DataFrame on the ID
    classify_df = pd.DataFrame(
        {
            "unique_id": classify_ids,
            "classified_by_classify": classified_by_classify,
            "classify_code": classify_code,
            "classify_description": classify_description,
            "classify_followup": classify_followup,
            "classify_candidates": classify_candidates,
            "classify_error": classify_error,
        }
    )
    output_df = output_df.merge(classify_df, on="unique_id", how="left")

    # fill NaN value for the classify columns, equivalent to filling the
    # records that were not classified by the classify API (i.e. lookup only
    # or lookup error)
    fillna_columns = {
        col: pd.NA for col in classify_df.columns if col != "unique_id"
    }
    output_df.fillna(fillna_columns, inplace=True)

    return output_df
