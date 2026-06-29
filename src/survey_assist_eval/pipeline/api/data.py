"""API evaluation data utilities."""

# turning off black for this file: set max line length to match PEP8 (79 chars)
# for improved readability
# fmt: off

import os
import re
from typing import Any

# ignoring E0611 (name not in module) for NAType import, as it is a valid
# import from pandas._libs.missing but pylint does not recognise it since
# static analysis of C extensions is not supported.
import numpy as np
import pandas as pd
from pandas._libs.missing import NAType  # pylint: disable=E0611
from survey_assist_utils.logging import get_logger

_RANDOM_SEED = 42
_REQUIRED_FIELDS_MAP = {
    "unique_id": "unique_id",
    "soc2020_job_title": "job_title",
    "soc2020_job_description": "job_description",
    "sic2007_self_employed": "self_employed",
    "sic2007_employee": "employee",
    "clerical_codes": "clerical_codes",
}
_TEST_INPUT_FIELDS = [
    "unique_id",
    "job_title",
    "job_description",
    "org_description",
    "clerical_codes",
    "api_payload",
]
_LOOKUP_RESULT_FIELDS = [
    "lookup_classified",
    "lookup_error",
    "lookup_code",
    "lookup_description",
]
_CLASSIFY_RESULT_FIELDS = [
    "classify_classified",
    "classify_error",
    "classify_code",
    "classify_description",
    "classify_followup",
    "classify_candidates",
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
    df = pd.read_parquet(filepath, columns=list(_REQUIRED_FIELDS_MAP.keys()))

    # generate a random sample only if env vars are set
    # pylint is wrong, it should be capitalised as it's an env var
    RANDOM_SAMPLE_SIZE = os.getenv(    # pylint: disable=C0103
        "API_EVAL_RANDOM_SAMPLE_SIZE", None
    )
    if RANDOM_SAMPLE_SIZE is not None:
        logger.info(
            f"Attempting to generate a random sample of size "
            f"{RANDOM_SAMPLE_SIZE} with seed {_RANDOM_SEED}..."
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
            random_state=_RANDOM_SEED
        ).reset_index(drop=True)
        logger.info(
            f"Random sample of size {random_sample} generated successfully."
        )
    else:
        logger.info(
            "No random sample generated, using full dataset for evaluation."
        )

    # perform common data preparation steps
    df.rename(columns=_REQUIRED_FIELDS_MAP, inplace=True)
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
    return df[_TEST_INPUT_FIELDS]


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


def _lookup_resp_handler(
    resp: dict[str, Any] | None
) -> dict[str, bool | str | NAType]:
    """Handle the response from the lookup API.

    Args:
        resp: The lookup API response from `ApiEvaluator` (dict or None).

    Returns:
        dict: A dictionary containing:
            - classified_by_lookup: True if the record was classified by the
                lookup API, False if not classified, or pd.NA if there was an
                error during the lookup call.
            - lookup_error: True if there was an error during the lookup call,
                False otherwise.
            - lookup_code: The code returned by the lookup API, or pd.NA if
                there was an error during the lookup call.
            - lookup_description: The description returned by the lookup API,
                if a lookup match was found, else pd.NA.
    """
    values: tuple[bool | str | NAType, bool, str | NAType, str | NAType]
    # case 1: valid response and classified by lookup
    if resp != {} and resp is not None:
        values = (True, False, resp["code"], resp["description"])
    # case 2: valid response but not found by lookup
    elif resp is None:
        values = (False, False, pd.NA, pd.NA)
    # case 3: invalid response (i.e. {}) = API error
    else:
        values = (pd.NA, True, pd.NA, pd.NA)
    return dict(zip(_LOOKUP_RESULT_FIELDS, values, strict=True))


def record_lookup_results(
    df: pd.DataFrame,
    lookup_ids: list[str],
    lookup_responses: list[dict[str, Any] | None]
) -> pd.DataFrame:
    """Record the results of the lookup evaluation.

    Notes:
        - This function calls dict keys directly to raise a KeyError if the
            expected keys are not present in the lookup_responses.

    Args:
        df: DataFrame containing the input test data.
        lookup_ids: List of unique IDs for the lookup calls.
        lookup_responses: List of dictionaries/None containing the API
            responses for each record.

    Returns:
        pd.DataFrame: DataFrame with the recorded lookup results.
    """
    if len(lookup_responses) == 0:
        raise ValueError("No lookup responses provided")
    if len(lookup_ids) != len(lookup_responses):
        raise ValueError(
            "Mismatch between number of lookup IDs and lookup responses. "
            "Expected a response for each ID."
        )
    output_df = df.copy()

    # build and unpack key results from the lookup responses
    rows = [_lookup_resp_handler(resp) for resp in lookup_responses]
    lookup_df = pd.DataFrame.from_records(rows, columns=_LOOKUP_RESULT_FIELDS)
    lookup_df.insert(0, "unique_id", pd.Series(lookup_ids))

    # join the lookup results back to the original DataFrame on the unique ID
    # validate a one-to-one merge to ensure that each unique_id has a
    # single lookup result (just another sense check)
    # no need to fillna values as lookup results are complete for all records
    output_df = output_df.merge(
        lookup_df, on="unique_id", how="left", validate="one_to_one"
    )
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
    if "lookup_classified" not in df.columns:
        raise KeyError(
            "DataFrame must contain 'lookup_classified' to prep for "
            "classify evaluation."
        )
    # select not classified by lookup and no error during the lookup call
    # (i.e. treat lookup errors as do not continue to classify to prevent
    # misclassification).
    classify_calls = df[
        df["lookup_classified"].eq(False)
        & df["lookup_error"].eq(False)
    ].copy()
    classify_ids = classify_calls["unique_id"].tolist()
    classify_payloads = classify_calls["api_payload"].tolist()
    return classify_ids, classify_payloads


def _classify_resp_handler(
    resp: dict[str, Any]
) -> dict[str, bool | str | NAType | list[dict[str, Any]]]:
    """Handle the response from the classify API.

    Args:
        resp: The classify API response from `ApiEvaluator` (dict or None).

    Returns:
        dict: A dictionary containing the processed classify results.
    """
    values: tuple[
        str | NAType,
        bool,
        str | NAType,
        str | NAType,
        str | NAType,
        list[dict[str, Any]] | NAType
    ]
    if resp != {}:
        result = resp["results"][0]
        values = (
            result["classified"],
            False,
            result["code"],
            result["description"],
            result["followup"],
            result["candidates"],
        )
    else:
        values = (pd.NA, True, pd.NA, pd.NA, pd.NA, pd.NA)
    return dict(zip(_CLASSIFY_RESULT_FIELDS, values, strict=True))


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
    rows = [_classify_resp_handler(resp) for resp in classify_responses]
    classify_df = pd.DataFrame.from_records(
        rows, columns=_CLASSIFY_RESULT_FIELDS
    )
    classify_df.insert(0, "unique_id", pd.Series(classify_ids))

    # join the classify results back to the original DataFrame on the unique ID
    output_df = output_df.merge(
        classify_df, on="unique_id", how="left", validate="one_to_one"
    )

    # fill NaN value for the classify columns, equivalent to filling the
    # records that were not classified by the classify API (i.e. lookup only
    # or lookup error)
    fillna_columns = {
        col: pd.NA for col in classify_df.columns if col != "unique_id"
    }
    output_df.fillna(fillna_columns, inplace=True)

    return output_df
