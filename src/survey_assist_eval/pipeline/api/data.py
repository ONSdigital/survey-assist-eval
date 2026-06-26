"""API evaluation data utilities."""

# turning off black for this file: set max line length to match PEP8 (79 chars)
# for improved readability
# fmt: off

import os
import re

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
