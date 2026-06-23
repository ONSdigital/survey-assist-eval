"""API evaluation data utilities."""

# turning off black for this file: set max line length to match PEP8 (79 chars)
# for improved readability
# fmt: off

import os

import numpy as np
import pandas as pd
from survey_assist_utils.logging import get_logger

RANDOM_SEED = 42
REQUIRED_COLUMNS = {
    "unique_id": "id",
    "soc2020_job_title": "job_title",
    "soc2020_job_description": "job_description",
    "sic2007_self_employed": "self_employed",
    "sic2007_employee": "employee",
    "clerical_codes": "clerical_codes",
}


def _build_org_description(*parts: str) -> str:
    """Concatenate strings to build an organisation description.

    Unknown (-9) values are removed from the concatenated string. If any input
    part is not a string it is ignored.

    Args:
        *parts: Strings to concatenate into an organisation description.

    Returns:
        str: Organisation description.
    """
    return "".join(
        p.strip().replace("-9", "")  # replace handles the unknown values
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
        except TypeError as e:
            raise TypeError(
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

    # return only the columns needed for API evaluation
    return df[
        [
            "id",
            "job_title",
            "job_description",
            "org_description",
            "clerical_codes"
        ]
    ]
