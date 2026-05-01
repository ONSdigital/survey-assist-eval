"""Record latency of a web request to BigQuery."""

import argparse
import datetime
import os
import random
import time

import requests
from dotenv import load_dotenv
from survey_assist_utils import get_logger
from survey_assist_utils.api_token.jwt_utils import (  # pylint: disable=C0411
    generate_jwt,
)

from .export_to_bq import confirm_bq_table_exists, schema_entry, write_to_bq

# Load environment variables from .env file
load_dotenv()


def check_constant(
    const_name: str,
    env_var: str,
    logger_tool,
    required: bool = True,
) -> bool:
    """Check an environment variable, log an error if it's required and not set."""
    if required and not env_var:
        logger_tool.error(f"Required environment variable {const_name} not set")
        raise OSError(f"Required environment variable {const_name} not set")
    logger_tool.debug(f"Environment variable {const_name} set to {env_var}")
    return True


# Set up constants
PROJECT_ID = os.getenv("PROJECT_ID", "")
API_GATEWAY = os.getenv("API_GATEWAY", "")
SA_EMAIL = os.getenv("SA_EMAIL", "")
ENDPOINT = "/v1/survey-assist/classify"
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "")
BQ_TABLE_ID = os.getenv("BQ_TABLE_ID", "")
BQ_TABLE_NAME = f"{PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}"
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
TIMEOUT = os.getenv("TIMEOUT", "60")

# Define schema
schema = [
    schema_entry("test_id", "INTEGER", "REQUIRED", "Unique test ID"),
    schema_entry("test_description", "STRING", "REQUIRED", "Test description."),
    schema_entry("test_timestamp", "DATETIME", "REQUIRED", "Test timestamp (UTC+0)."),
    schema_entry("step", "INTEGER", "REQUIRED", "Step number."),
    schema_entry(
        "total_latency", "FLOAT", "REQUIRED", "Total API call latency (in seconds)"
    ),
    schema_entry(
        "response_time", "FLOAT", "REQUIRED", "Request response time (in seconds)"
    ),
    schema_entry("status_code", "STRING", "REQUIRED", "HTTP status code."),
    schema_entry("job_title", "STRING", "REQUIRED", "Job title."),
    schema_entry("job_description", "STRING", "REQUIRED", "Job description."),
    schema_entry("org_description", "STRING", "REQUIRED", "Organisation description."),
    schema_entry(
        "classified", "INTEGER", "REQUIRED", "Classification success or failure."
    ),
    schema_entry("followup", "STRING", "REQUIRED", "Followup question."),
    schema_entry("code", "STRING", "REQUIRED", "Assigned SIC code."),
]


def prepare_auth(api_gateway_url: str, sa_email: str, logger_tool):
    """Generates a JWT token and forms authorisation headers."""
    logger_tool.debug("Generating api token...")
    try:
        token = generate_jwt(
            audience=api_gateway_url, sa_email=sa_email, expiry_length=3600
        )
    except Exception as e:
        logger_tool.error(f"Error generating api token: {e}")
        raise
    auth_headers = {"Authorization": f"Bearer {token}"}
    return auth_headers


def prepare_payload(fake_data_csv: str, logger_tool, row: int | None = None) -> dict:
    """Reads a random line from a CSV and prepares the request payload."""
    try:
        logger_tool.debug(f"Reading fake responses csv file: '{fake_data_csv}'")
        with open(fake_data_csv, encoding="utf-8") as file:
            lines = file.readlines()

        # The first line is the header
        line_count = len(lines)
        logger_tool.debug(f"Found {line_count} lines in '{fake_data_csv}'.")
        chosen_row_index = (
            row if row else random.randint(1, line_count - 1)  # noqa: S311
        )
        payload_data = lines[chosen_row_index].strip()
        logger_tool.debug(f"Selected line {chosen_row_index} for payload.")
        payload = {
            "llm": "gemini",
            "type": "sic",
            "job_title": "string",
            "job_description": "string",
            "org_description": "string",
            "options": {"sic": {"rephrased": "true"}, "soc": {"rephrased": "false"}},
        }

        try:
            j_title, j_desc, ind_desc = payload_data.split("|")
            payload["job_title"] = j_title
            payload["job_description"] = j_desc
            payload["org_description"] = ind_desc
            logger_tool.info("Successfully prepared payload from CSV.")
            return payload
        except ValueError as e:
            logger_tool.error(
                f"Wrong number of fields in CSV row: '{payload_data}'. Error: {e}"
            )
            raise
    except FileNotFoundError as e:
        logger_tool.error(f"Error: File '{fake_data_csv}' not found. {e}")
        raise
    except Exception as e:
        logger_tool.error(f"An unexpected error occurred in prepare_payload: {e}")
        raise


def step_1(  # pylint: disable=R0917,R0913 # noqa: PLR0913
    test_id: int,
    test_description: str,
    test_timestamp: str,
    logger_tool,
    payload: dict,
    headers: dict,
):
    """Step 1: Measure latency of a web request."""
    logger_tool.debug(f"Step 1: Initiating API call to {API_GATEWAY}{ENDPOINT}")

    request_succeeded = False
    start = time.time()
    try:
        response = requests.post(
            f"{API_GATEWAY}{ENDPOINT}",
            json=payload,
            headers=headers,
            timeout=int(TIMEOUT),
        )
        try:
            response.raise_for_status()
            request_succeeded = True
            response_status_code = response.status_code
            response_time = response.elapsed.total_seconds()
        except requests.exceptions.HTTPError as e:
            logger_tool.error(
                f"Step 1: API call failed with status code {response.status_code}: {e}"
            )
            response_status_code = response.status_code
            response_time = response.elapsed.total_seconds()
    except (requests.exceptions.ReadTimeout, TimeoutError) as e:
        logger_tool.error(f"Step 1: API call timed out: {e}")
        response_status_code = "Timeout"
        response_time = float("inf")
    logger_tool.debug(
        f"Step 1: API call completed with status code {response_status_code}"
    )
    end = time.time()
    latency = end - start
    if request_succeeded:
        result = {
            "test_id": [test_id],
            "test_description": [test_description],
            "test_timestamp": [test_timestamp],
            "step": [1],
            "total_latency": [latency],
            "response_time": [response_time],
            "status_code": [f"{response_status_code}"],
            "job_title": [payload["job_title"]],
            "job_description": [payload["job_description"]],
            "org_description": [payload["org_description"]],
            "classified": [int(response.json()["results"][0]["classified"])],
            "followup": [response.json()["results"][0]["followup"]],
            "code": [f"{response.json()["results"][0]["code"]}"],
        }
    else:
        result = {
            "test_id": [test_id],
            "test_description": [test_description],
            "test_timestamp": [test_timestamp],
            "step": [1],
            "total_latency": [latency],
            "response_time": [response_time],
            "status_code": [f"{response_status_code}"],
            "job_title": [payload["job_title"]],
            "job_description": [payload["job_description"]],
            "org_description": [payload["org_description"]],
            "classified": [0],
            "followup": [""],
            "code": [""],
        }
    logger_tool.debug("Step 1: Preparing to write results to BigQuery.")
    write_to_bq(
        result,
        logger_tool,
        schema,
        gcp_kwargs={
            "project_id": PROJECT_ID,
            "url": API_GATEWAY,
            "table_name": BQ_TABLE_NAME,
        },
    )
    logger_tool.debug("Step 1: Results written to BigQuery.")


def setup_logger():
    """Set up the logger."""
    logger_tool = get_logger("volume_test", level=LOG_LEVEL.upper())
    return logger_tool


def main(args, logger_tool):
    """Run a single latency test and store the result in BigQuery."""
    logger_tool.debug(f"Starting test {args.test_id}: {args.test_description}")
    # wait until the specified sync time
    # Parse sync_time as UTC+0
    sync_time = datetime.datetime.strptime(args.test_sync_timestamp, "%H:%M:%S").time()
    request_payload = prepare_payload(
        "../../data/artificial_data/fake_responses.csv", logger_tool, args.userow
    )
    request_headers = prepare_auth(API_GATEWAY, SA_EMAIL, logger_tool)
    now = datetime.datetime.now(datetime.UTC)
    time_to_wait = (
        datetime.datetime.combine(now.date(), sync_time, tzinfo=datetime.UTC) - now
    ).total_seconds()
    logger_tool.debug(
        f"Waiting for sync time {sync_time} (in {time_to_wait:.2f} " "seconds)..."
    )
    time.sleep(time_to_wait)
    start_time = datetime.datetime.now(datetime.UTC)
    logger_tool.debug("Starting loadrunner run...")
    logger_tool.debug(f"Test ID: {args.test_id}")
    logger_tool.debug(f"Test description: {args.test_description}")
    logger_tool.debug(f"Test start time (UTC+0): {start_time.isoformat()}")
    logger_tool.debug("Running step 1:...")
    step_1(
        args.test_id,
        args.test_description,
        start_time,
        logger_tool,
        request_payload,
        request_headers,
    )
    logger_tool.debug("Step 1 complete.")
    logger_tool.debug("loadrunner run complete.")


def setup_parser():
    """Set up the argument parser."""
    _parser = argparse.ArgumentParser(
        description=("loadrunner: a lightweight utility to execute system loads."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    _parser.add_argument(
        "test_id", type=int, help="The unique identifier for the test to be executed."
    )
    _parser.add_argument(
        "test_description",
        type=str,
        help="The brief description of the test to be executed.",
    )
    _parser.add_argument(
        "test_sync_timestamp",
        type=str,
        help="The synchronization timestamp for the test (UTC+0), in HH:MM:SS format.",
    )
    _parser.add_argument(
        "--userow",
        "-r",
        type=int,
        default=None,
        help="Specify the index of a particular row from the fake data CSV to use for the payload.",
    )
    return _parser


if __name__ == "__main__":
    # set up the parser
    parser = setup_parser()
    # set up the logger
    logger = setup_logger()
    # check the constants
    constants = {
        "PROJECT_ID": PROJECT_ID,
        "API_GATEWAY": API_GATEWAY,
        "SA_EMAIL": SA_EMAIL,
        "ENDPOINT": ENDPOINT,
        "BQ_DATASET_ID": BQ_DATASET_ID,
        "BQ_TABLE_ID": BQ_TABLE_ID,
        "BQ_TABLE_NAME": BQ_TABLE_NAME,
        "LOG_LEVEL": LOG_LEVEL,
        "TIMEOUT": TIMEOUT,
    }
    for name, value in constants.items():
        check_constant(name, value, logger, required=True)
    # confirm the BigQuery table exists
    confirm_bq_table_exists(
        logger,
        gcp_kwargs={
            "project_id": PROJECT_ID,
            "url": API_GATEWAY,
            "table_name": BQ_TABLE_NAME,
        },
    )
    # parse the arguments
    arguments = parser.parse_args()
    # run the main function
    main(arguments, logger)
