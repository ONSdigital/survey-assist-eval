"""Execute end-to-end API evaluation."""

# turning off black for this file: set max line length to match PEP8 (79 chars)
# for improved readability
# fmt: off

# Allow todo comments without triggering errors (keep useful reminders)
# pylint: disable=fixme

# TODO: move main into ApiEvalutor class method when it's fully built.


import argparse
import datetime
import os
from typing import Literal

from dotenv import load_dotenv
from survey_assist_utils.logging import get_logger

from survey_assist_eval.pipeline.api.core import (
    ApiEvaluator,
    ApiEvaluatorConfig,
)
from survey_assist_eval.pipeline.api.data import (
    get_and_prepare_test_data,
    prep_data_for_classify,
    prep_data_for_lookup,
    record_classify_results,
    record_lookup_results,
)

load_dotenv()
GCP_PROJECT_ID = os.getenv("PROJECT_ID")
GCP_TEST_DATA_BUCKET_PATH = os.getenv("EVALUATION_BUCKET_NAME")
API_GW_URL = f"https://{os.getenv('API_GATEWAY')}"
API_GW_SA_EMAIL = os.getenv("SA_EMAIL")
FIRESTORE_DB_ID = os.getenv("API_EVAL_FIRESTORE_DB_ID")
FIRESTORE_COLLECTION_ID = os.getenv("API_EVAL_FIRESTORE_COLLECTION_ID")
ENVIRONMENT = os.getenv("API_EVAL_ENVIRONMENT")
EXECUTION_ID = os.getenv("CLOUD_RUN_EXECUTION")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logger = get_logger("api_e2e", level=LOG_LEVEL)


def main(classify_type: Literal["sic", "soc"]) -> None:
    """Run end-to-end API evaluation pipeline."""
    start_time = datetime.datetime.now(tz=datetime.UTC)

    # configure pipeline and evaluator
    logger.info(
        "Starting end-to-end API evaluation pipeline with classify type: "
        f"{classify_type}..."
    )
    api_evaluator_cfg = ApiEvaluatorConfig(
        gcp_project_id=GCP_PROJECT_ID,
        gcp_test_data_bucket_path=GCP_TEST_DATA_BUCKET_PATH,
        api_gw_url=API_GW_URL,
        api_gw_sa_email=API_GW_SA_EMAIL,
        classify_type=classify_type,
        firestore_db_id=FIRESTORE_DB_ID,
        firestore_collection_id=FIRESTORE_COLLECTION_ID,
        execution_id=EXECUTION_ID,
        environment=ENVIRONMENT,
        log_level=LOG_LEVEL,
    )
    api_evaluator = ApiEvaluator(api_evaluator_cfg)

    # collect API config to record system under test as results metadata
    logger.info("Collecting API configuration for evaluation metadata...")
    api_config = api_evaluator.get_api_config()

    logger.info("Collecting and preparing input data...")
    # temp usage of internal parameter during developmend only, see top TODO
    df = get_and_prepare_test_data(
        api_evaluator_cfg._test_data_file_path,  # pylint: disable=w0212
    )
    logger.info(f"Input data collected and prepared: {len(df)} records.")

    logger.info("Performing lookup calls to API...")
    lookup_ids, lookup_payloads = prep_data_for_lookup(df)
    logger.debug(
        f"Prepared {len(lookup_payloads)} lookup payloads for API calls."
    )
    lookup_responses = api_evaluator.call_api_endpoint(
        "lookup", lookup_payloads
    )
    logger.info("Lookup API calls completed. Analysing lookup responses...")
    df = record_lookup_results(df, lookup_ids, lookup_responses)

    logger.info("Performing classify calls to API...")
    classify_ids, classify_payloads = prep_data_for_classify(df)
    logger.debug(
        f"Prepared {len(classify_payloads)} classify payloads for API calls."
    )
    classify_responses = api_evaluator.call_api_endpoint(
        "classify", classify_payloads
    )
    logger.info(
        "Classify API calls completed. Analysing classify responses..."
    )
    df = record_classify_results(df, classify_ids, classify_responses)

    # TODO: update with results of metrics calculation once implemented
    logger.info("Calculating evaluation metrics...")
    metrics = {}

    # record for evaluation metadata purposes
    end_time = datetime.datetime.now(tz=datetime.UTC)
    duration = (end_time - start_time).total_seconds()
    logger.info(f"API evaluation completed in {duration}s.")

    # write evaluation results to firestore for future analysis
    logger.info("Storing evaluation results in firestore...")
    api_evaluator.store_eval_results(
        start_time,
        end_time,
        duration,
        api_config,
        metrics,
    )
    logger.info("Evaluation results stored successfully.")

    logger.info("End-to-end API evaluation pipeline completed.")


if __name__ == "__main__":
    # allow user to specify classify type (sic or soc) as command line argument
    arg_parser = argparse.ArgumentParser(
        description="Run end-to-end API evaluation.",
        add_help=True
    )
    arg_parser.add_argument(
        "classify_type",
        type=str,
        choices=["sic", "soc"],
        help="The type of classification to perform."
    )
    args = arg_parser.parse_args()

    main(args.classify_type)
