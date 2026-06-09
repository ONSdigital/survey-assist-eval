"""Execute end-to-end API evaluation."""

# turning off black for this file: set max line length to match PEP8 (79 chars)
# for improved readability
# fmt: off

# Allow todo comments without triggering errors (keep useful reminders)
# pylint: disable=fixme


import argparse
import os
from typing import Literal

from dotenv import load_dotenv
from survey_assist_utils.logging import get_logger

from survey_assist_eval.pipeline.api import ApiEvaluator, ApiEvaluatorConfig

load_dotenv()
GCP_PROJECT_ID = os.getenv("PROJECT_ID")
API_GW_URL = f"https://{os.getenv('API_GATEWAY')}"
API_GW_SA_EMAIL = os.getenv("SA_EMAIL")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logger = get_logger(__name__, level=LOG_LEVEL)
test_data = [
    {
        "job_title": "Data Scientist",
        "job_description": (
            "Use machine learning, statistical analysis, and data"
            "visualisation to extract insights from data and inform business"
            "decisions."
        ),
        "org_description": (
            "A technology company specialising in e-commerce and cloud"
            "computing services."
        ),
    },
    {
        "job_title": "Secondary school mathematics teacher",
        "job_description": (
            "Teach mathematics to secondary school students, develop lesson "
            "plans, and assessing student progress."
        ),
        "org_description": (
            "A public secondary school providing education to children in the"
            "local community."
        ),
    },
    {
        "job_title": "Nursery assistant",
        "job_description": (
            "Care for plants and assist with gardening tasks. Provide advice"
            " on plant care and maintenance."
        ),
        "org_description": "Garden centre and plant care services.",
    },
    {  # flex soc vector store match (expect no 404)
        "job_title": "Chief executives and senior officials",
        "job_description": "Overlord of a company.",
        "org_description": "Technology and consulting company."
    }
]


def main(classify_type: Literal["sic", "soc"]) -> None:
    """Run end-to-end API evaluation pipeline."""
    # configure pipeline and evaluator
    api_evaluator_cfg = ApiEvaluatorConfig(
        gcp_project_id=GCP_PROJECT_ID,
        api_gw_url=API_GW_URL,
        api_gw_sa_email=API_GW_SA_EMAIL,
        classify_type=classify_type,
        log_level=LOG_LEVEL,
    )
    api_evaluator = ApiEvaluator(api_evaluator_cfg)

    # TODO: update with actual pipeline workflow, currently just mimicking
    # lookup and classify stages for early stage dev only
    lookup_responses = api_evaluator.call_api_endpoint("lookup", test_data)
    classify_responses = api_evaluator.call_api_endpoint("classify", test_data)

    # TODO: remove, used for early dev and debugging only
    if classify_type == "soc":
        logger.info(
            f"Expected 404 from lookup (i.e. none): {lookup_responses[1]}"
        )
        logger.info(
            f"Expected non-404 (i.e. a response): {lookup_responses[3]}"
        )
        logger.info(
            f"Expected classifed: {classify_responses[0]}"
        )
        logger.info(
            f"Expected follow-up Q post classify: {classify_responses[2]}"
        )


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
