"""Core Survey Assist API Evaluator."""

# turning off black for this file: set max line length to match PEP8 (79 chars)
# for improved readability
# fmt: off

import asyncio
import datetime
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import aiohttp
import requests
import shortuuid
from survey_assist_utils.api_token.jwt_utils import check_and_refresh_token
from survey_assist_utils.logging import get_logger
from survey_assist_utils.logging.logging_utils import VALID_LOG_LEVELS

from survey_assist_eval.pipeline.api.db import (
    eval_results_to_firestore,
    initialise_firestore,
)

HTTP_STATUS_OK = 200
HTTP_STATUS_NOT_FOUND = 404


# ignore pylint as parameters are required to configure API evaluation
@dataclass(slots=True)
class ApiEvaluatorConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for the API Evaluator Class.

    Also creates a UUID to uniquely identify the evaluation job. This is a
    combination of a timestamp and random string, to ensure uniqueness while
    also maintaining human readability for easier debugging and traceability in
    Firestore.

    Args:
        gcp_project_id: The GCP project ID where the API is hosted.
        api_gw_url: The URL of the API Gateway.
        api_gw_sa_email: The service account email to use for authenticating
            with the API.
        classify_type: The type of classification to perform. Must be either
            "sic" or "soc".
        firestore_db_id: The Firestore database ID to use for storing
            evaluation results.
        firestore_collection_id: The Firestore collection ID to use for storing
            evaluation results.
        execution_id: A unique identifier for the evaluation job, typically
            a UUID set by the caller. Can be None if not required.
        environment: The environment in which the evaluation is being run, e.g.
            "sandbox", "dev", "preprod", "prod" etc.
        classify_semaphore_limit: The maximum number of concurrent classify
            API calls to make. Defaults to 2.
        lookup_semaphore_limit: The maximum number of concurrent lookup API
            calls to make. Defaults to 5.
        log_level: The logging level to use. Must be one of "DEBUG", "INFO",
            "WARNING", "ERROR", or "CRITICAL". Defaults to "INFO".

    Raises:
        ValueError: If an invalid classify_type and/or log_level is provided,
            or if any *_semaphore_limit argument is less than 1.
    """

    gcp_project_id: str
    api_gw_url: str
    api_gw_sa_email: str
    classify_type: str
    firestore_db_id: str
    firestore_collection_id: str
    execution_id: str | None
    environment: str
    classify_semaphore_limit: int = 2
    lookup_semaphore_limit: int = 5
    log_level: str = "INFO"
    _job_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        normalised = self.classify_type.lower()
        if normalised not in ("sic", "soc"):
            raise ValueError(
                f"Invalid classify type: {self.classify_type}. "
                "Valid classify types are: ['sic', 'soc']"
            )
        self.classify_type = normalised  # keep normalised value

        if self.log_level not in VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log level: {self.log_level}. "
                f"Valid log levels are: {VALID_LOG_LEVELS}"
            )

        # ensure semaphore limits are at least 1
        for attr_name in (
            "classify_semaphore_limit", "lookup_semaphore_limit"
        ):
            attr_value = getattr(self, attr_name)
            if attr_value < 1:
                raise ValueError(
                    f"{attr_name} must be at least 1. Got {attr_value}."
                )

        # create unique job ID for this evaluation run
        self._job_id = self._create_job_id()

    @staticmethod
    def _create_job_id() -> str:
        """Create a unique job ID using a timestamp and random string."""
        uuid = shortuuid.uuid()
        timestamp = datetime.datetime.now(tz=datetime.UTC).strftime(
            "%Y%m%d%H%M%S"
        )
        return f"{uuid}-{timestamp}"


class ApiEvaluator:
    """Evalutate the Survey Assist API classification performance.

    Args:
        config: The configuration for the API Evaluator.

    Raises:
        ValueError: If an invalid classify_type and/or log_level is provided.
    """

    _API_BASE_ENDPOINT: ClassVar[str] = "/v1/survey-assist"
    _CLASSIFY_ENDPOINT: ClassVar[str] = "/classify"
    _CONFIG_ENDPOINT: ClassVar[str] = "/config"
    _VALID_CLASSIFY_TYPES: ClassVar[list[str]] = ["sic", "soc"]

    def __init__(self, config: ApiEvaluatorConfig) -> None:
        # setup and pass through inputs
        self._gcp: dict[str, Any] = {}
        self._gcp["project_id"] = config.gcp_project_id
        self._gcp["firestore_db_id"] = config.firestore_db_id
        self._gcp["firestore_collection_id"] = config.firestore_collection_id
        self._gcp["environment"] = config.environment
        self._gcp["execution_id"] = config.execution_id
        self._gcp["job_id"] = config._job_id
        self._api: dict[str, Any] = {}
        self._api["gw_url"] = config.api_gw_url
        self._api["gw_sa_email"] = config.api_gw_sa_email
        self._classify_type = config.classify_type
        self._logger = get_logger(__name__, level=config.log_level)
        self._classify: dict[str, Any] = {}
        self._classify["semaphore"] = asyncio.Semaphore(
            config.classify_semaphore_limit
        )
        self._classify["endpoint"] = (
            self._api["gw_url"]
            + self._API_BASE_ENDPOINT
            + self._CLASSIFY_ENDPOINT
        )
        self._lookup: dict[str, Any] = {}
        self._lookup["semaphore"] = asyncio.Semaphore(
            config.lookup_semaphore_limit
        )
        self._lookup["endpoint"] = (
            self._api["gw_url"]
            + self._API_BASE_ENDPOINT
            + f"/{self._classify_type}-lookup"
        )

        # generate initial JWT token and lock
        self._jwt: dict[str, Any] = {}
        self._jwt["start_time"], self._jwt["token"] = check_and_refresh_token(
            0, "", self._api["gw_url"], self._api["gw_sa_email"]
        )
        self._jwt["lock"] = asyncio.Lock()

        # check and collect firestore doc ref for eval results storage
        self._gcp["firestore_doc"] = initialise_firestore(
            config.gcp_project_id,
            config.firestore_db_id,
            config.firestore_collection_id,
            config._job_id,
            config.log_level,
        )

    def _build_classify_payload(self, params: dict) -> dict:
        """Construct the classify payload.

        Raises:
            KeyError: Params must include "job_title", "job_description" and
                "org_description" keys.
        """
        return {
            "llm": "gemini",
            "type": self._classify_type,
            "job_title": params["job_title"],
            "job_description": params["job_description"],
            "org_description": params["org_description"],
        }

    def _build_lookup_payload(self, params: dict) -> dict:
        """Construct the lookup payload.

        Assigns the "job_title" as the lookup description.

        Raises:
            KeyError: If params does not include "job_title" key when running
                a SOC evaluation, or "org_description" key when running a SIC
                evaluation.
        """
        description = (
            "job_title"
            if self._classify_type == "soc" else "org_description"
        )
        return {"description": params[description]}

    async def _call_api_endpoint(
        self,
        endpoint: Literal["classify", "lookup"],
        session: aiohttp.ClientSession,
        params: dict,
    ) -> dict | None:
        """Call the specified API endpoint with the given parameters.

        Args:
            endpoint: The API endpoint to call.
            session: The aiohttp session to use for making the API call.
            params: The parameters to include in the API request. Must include
                keys "job_title", "job_description" and "org_description" for
                the "classify" and "lookup" endpoints.

        Returns:
            The JSON response from the API call, or None if the call failed.

        Raises:
            ValueError: If an invalid endpoint is specified.
        """
        request_kwargs: dict[str, Any] = {}
        match endpoint:
            case "classify":
                endpoint_url = self._classify["endpoint"]
                semaphore = self._classify["semaphore"]
                session_method = session.post
                request_kwargs["json"] = self._build_classify_payload(params)
                request_kwargs["timeout"] = 30
            case "lookup":
                endpoint_url = self._lookup["endpoint"]
                semaphore = self._lookup["semaphore"]
                session_method = session.get
                request_kwargs["params"] = self._build_lookup_payload(params)
                request_kwargs["timeout"] = 10
            case _:
                raise ValueError(f"Invalid endpoint: {endpoint}")

        async with semaphore:
            self._logger.debug(
                f"Calling {endpoint_url} endpoint for job_title: "
                f"{params.get('job_title', 'N/A')}"
            )
            # Ensure JWT is valid before call, using lock to prevent concurrent
            # refreshes across async calls
            async with self._jwt["lock"]:
                self._jwt["start_time"], self._jwt["token"] = (
                    check_and_refresh_token(
                        self._jwt["start_time"],
                        self._jwt["token"],
                        self._api["gw_url"],
                        self._api["gw_sa_email"],
                    )
                )
            request_kwargs["headers"] = {
                "Authorization": f"Bearer {self._jwt['token']}"
            }
            async with session_method(
                endpoint_url,
                **request_kwargs,
            ) as response:
                if response.status != HTTP_STATUS_OK:
                    # handle expected 404 for lookup endpoint as "not found"
                    # rather than error
                    if (
                        response.status == HTTP_STATUS_NOT_FOUND
                        and endpoint == "lookup"
                    ):
                        self._logger.debug(
                            f"{endpoint} API call returned 404 indicating no "
                            f"lookup match for {params.get('job_title')}"
                        )
                        return None
                    try:
                        error_payload = await response.json()
                        detail = error_payload.get("detail", [])
                        if isinstance(detail, list) and detail:
                            error_msg = detail[0].get("msg", str(detail[0]))
                        elif isinstance(detail, dict):
                            error_msg = detail.get("msg", str(detail))
                        else:
                            error_msg = str(error_payload)
                    except aiohttp.ContentTypeError:
                        error_msg = await response.text()
                    self._logger.error(
                        f"{endpoint} API call failed with status code "
                        f"{response.status}. Message: "
                        f"{error_msg}"
                    )
                    return None
                return await response.json()

    async def call_api_endpoint_async(
        self,
        endpoint: Literal["classify", "lookup"],
        data: list[dict[str, str]],
    ) -> list[dict | None]:
        """Call an API endpoint asynchonously.

        Args:
            endpoint: The endpoint to make the call to.
            data: A list of dictionaries, whose key-value pairs represent the
                params to pass to the endpoint.

        Returns:
            A list of API responses.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._call_api_endpoint(endpoint, session, params)
                for params in data
            ]
            return await asyncio.gather(*tasks)

    def call_api_endpoint(
        self,
        endpoint: Literal["classify", "lookup"],
        data: list[dict[str, str]]
    ) -> list[dict | None]:
        """Batch call an API endpoint synchronously.

        Args:
            endpoint: The endpoint to make the call to.
            data: A list of dictionaries, whose key-value pairs represent the
                params to pass to the endpoint.

        Returns:
            A list of API responses.
        """
        return asyncio.run(self.call_api_endpoint_async(endpoint, data))

    def get_api_config(self) -> dict:
        """Get the current API configuration information."""
        self._logger.debug("Retrieving API configuration...")
        config_url = (
            self._api["gw_url"]
            + self._API_BASE_ENDPOINT
            + self._CONFIG_ENDPOINT
        )
        self._jwt["start_time"], self._jwt["token"] = check_and_refresh_token(
            self._jwt["start_time"],
            self._jwt["token"],
            self._api["gw_url"],
            self._api["gw_sa_email"],
        )
        headers = {"Authorization": f"Bearer {self._jwt['token']}"}
        response = requests.get(config_url, headers=headers, timeout=10)
        response.raise_for_status()
        self._logger.debug("API configuration retrieved successfully.")
        config = response.json()
        return {
            "llm_model": config["llm_model"],
            "embedding_model": config["embedding_model"],
        }

    # ignoring pylint as it thinks there are 6 args, but only 5 are passed in
    def store_eval_results(  # pylint: disable=too-many-arguments,R0917
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        duration_s: float,
        api_config: dict,
        metrics: dict,
    ) -> None:
        """Collect and store the results of an evaluation.

        Args:
            start_time: The start time of the evaluation.
            end_time: The end time of the evaluation.
            duration_s: The duration of the evaluation in seconds.
            api_config: The API configuration used for the evaluation.
            metrics: The evaluation metrics.
        """
        eval_results = {
            "gcp_project_id": self._gcp["project_id"],
            "environment": self._gcp["environment"],
            "execution_id": self._gcp["execution_id"],
            "classify_type": self._classify_type,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": duration_s,
            "api_config": api_config,
            "metrics": metrics,
        }
        self._logger.debug(f"Evaluation results: {eval_results}")

        eval_results_to_firestore(
            self._gcp["firestore_doc"],
            self._gcp["job_id"],
            eval_results,
            self._logger.level,
        )
