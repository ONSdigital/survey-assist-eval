"""Survey Assist API Evaluator."""

# turning off black for this file: set max line length to match PEP8 (79 chars)
# for improved readability
# fmt: off

import asyncio
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import aiohttp
from survey_assist_utils.api_token.jwt_utils import check_and_refresh_token
from survey_assist_utils.logging import get_logger
from survey_assist_utils.logging.logging_utils import VALID_LOG_LEVELS


@dataclass(slots=True)
class ApiEvaluatorConfig:
    """Configuration for the API Evaluator Class.

    Parameters
    ----------
    gcp_project_id : str
        The GCP project ID where the API is hosted.
    api_gw_url : str
        The URL of the API Gateway.
    api_gw_sa_email : str
        The service account email to use for authenticating with the API.
    classify_type : Literal["sic", "soc"]
        The type of classification to perform. Must be either "sic" or "soc".
    classify_semaphore_limit : int, optional
        The maximum number of concurrent classify API calls to make. Default is
        2.
    log_level : str, optional
        The logging level to use. Must be one of "DEBUG", "INFO", "WARNING",
        "ERROR", or "CRITICAL". Default is "INFO".

    Raises:
    ------
    ValueError
        - If an invalid classify_type and/or log_level is provided.
        - If classify_semaphore_limit is less than 1.
    """

    gcp_project_id: str
    api_gw_url: str
    api_gw_sa_email: str
    classify_type: str
    classify_semaphore_limit: int = 2
    log_level: str = "INFO"

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

        if self.classify_semaphore_limit < 1:
            raise ValueError("classify_semaphore_limit must be at least 1.")


class ApiEvaluator:
    """Evalutate the Survey Assist API classification performance.

    Parameters
    ----------
    config : ApiEvaluatorConfig
        The configuration for the API Evaluator.

    Raises:
    ------
    ValueError
        Invalid classify_type and/or log_level is provided.
    """

    _API_BASE_ENDPOINT: ClassVar[str] = "/v1/survey-assist"
    _CLASSIFY_ENDPOINT: ClassVar[str] = "/classify"
    _CLASSIFY_SEMAPHORE_LIMIT: ClassVar[int] = 2
    _VALID_CLASSIFY_TYPES: ClassVar[list[str]] = ["sic", "soc"]

    def __init__(self, config: ApiEvaluatorConfig) -> None:
        # setup and pass through inputs
        self._gcp_project_id = config.gcp_project_id
        self._api_gw_url = config.api_gw_url
        self._api_gw_sa_email = config.api_gw_sa_email
        self._classify_type = config.classify_type
        self._logger = get_logger(__name__, level=config.log_level)
        self._classify: dict[str, Any] = {}
        self._classify["semaphore"] = asyncio.Semaphore(
            config.classify_semaphore_limit
        )
        self._classify["endpoint"] = (
            self._api_gw_url
            + self._API_BASE_ENDPOINT
            + self._CLASSIFY_ENDPOINT
        )

        # generate initial JWT token and lock
        self._jwt: dict[str, Any] = {}
        self._jwt["start_time"], self._jwt["token"] = check_and_refresh_token(
            0, "", self._api_gw_url, self._api_gw_sa_email
        )
        self._jwt["lock"] = asyncio.Lock()

    def _build_classify_payload(self, params: dict) -> dict:
        """Construct the classify payload.

        Raises:
        ------
        KeyError
            Params must include "job_title", "job_description" and
            "org_description" keys.
        """
        return {
            "llm": "gemini",
            "type": self._classify_type,
            "job_title": params["job_title"],
            "job_description": params["job_description"],
            "org_description": params["org_description"],
        }

    async def _call_api_endpoint(
        self,
        endpoint: Literal["classify"],
        session: aiohttp.ClientSession,
        params: dict,
    ) -> dict | None:
        """Call the specified API endpoint with the given parameters.

        Parameters
        ----------
        endpoint : Literal["classify"]
            The API endpoint to call.
        session : aiohttp.ClientSession
            The aiohttp session to use for making the API call.
        params : dict
            The parameters to include in the API request. Must include keys
            "job_title", "job_description" and "org_description" for the
            "classify" endpoint.

        Returns:
        -------
        dict | None
            The JSON response from the API call, or None if the call failed.

        Raises:
        ------
        ValueError
            If an invalid endpoint is specified.
        """
        request_kwargs: dict[str, Any] = {}
        match endpoint:
            case "classify":
                endpoint_url = self._classify["endpoint"]
                semaphore = self._classify["semaphore"]
                session_method = session.post
                request_kwargs["json"] = self._build_classify_payload(params)
            case _:
                raise ValueError(f"Invalid endpoint: {endpoint}")

        async with semaphore:
            self._logger.debug(
                f"Calling {endpoint} endpoint with job_title: "
                f"{params.get('job_title', 'N/A')}"
            )
            # Ensure JWT is valid before call, using lock to prevent concurrent
            # refreshes across async calls
            async with self._jwt["lock"]:
                self._jwt["start_time"], self._jwt["token"] = (
                    check_and_refresh_token(
                        self._jwt["start_time"],
                        self._jwt["token"],
                        self._api_gw_url,
                        self._api_gw_sa_email,
                    )
                )
            request_kwargs["headers"] = {
                "Authorization": f"Bearer {self._jwt['token']}"
            }
            async with session_method(
                endpoint_url,
                **request_kwargs,
            ) as response:
                if response.status != 200:  # noqa: PLR2004 (200 err code OK)
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
        endpoint: Literal["classify"],
        data: list[dict[str, str]],
    ) -> list[dict | None]:
        """Call an API endpoint asynchonously.

        Parameters
        ----------
        endpoint : Literal["classify"]
            The endpoint to make the call to.
        data : list[dict[str, str]]
            A list of dictionaries, whose key-value pairs represent the params
            to pass to the endpoint.

        Returns:
        -------
        list[dict | None]
            The API responses.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._call_api_endpoint(endpoint, session, params)
                for params in data
            ]
            return await asyncio.gather(*tasks)

    def call_api_endpoint(
        self, endpoint: Literal["classify"], data: list[dict[str, str]]
    ) -> list[dict | None]:
        """Batch call an API endpoint synchronously.

        Parameters
        ----------
        endpoint : Literal["classify"]
            The endpoint to make the call to.
        data : list[dict[str, str]]
            A list of dictionaries, whose key-value pairs represent the params
            to pass to the endpoint.

        Returns:
        -------
        list[dict | None]
            The API responses.
        """
        return asyncio.run(self.call_api_endpoint_async(endpoint, data))
