"""Unit tests for ApiEvaluator core functionality."""

import datetime
from contextlib import ExitStack, contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from requests.exceptions import HTTPError

from survey_assist_eval.pipeline.api import core as core_module

# turning off black for this file: set max line length to match PEP8 (79 chars)
# for improved readability
# fmt: off

# pylint is not undstanding pytest's fixture handling mechanisms
# pylint: disable=redefined-outer-name


@pytest.fixture
def default_api_eval_config() -> dict:
    """Fixture for default ApiEvaluatorConfig values."""
    return {
        "gcp_project_id": "test_project",
        "api_gw_url": "https://api-gw-url",
        "api_gw_sa_email": "api-gw-sa@email.com",
        "firestore_db_id": "test_db",
        "firestore_collection_id": "test_collection",
        "execution_id": "test_execution_id",
        "environment": "test",
        "classify_semaphore_limit": 2,
        "lookup_semaphore_limit": 2,
        "log_level": "DEBUG",
    }


@pytest.fixture
def api_eval_config(
    request: pytest.FixtureRequest,
    default_api_eval_config: dict
) -> core_module.ApiEvaluatorConfig:
    """Fixture for ApiEvaluatorConfig with specified classify_type."""
    classify_type = request.param
    config = default_api_eval_config.copy()
    config["classify_type"] = classify_type
    return core_module.ApiEvaluatorConfig(**config)


@pytest.fixture
def api_evaluator_test_data() -> list[dict[str, str]]:
    """Fixture for test data to be used in API evaluation."""
    return [
        {
            "job_title": "Data Scientist",
            "job_description": (
                "Use machine learning, statistical analysis, and data"
                "visualisation to extract insights from data and inform "
                "business decisions."
            ),
            "org_description": (
                "A technology company specialising in e-commerce and cloud"
                "computing services."
            ),
        }
    ]


@contextmanager
def api_eval_init_mocks():
    """Handler for mocking ApiEvaluator init and method calls."""
    with ExitStack() as stack:
        mock_logger = MagicMock()
        mock_job_doc = MagicMock()

        # mock JWT retrieval and validation to bypass auth in tests
        mock_get_jwt = stack.enter_context(
            patch(
                "survey_assist_eval.pipeline.api.core.check_and_refresh_token",
                return_value=(1234, "test_jwt"),
            )
        )
        mock_initialise_firestore = stack.enter_context(
            patch(
                "survey_assist_eval.pipeline.api.core.initialise_firestore",
                return_value=mock_job_doc,
            )
        )
        mock_get_logger = stack.enter_context(
            patch(
                "survey_assist_eval.pipeline.api.core.get_logger",
                return_value=mock_logger,
            )
        )

        yield {
            "logger": mock_logger,
            "job_doc": mock_job_doc,
            "get_jwt": mock_get_jwt,
            "initialise_firestore": mock_initialise_firestore,
            "get_logger": mock_get_logger,
        }


@contextmanager
def api_eval_get_api_config_mocks(
    non_2xx_response: bool = False,
    missing_key: bool = False,
):
    """Handler for mocking external calls in get_api_config."""
    if missing_key:
        mock_json_response = {
            "llm_model": "test-llm-model",
            # "embedding_model" key is intentionally omitted for this test
        }
    else:
        mock_json_response = {
            "llm_model": "test-llm-model",
            "embedding_model": "test-embedding-model",
        }
    with ExitStack() as stack:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_json_response
        mock_request_get = stack.enter_context(
            patch(
                "survey_assist_eval.pipeline.api.core.requests.get",
                return_value=mock_response,
            )
        )
        if non_2xx_response:
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = HTTPError(
                "Internal Server Error"
            )
        yield {
            "request_get": mock_request_get,
            "response": mock_response,
        }


@contextmanager
def api_eval_call_api_lookup_mocks(lookup_not_found: bool = False):
    """Handler for mocking external calls in call_api_endpoint for 'lookup'."""
    with ExitStack() as stack:
        mock_response = MagicMock()
        if lookup_not_found:
            mock_response.status = 404
            mock_response.json = AsyncMock(
                return_value={"error": "not found"}
            )
        else:
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"result": "success"})

        # construct mock aiohttp request, session, and client context managers
        # mimicing the pattern used in ApiEvaluator:
        # async with aiohttp.ClientSession() as session:
        #     async with session.get(url, ...) as response:
        #         data = await response.json()
        # __aenter__ and __aexit__ are used to mock the context manager
        # behavior of both the ClientSession and the session.get() call.
        # expected mock value on enter is required.
        # None on exit is OK as no exception handling required.
        mock_request_cm = AsyncMock()  # context manager for session.get() call
        mock_request_cm.__aenter__.return_value = mock_response
        mock_request_cm.__aexit__.return_value = None
        mock_session = MagicMock()  # cm for aiohttp client session
        mock_session.get.return_value = mock_request_cm
        mock_client_session_cm = AsyncMock()
        mock_client_session_cm.__aenter__.return_value = mock_session
        mock_client_session_cm.__aexit__.return_value = None

        mock_client_session = stack.enter_context(
            patch(
                "survey_assist_eval.pipeline.api.core.aiohttp.ClientSession",
                return_value=mock_client_session_cm,
            )
        )

        yield {
            "client": mock_client_session,
            "session": mock_session,
            "response": mock_response,
        }


@contextmanager
def api_eval_call_api_classify_mocks(mock_api_error: bool = False):
    """Handler mocking external calls in call_api_endpoint for 'classify'."""
    with ExitStack() as stack:
        mock_response = MagicMock()
        if mock_api_error:
            mock_response.status = 500
            mock_response.json = AsyncMock(
                return_value={"details": [{"msg": "Internal Server Error"}]}
            )
        else:
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"result": "success"})

        # construct mock aiohttp request, session, and client context managers
        # mimicing the pattern used in ApiEvaluator:
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(url, ...) as response:
        #         data = await response.json()
        # __aenter__ and __aexit__ are used to mock the context manager
        # behavior of both the ClientSession and the session.post() call.
        # expected mock value on enter is required.
        # None on exit is OK as no exception handling required.
        mock_request_cm = AsyncMock()  # context manager for session.post()
        mock_request_cm.__aenter__.return_value = mock_response
        mock_request_cm.__aexit__.return_value = None
        mock_session = MagicMock()  # cm for aiohttp client session
        mock_session.post.return_value = mock_request_cm
        mock_client_session_cm = AsyncMock()
        mock_client_session_cm.__aenter__.return_value = mock_session
        mock_client_session_cm.__aexit__.return_value = None

        mock_client_session = stack.enter_context(
            patch(
                "survey_assist_eval.pipeline.api.core.aiohttp.ClientSession",
                return_value=mock_client_session_cm,
            )
        )

        yield {
            "client": mock_client_session,
            "session": mock_session,
            "response": mock_response,
        }


class TestApiEvaluatorConfig:
    """Unit tests for ApiEvaluatorConfig dataclass."""

    @pytest.mark.parametrize("classify_type", ["sic", "soc"])
    def test_api_evaluator_config_initialisation(
        self,
        classify_type: str,
        default_api_eval_config: dict,
    ) -> None:
        """Integration test for ApiEvaluatorConfig."""
        config = default_api_eval_config.copy()
        config["classify_type"] = classify_type
        aeconfig = core_module.ApiEvaluatorConfig(**config)

        # assert all values are correctly assigned in the dataclass
        for key, value in config.items():
            assert getattr(aeconfig, key) == value, (
                f"Expected {key} to be {value} but got "
                f"{getattr(aeconfig, key)}"
            )

        # confirm classify type is assigned correctly
        assert aeconfig.classify_type == classify_type, (
            f"Expected classify_type to be {classify_type} but got "
            f"{aeconfig.classify_type}"
        )

        # ensure job_id is generated and is not empty
        assert aeconfig._job_id != "", (    # pylint: disable=W0212
            "Expected _job_id to be generated and not be empty."
        )

    def test_api_evaluator_config_invalid_classify_type(
        self, default_api_eval_config: dict
    ) -> None:
        """Ensure ValueError is raised for invalid classify_type."""
        config = default_api_eval_config.copy()
        config["classify_type"] = "invalid_type"

        with pytest.raises(
            ValueError,
            match="Invalid classify type: invalid_type. Valid classify types",
        ):
            core_module.ApiEvaluatorConfig(**config)

    @pytest.mark.parametrize(
        "semaphore_limit, limit", [("classify", 0), ("lookup", -1)]
    )
    def test_api_evaluator_config_invalid_semaphore_limit(
        self, semaphore_limit: str, limit: int, default_api_eval_config: dict
    ) -> None:
        """Ensure ValueError is raised for invalid semaphore limits."""
        config = default_api_eval_config.copy()
        config["classify_type"] = "sic"  # valid classify type for this test
        full_semaphore_limit = f"{semaphore_limit}_semaphore_limit"
        config[full_semaphore_limit] = limit

        with pytest.raises(
            ValueError,
            match=f"{full_semaphore_limit} must be at least 1. Got {limit}",
        ):
            core_module.ApiEvaluatorConfig(**config)

    def test_api_evaluator_config_invalid_log_level(
        self, default_api_eval_config: dict
    ) -> None:
        """Ensure ValueError is raised for invalid log_level."""
        config = default_api_eval_config.copy()
        config["classify_type"] = "sic"  # valid classify type for this test
        config["log_level"] = "INVALID"

        with pytest.raises(
            ValueError,
            match="Invalid log level: INVALID. Valid log levels are",
        ):
            core_module.ApiEvaluatorConfig(**config)

    # ignoring pyline to unit test the private method _create_job_id
    # pylint: disable=W0212
    def test_api_evaluator_config__create_job_id(self) -> None:
        """Ensure created job IDs are of the expected format."""
        job_id = core_module.ApiEvaluatorConfig._create_job_id()
        uuid, timestamp = job_id.split("-")
        # base57-encoded UUIDs are always 22 chars
        assert len(uuid) == 22, (
            "Expected UUID part of job_id to be 22 characters long."
        )
        try:
            datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S")
        except ValueError:
            pytest.fail(
                f"Timestamp part of job_id '{timestamp}' is not in the "
                "expected format YYYYMMDDHHMMSS."
            )


class TestApiEvaluator:
    """Unit tests for ApiEvaluator class."""

    # need to access private attributes for testing purposes
    # pylint: disable=W0212
    @pytest.mark.parametrize("api_eval_config", ["sic", "soc"], indirect=True)
    def test_api_evaluator_initialisation(
        self, api_eval_config: core_module.ApiEvaluatorConfig
    ) -> None:
        """Ensure ApiEvaluator initialises with correct config and job doc."""
        with api_eval_init_mocks() as init_mock:
            ae = core_module.ApiEvaluator(api_eval_config)

        init_mock["get_jwt"].assert_called_once()
        init_mock["initialise_firestore"].assert_called_once_with(
            api_eval_config.gcp_project_id,
            api_eval_config.firestore_db_id,
            api_eval_config.firestore_collection_id,
            api_eval_config._job_id,
            api_eval_config.log_level,
        )
        assert ae._gcp["firestore_doc"] == init_mock["job_doc"], (
            "Expected _job_doc to be the mock job_doc returned from "
            "initialise_firestore."
        )

    @pytest.mark.parametrize("api_eval_config", ["sic", "soc"], indirect=True)
    def test_api_evaluator_get_api_config_success(
        self, api_eval_config: core_module.ApiEvaluatorConfig
    ) -> None:
        """Test get_api_config method returns successfully."""
        with (
            api_eval_init_mocks() as init_mock,
            api_eval_get_api_config_mocks() as get_config_mock
        ):
            ae = core_module.ApiEvaluator(api_eval_config)
            api_config = ae.get_api_config()

        assert api_config == get_config_mock["response"].json(), (
            "Expected get_api_config to return the JSON from the mocked "
            "response."
        )
        num_jwt_calls = init_mock["get_jwt"].call_count
        assert num_jwt_calls == 2, (
            "Expected jwt check/refresh to be called twice; once during init "
            f"and once during get_api_config. Got {num_jwt_calls}."
        )

    @pytest.mark.parametrize("api_eval_config", ["sic", "soc"], indirect=True)
    def test_api_evaluator_get_api_config_missing_key(
        self, api_eval_config: core_module.ApiEvaluatorConfig
    ) -> None:
        """Test get_api_config raises KeyError when a key is missing.

        This simulates a scenario where the API does not return a required
        configuration paramter and demonstrates that the ApiEvaluator class
        correctly raises a KeyError.
        """
        with (
            api_eval_init_mocks() as init_mock,
            api_eval_get_api_config_mocks(missing_key=True),
        ):
            ae = core_module.ApiEvaluator(api_eval_config)
            with pytest.raises(KeyError, match="embedding_model"):
                ae.get_api_config()

        num_jwt_calls = init_mock["get_jwt"].call_count
        assert num_jwt_calls == 2, (
            "Expected jwt check/refresh to be called twice; once during init "
            f"and once during get_api_config. Got {num_jwt_calls}."
        )

    @pytest.mark.parametrize("api_eval_config", ["sic", "soc"], indirect=True)
    def test_api_evaluator_get_api_config_non_2xx_response(
        self, api_eval_config: core_module.ApiEvaluatorConfig
    ) -> None:
        """Test get_api_config raises HTTPError for non-2XX responses.

        This simulates a scenario where the API returns a non-2XX response and
        demonstrates that the ApiEvaluator class correctly raises an HTTPError.
        """
        with (
            api_eval_init_mocks() as init_mock,
            api_eval_get_api_config_mocks(non_2xx_response=True),
        ):
            ae = core_module.ApiEvaluator(api_eval_config)
            with pytest.raises(HTTPError, match="Internal Server Error"):
                ae.get_api_config()

        num_jwt_calls = init_mock["get_jwt"].call_count
        assert num_jwt_calls == 2, (
            "Expected jwt check/refresh to be called twice; once during init "
            f"and once during get_api_config. Got {num_jwt_calls}."
        )

    @pytest.mark.parametrize("api_eval_config", ["sic", "soc"], indirect=True)
    def test_api_evaluator_call_api_endpoint_lookup_success(
        self,
        api_eval_config: core_module.ApiEvaluatorConfig,
        api_evaluator_test_data: list[dict[str, str]],
    ) -> None:
        """Test call_api_endpoint method for 'lookup' endpoint."""
        with (
            api_eval_init_mocks() as init_mock,
            api_eval_call_api_lookup_mocks() as call_api_mock,
        ):
            ae = core_module.ApiEvaluator(api_eval_config)
            response = ae.call_api_endpoint("lookup", api_evaluator_test_data)

        num_jwt_calls = init_mock["get_jwt"].call_count
        expected_jwt_calls = 1 + len(api_evaluator_test_data)
        assert num_jwt_calls == expected_jwt_calls, (
            f"Expected jwt check/refresh to be called {expected_jwt_calls}; "
            f"num test data inputs + 1 for the init. Got {num_jwt_calls}."
        )

        # ensure correct and expected reponse are returned
        num_test_data_inputs = len(api_evaluator_test_data)
        assert response == [{"result": "success"}] * num_test_data_inputs, (
            "Expected call_api_endpoint to return a list of JSON responses "
            "from the mocked responses."
        )
        assert call_api_mock[
            "session"
        ].get.call_count == num_test_data_inputs, (
            "Expected session.get to be called once for each test data input."
        )
        all_urls = [
            call.args[0]
            for call in call_api_mock["session"].get.call_args_list
        ]
        assert all(  # ensure sic/soc lookup endpoints were used
            url.endswith(f"/{api_eval_config.classify_type}-lookup")
            for url in all_urls
        ), (
            "Expected all session calls to be made to the /lookup endpoint."
        )

        # ensure correct input params were passed to the lookup endpoint
        lookup_desc = call_api_mock[
            "session"
        ].get.call_args_list[0].kwargs.get("params").get("description")
        if api_eval_config.classify_type == "sic":
            assert api_evaluator_test_data[0][
                "org_description"
            ] in lookup_desc, (
                "Expected 'org_description' to be in the params for SIC "
                "evaluation."
            )
        else:
            assert api_evaluator_test_data[0]["job_title"] in lookup_desc, (
                "Expected 'job_title' to be in the params for SOC evaluation."
            )

    @pytest.mark.parametrize("api_eval_config", ["sic", "soc"], indirect=True)
    def test_api_evaluator_call_api_endpoint_lookup_not_found(
        self,
        api_eval_config: core_module.ApiEvaluatorConfig,
        api_evaluator_test_data: list[dict[str, str]],
    ) -> None:
        """Test call_api_endpoint method for 'lookup' endpoint + not found."""
        with (
            api_eval_init_mocks() as init_mock,
            api_eval_call_api_lookup_mocks(
                lookup_not_found=True
            ) as call_api_mock,
        ):
            ae = core_module.ApiEvaluator(api_eval_config)
            response = ae.call_api_endpoint("lookup", api_evaluator_test_data)

        num_jwt_calls = init_mock["get_jwt"].call_count
        expected_jwt_calls = 1 + len(api_evaluator_test_data)
        assert num_jwt_calls == expected_jwt_calls, (
            f"Expected jwt check/refresh to be called {expected_jwt_calls}; "
            f"num test data inputs + 1 for the init. Got {num_jwt_calls}."
        )

        # ensure correct and expected reponse are returned
        num_test_data_inputs = len(api_evaluator_test_data)
        assert response == [None] * num_test_data_inputs, (
            "Expected call_api_endpoint to return None for each test data "
            "input in the response when lookup match is not found."
        )
        assert call_api_mock[
            "session"
        ].get.call_count == num_test_data_inputs, (
            "Expected session.get to be called once for each test data input."
        )
        all_urls = [
            call.args[0]
            for call in call_api_mock["session"].get.call_args_list
        ]
        assert all(  # ensure sic/soc lookup endpoints were used
            url.endswith(f"/{api_eval_config.classify_type}-lookup")
            for url in all_urls
        ), (
            "Expected all session calls to be made to the /lookup endpoint."
        )

    @pytest.mark.parametrize("api_eval_config", ["sic", "soc"], indirect=True)
    def test_api_evaluator_call_api_endpoint_lookup_missing_params(
        self,
        api_eval_config: core_module.ApiEvaluatorConfig,
        api_evaluator_test_data: list[dict[str, str]]
    ) -> None:
        """Test call_api_endpoint raises KeyError for missing input params."""
        with api_eval_init_mocks(), api_eval_call_api_lookup_mocks():
            ae = core_module.ApiEvaluator(api_eval_config)
            test_data = [api_evaluator_test_data[0].copy()]
            if api_eval_config.classify_type == "sic":
                key = "org_description"
                test_data[0].pop(key, None)
            else:
                key = "job_title"
                test_data[0].pop(key, None)

            with pytest.raises(KeyError, match=key):
                ae.call_api_endpoint("lookup", test_data)

    @pytest.mark.parametrize("api_eval_config", ["sic", "soc"], indirect=True)
    def test_api_evaluator_call_api_endpoint_classify_success(
        self,
        api_eval_config: core_module.ApiEvaluatorConfig,
        api_evaluator_test_data: list[dict[str, str]],
    ) -> None:
        """Test call_api_endpoint method for 'classify' endpoint."""
        with (
            api_eval_init_mocks() as init_mock,
            api_eval_call_api_classify_mocks() as call_api_mock,
        ):
            ae = core_module.ApiEvaluator(api_eval_config)
            response = ae.call_api_endpoint(
                "classify", api_evaluator_test_data
            )

        num_jwt_calls = init_mock["get_jwt"].call_count
        expected_jwt_calls = 1 + len(api_evaluator_test_data)
        assert num_jwt_calls == expected_jwt_calls, (
            f"Expected jwt check/refresh to be called {expected_jwt_calls}; "
            f"num test data inputs + 1 for the init. Got {num_jwt_calls}."
        )

        # ensure correct and expected reponse are returned
        num_test_data_inputs = len(api_evaluator_test_data)
        assert response == [{"result": "success"}] * num_test_data_inputs, (
            "Expected call_api_endpoint to return a list of JSON responses "
            "from the mocked responses."
        )
        assert call_api_mock[
            "session"
        ].post.call_count == num_test_data_inputs, (
            "Expected session.post to be called once for each test data input."
        )
        all_urls = [
            call.args[0]
            for call in call_api_mock["session"].post.call_args_list
        ]
        assert all(url.endswith("/classify") for url in all_urls), (
            "Expected all session calls to be made to the /classify endpoint."
        )

        # ensure classify_type appears in the request payload
        classify_type = call_api_mock[
            "session"
        ].post.call_args_list[0].kwargs.get("json").get("type")
        assert classify_type == api_eval_config.classify_type, (
            "Expected classify_type to be included in the request payload."
        )

    @pytest.mark.parametrize("api_eval_config", ["sic", "soc"], indirect=True)
    def test_api_evaluator_call_api_endpoint_classify_api_error(
        self,
        api_eval_config: core_module.ApiEvaluatorConfig,
        api_evaluator_test_data: list[dict[str, str]],
    ) -> None:
        """Test call_api_endpoint raises HTTPError for classify API error.

        This simulates a scenario where the classify API returns a non-2XX
        response and shows that the ApiEvaluator class correctly returns a
        response without raising an exception.
        """
        with (
            api_eval_init_mocks() as init_mock,
            api_eval_call_api_classify_mocks(mock_api_error=True),
        ):
            ae = core_module.ApiEvaluator(api_eval_config)
            response = ae.call_api_endpoint(
                "classify", api_evaluator_test_data
            )

        num_jwt_calls = init_mock["get_jwt"].call_count
        expected_jwt_calls = 1 + len(api_evaluator_test_data)
        assert num_jwt_calls == expected_jwt_calls, (
            f"Expected jwt check/refresh to be called {expected_jwt_calls}; "
            f"num test data inputs + 1 for the init. Got {num_jwt_calls}."
        )

        # ensure correct and expected reponse are returned
        num_test_data_inputs = len(api_evaluator_test_data)
        assert response == [None] * num_test_data_inputs, (
            "Expected call_api_endpoint to return None for each test data "
            "input in the response when lookup match is not found."
        )

    @pytest.mark.parametrize("api_eval_config", ["sic", "soc"], indirect=True)
    def test_api_evaluator_call_api_endpoint_classify_missing_params(
        self,
        api_eval_config: core_module.ApiEvaluatorConfig,
        api_evaluator_test_data: list[dict[str, str]],
    ) -> None:
        """Test call_api_endpoint raises KeyError for missing input params."""
        with api_eval_init_mocks(), api_eval_call_api_classify_mocks():
            ae = core_module.ApiEvaluator(api_eval_config)
            for key in ["job_title", "job_description", "org_description"]:
                test_data = [api_evaluator_test_data[0].copy()]
                test_data[0].pop(key, None)
                with pytest.raises(KeyError, match=key):
                    ae.call_api_endpoint("classify", test_data)

    @pytest.mark.parametrize("api_eval_config", ["sic", "soc"], indirect=True)
    def test_api_evaluator_call_api_endpoint_invalid_endpoint(
        self,
        api_eval_config: core_module.ApiEvaluatorConfig,
        api_evaluator_test_data: list[dict[str, str]],
    ) -> None:
        """Test call_api_endpoint raises ValueError for invalid endpoint."""
        with api_eval_init_mocks():
            ae = core_module.ApiEvaluator(api_eval_config)
            with pytest.raises(
                ValueError, match="Invalid endpoint: invalid_endpoint"
            ):
                ae.call_api_endpoint(
                    "invalid_endpoint", api_evaluator_test_data
                )
