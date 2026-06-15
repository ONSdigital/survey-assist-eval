"""Unit tests for ApiEvaluator core functionality."""

import datetime
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import pytest

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
def api_eval_get_api_config_mocks():
    """Handler for mocking external calls in get_api_config."""
    with ExitStack() as stack:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "llm_model": "test-llm-model",
            "embedding_model": "test-embedding-model",
        }
        mock_request_get = stack.enter_context(
            patch(
                "survey_assist_eval.pipeline.api.core.requests.get",
                return_value=mock_response,
            )
        )
        yield {
            "request_get": mock_request_get,
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
