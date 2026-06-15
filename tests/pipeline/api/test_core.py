"""Unit tests for ApiEvaluator core functionality."""

import datetime

import pytest

from survey_assist_eval.pipeline.api import core as db_module

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
        aeconfig = db_module.ApiEvaluatorConfig(**config)

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
            db_module.ApiEvaluatorConfig(**config)

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
            db_module.ApiEvaluatorConfig(**config)

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
            db_module.ApiEvaluatorConfig(**config)

    # ignoring pyline to unit test the private method _create_job_id
    # pylint: disable=W0212
    def test_api_evaluator_config__create_job_id(self) -> None:
        """Ensure created job IDs are of the expected format."""
        job_id = db_module.ApiEvaluatorConfig._create_job_id()
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
