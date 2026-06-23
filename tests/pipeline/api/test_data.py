"""Unit tests for ApiEvaluator data functionality."""

from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from survey_assist_eval.pipeline.api import data as data_module

# turning off black for this file: set max line length to match PEP8 (79 chars)
# for improved readability
# fmt: off

# pylint is not undstanding pytest's fixture handling mechanisms
# pylint: disable=redefined-outer-name


@pytest.fixture
def dummy_input_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Fixture to provide dummy input test data for API evaluation."""
    data = {
        "unique_id": [1, 2, 3],
        "soc2020_job_title": [
            "Data Scientist",
            "Secondary school mathematics teacher",
            "CEO of a tech company",
        ],
        "soc2020_job_description": [
            "Use machine learning etc. inform business decisions.",
            "Write lesson plans, teach, coach...",
            "-8",
        ],
        "sic2007_self_employed": [
            "Contractor",
            "-8",
            "-9",
        ],
        "sic2007_employee": [
            "-8",
            "Secondary education",
            "-9",
        ],
        "clerical_codes": ["1234", "5678", "91011"],
    }
    expected_org_descriptions = [
        "Contractor",
        "Secondary education",
        "",
    ]
    return pd.DataFrame(data), pd.Series(expected_org_descriptions)


@contextmanager
def get_and_prepare_test_data_mocks(df: pd.DataFrame):
    """Context manager to mock get_and_prepare_test_data function."""
    with ExitStack() as stack:
        mock_get_logger = stack.enter_context(
            patch(
                "survey_assist_eval.pipeline.api.data.get_logger",
                return_value=MagicMock()
            )
        )
        mock_pd_read_parquet = stack.enter_context(
            patch(
                "survey_assist_eval.pipeline.api.data.pd.read_parquet",
                return_value=df,
            )
        )
        yield {
            "get_logger": mock_get_logger,
            "pd.read_parquet": mock_pd_read_parquet
        }


class TestBuildOrgDescription:
    """Unit tests for the _build_org_description function.

    Ignoring pylint as access to private method is required for testing
    purposes and is acceptable for test usage.
    """

    def test_build_org_description_with_valid_strings(self):
        """Test that valid strings are concatenated correctly."""
        result = data_module._build_org_description(  # pylint: disable=W0212
            "TestCompany", "Inc.", "Ltd."
        )
        assert result == "TestCompanyInc.Ltd."

    def test_build_org_description_with_unknown_missing_values(self):
        """Test that -9 and -8 values are removed from the end string."""
        result = data_module._build_org_description(  # pylint: disable=W0212
            "TestCompany", "-9", "Ltd.", "-8"
        )
        assert result == "TestCompanyLtd."

    def test_build_org_description_with_non_string_inputs(self):
        """Test that non-string inputs are ignored."""
        result = data_module._build_org_description(  # pylint: disable=W0212
            "TestCompany", 123, None, "Ltd."
        )
        assert result == "TestCompanyLtd."


class TestGetAndPrepareTestData:
    """Unit tests for the get_and_prepare_test_data function."""

    def test_get_and_prepare_test_data(self, dummy_input_test_data):
        """Test that the function processes input data correctly."""
        input_df, expected_org_descs = dummy_input_test_data
        with get_and_prepare_test_data_mocks(input_df) as mocks:
            result_df = data_module.get_and_prepare_test_data(
                "dummy_path"
            )
        # Check that the DataFrame has the expected columns
        expected_columns = data_module.TEST_INPUT_COLUMNS
        assert set(result_df.columns) == set(expected_columns), (
            f"Expected columns: {expected_columns}, "
            f"but got: {set(result_df.columns)}"
        )
        # Check that the org_description column is as expected
        org_descs = result_df["org_description"].tolist()
        assert all(org_descs == expected_org_descs), (
            f"Expected org_description: {expected_org_descs}, "
            f"but got: {org_descs}"
        )
        mocks["pd.read_parquet"].assert_called_once_with(
            "dummy_path", columns=list(data_module.REQUIRED_COLUMNS.keys())
        )

    def test_get_and_prepare_test_data_with_random_sample(
        self, dummy_input_test_data
    ):
        """Test generates a random sample when env var is set."""
        input_df, _ = dummy_input_test_data
        with patch.dict(
            "os.environ", {"API_EVAL_RANDOM_SAMPLE_SIZE": "2"}
        ), get_and_prepare_test_data_mocks(input_df):
            result_df = data_module.get_and_prepare_test_data(
                "dummy_path"
            )
        # Check that the DataFrame has the expected number of rows
        num_rows = len(result_df)
        assert num_rows == 2, (
            f"Expected 2 rows in the sampled DataFrame, but got: {num_rows}"
        )

    def test_get_and_prepare_test_data_with_invalid_random_sample(
        self, dummy_input_test_data
    ):
        """Test that invalid random sample size raises ValueError."""
        input_df, _ = dummy_input_test_data
        with (
            patch.dict(
                "os.environ", {"API_EVAL_RANDOM_SAMPLE_SIZE": "invalid"}
            ),
            get_and_prepare_test_data_mocks(input_df),
            pytest.raises(ValueError, match="Must be an integer")
        ):
            data_module.get_and_prepare_test_data("dummy_path")

    def test_get_and_prepare_test_data_with_large_random_sample(
        self, dummy_input_test_data
    ):
        """Test random sample size larger than dataset raises ValueError."""
        input_df, _ = dummy_input_test_data
        with (
            patch.dict(
                "os.environ", {"API_EVAL_RANDOM_SAMPLE_SIZE": "10"}
            ),
            get_and_prepare_test_data_mocks(input_df),
            pytest.raises(ValueError, match="Can not sample more rows")
        ):
            data_module.get_and_prepare_test_data("dummy_path")

    def test_get_and_prepare_test_data_with_non_positive_random_sample(
        self, dummy_input_test_data
    ):
        """Test non-positive random sample size raises ValueError."""
        input_df, _ = dummy_input_test_data
        with (
            patch.dict(
                "os.environ", {"API_EVAL_RANDOM_SAMPLE_SIZE": "0"}
            ),
            get_and_prepare_test_data_mocks(input_df),
            pytest.raises(ValueError, match="Must be greater than 0")
        ):
            data_module.get_and_prepare_test_data("dummy_path")
