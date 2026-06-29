"""Unit tests for ApiEvaluator data functionality."""

from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

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


@pytest.fixture
def dummy_data_lookup_prep() -> pd.DataFrame:
    """Dummy data for lookup call preparation."""
    data = {
        "unique_id": [1, 2, 3, 4],
        "job_title": ["job_1", "job_2", "job_3", "job_4"],
        "job_description": ["desc_1", "desc_2", "desc_3", "desc_4"],
        "org_description": ["org_1", "org_2", "org_3", "org_4"],
        "clerical_codes": ["code_1", "code_2", "code_3", "code_4"],
        "api_payload": [
            {
                "job_title": "job_1",
                "job_description": "desc_1",
                "org_description": "org_1",
            },
            {
                "job_title": "job_2",
                "job_description": "desc_2",
                "org_description": "org_2",
            },
            {
                "job_title": "job_3",
                "job_description": "desc_3",
                "org_description": "org_3",
            },
            {
                "job_title": "job_4",
                "job_description": "desc_4",
                "org_description": "org_4",
            },
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def dummy_lookup_results(dummy_data_lookup_prep) -> pd.DataFrame:
    """Dummy lookup results for testing the record_lookup_results function."""
    # dummpy lookup API responses
    responses = [
        {"code": "1", "description": "description 1"},
        None,  # simulating no lookup result
        {},  # simulating an API failure
        {"code": "4", "description": "description 4"},
    ]

    # add expected lookup results to the dummy data for comparison in tests
    expected_results = dummy_data_lookup_prep.copy()
    expected_results["lookup_classified"] = [True, False, pd.NA, True]
    expected_results["lookup_error"] = [False, False, True, False]
    expected_results["lookup_code"] = ["1", pd.NA, pd.NA, "4"]
    expected_results["lookup_description"] = [
        "description 1", pd.NA, pd.NA, "description 4"
    ]

    return responses, expected_results


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

    # pylint: disable=W0212
    def test_get_and_prepare_test_data(self, dummy_input_test_data):
        """Test that the function processes input data correctly."""
        input_df, expected_org_descs = dummy_input_test_data
        with (
            patch.dict("os.environ", {}, clear=True),
            get_and_prepare_test_data_mocks(input_df) as mocks
        ):
            result_df = data_module.get_and_prepare_test_data(
                "dummy_path"
            )
        # Check that the DataFrame has the expected columns
        expected_columns = data_module._TEST_INPUT_FIELDS
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
            "dummy_path", columns=list(data_module._REQUIRED_FIELDS_MAP.keys())
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


# Allowing a single test class method to improve test suite organisation
# pylint: disable=R0903
class TestPrepDataForLookup:
    """Unit tests for the prep_data_for_lookup function."""

    def test_prep_data_for_lookup(self, dummy_data_lookup_prep):
        """Test that the function prepares data correctly for lookup."""
        input_df = dummy_data_lookup_prep
        num_rows = len(input_df.index)
        results = data_module.prep_data_for_lookup(input_df)

        for result in results:
            assert isinstance(result, list), (
                f"Expected each result to be a list, but got: {type(result)}"
            )
            assert len(result) == num_rows, (
                f"Expected each result list to have length {num_rows}, "
                f"but got: {len(result)}"
            )

        ids, payloads = results
        assert ids == input_df["unique_id"].tolist(), (
            f"Expected IDs: {input_df['unique_id'].tolist()}, "
            f"but got: {ids}"
        )
        assert all(isinstance(payload, dict) for payload in payloads), (
            "Expected each payload to be a dictionary."
        )
        assert payloads == input_df["api_payload"].tolist(), (
            f"Expected payloads: {input_df['api_payload'].tolist()}, "
            f"but got: {payloads}"
        )


class TestRecordLookupResults:
    """Unit tests for the record_lookup_results function."""

    def test_record_lookup_results(
        self, dummy_data_lookup_prep, dummy_lookup_results
    ):
        """Test that the function records lookup results correctly."""
        responses, expected_results = dummy_lookup_results
        input_df = dummy_data_lookup_prep
        ids = dummy_data_lookup_prep["unique_id"].tolist()

        result_df = data_module.record_lookup_results(
            input_df, ids, responses
        )

        assert_frame_equal(result_df, expected_results)

    def test_record_lookup_results_no_responses(self, dummy_data_lookup_prep):
        """Test function raises ValueError when no responses are provided."""
        input_df = dummy_data_lookup_prep
        ids = dummy_data_lookup_prep["unique_id"].tolist()
        responses = []  # No responses

        with pytest.raises(ValueError, match="No lookup responses provided"):
            data_module.record_lookup_results(input_df, ids, responses)

    def test_record_lookup_results_mismatched_lengths(
        self, dummy_data_lookup_prep
    ):
        """Test for ValueError when lengths of ids and responses mismatch."""
        input_df = dummy_data_lookup_prep
        ids = dummy_data_lookup_prep["unique_id"].tolist()
        responses = [{"code": "1", "description": "description 1"}]

        with pytest.raises(
            ValueError,
            match="Mismatch between number of lookup IDs and lookup responses"
        ):
            data_module.record_lookup_results(input_df, ids, responses)
