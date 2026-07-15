"""Unit tests for ApiEvaluator data functionality."""

import datetime
from contextlib import ExitStack, contextmanager
from unittest.mock import ANY, MagicMock, patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from survey_assist_eval.evaluation.metrics import (
    AccuracyMetrics,
    AmbiguityMetrics,
    CodabilityMetrics,
    SimpleMetrics,
)
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
        "unique_id": [1, 2, 3, 4, 5, 6],
        "job_title": [
            "job_1", "job_2", "job_3", "job_4", "job_5", "job_6",
        ],
        "job_description": [
            "desc_1", "desc_2", "desc_3", "desc_4", "desc_5", "desc_6",
        ],
        "org_description": [
            "org_1", "org_2", "org_3", "org_4", "org_5", "org_6",
        ],
        "clerical_codes": [
            "code_1", "code_2", "code_3", "code_4", "code_5", "code_6",
        ],
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
            {
                "job_title": "job_5",
                "job_description": "desc_5",
                "org_description": "org_5",
            },
            {
                "job_title": "job_6",
                "job_description": "desc_6",
                "org_description": "org_6",
            },
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def dummy_lookup_results(
    dummy_data_lookup_prep
) -> tuple[list[dict[str, str] | None], pd.DataFrame, list[int]]:
    """Dummy lookup results for testing the record_lookup_results function."""
    # dummpy lookup API responses
    responses = [
        {"code": "1", "description": "description 1"},
        None,  # simulating no lookup result
        {},  # simulating an API failure
        {"code": "4", "description": "description 4"},
        None,
        None,
    ]

    # add expected lookup results to the dummy data for comparison in tests
    expected_results = dummy_data_lookup_prep.copy()
    expected_results["lookup_classified"] = [
        True, False, pd.NA, True, False, False
    ]
    expected_results["lookup_error"] = [
        False, False, True, False, False, False
    ]
    expected_results["lookup_code"] = ["1", pd.NA, pd.NA, "4", pd.NA, pd.NA]
    expected_results["lookup_description"] = [
        "description 1", pd.NA, pd.NA, "description 4", pd.NA, pd.NA
    ]

    return responses, expected_results, [2, 5, 6]


@pytest.fixture
def dummy_classify_results(
    dummy_lookup_results
) -> tuple[list[dict[str, str] | None], pd.DataFrame]:
    """Dummy classify results for testing the record_classify_results."""
    # simulate the API responses for classification
    responses = [
        # classified by the API
        {
            "results":
                [{
                    "classified": True,
                    "code": "2",
                    "description": "description 1",
                    "followup": None,
                    "candidates": [
                        {"code": "2a", "description": "description 2a"}
                    ],
                }],
        },
        # not classified by the API - so follow up is required
        {
            "results":
                [{
                    "classified": False,
                    "code": None,
                    "description": None,
                    "followup": "test follow up",
                    "candidates": [
                        {"code": "5a", "description": "description 5a"}
                    ],
                }],

        },
        {},  # simulate an API failure
    ]

    # build the expected results DataFrame for comparison in tests
    expected_results = dummy_lookup_results[1].copy()
    expected_results["classify_classified"] = [
        pd.NA, True, pd.NA, pd.NA, False, pd.NA
    ]
    expected_results["classify_error"] = [
        pd.NA, False, pd.NA, pd.NA, False, True
    ]
    expected_results["classify_code"] = [
        pd.NA, "2", pd.NA, pd.NA, pd.NA, pd.NA
    ]
    expected_results["classify_description"] = [
        pd.NA, "description 1", pd.NA, pd.NA, pd.NA, pd.NA
    ]
    expected_results["classify_followup"] = [
        pd.NA, pd.NA, pd.NA, pd.NA, "test follow up", pd.NA
    ]
    expected_results["classify_candidates"] = [
        pd.NA,
        [{"code": "2a", "description": "description 2a"}],
        pd.NA,
        pd.NA,
        [{"code": "5a", "description": "description 5a"}],
        pd.NA,
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


@pytest.fixture(params=[True, False])
def keep_errors(request) -> bool:
    """Factor fixture to handle paramertisation for dummy_calc_eval_..."""
    return request.param


@pytest.fixture
def dummy_calc_eval_metrics_data(keep_errors: bool) -> tuple[
    pd.DataFrame, pd.Series, pd.Series, pd.Series
]:
    """Fixture to provide dummy data for testing calc_eval_metrics function.

    Parameters:
        keep_errors (bool): Whether to keep API error cases in the expected
            results. If True, errored cases are included; if False, they are
            excluded.

    Returns:
        tuple: A tuple containing a DataFrame with dummy data for evaluation
            metrics, a Series with unique_ids, expected model results, and
            expected candidate results.
    """
    dummy_candidate_data = [
        {"code": "C4a", "description": "desc 4a", "likelihood": 0.9},
        {"code": "C5b", "description": "desc 5b", "likelihood": 0.8},
    ]
    data = {
        "unique_id": [1, 2, 3, 4, 5, 6],
        "lookup_classified": [True, pd.NA, True, False, False, False],
        "lookup_code": ["L1", pd.NA, "L3", pd.NA, pd.NA, pd.NA],
        "lookup_error": [False, True, False, False, False, False],
        "classify_classified": [pd.NA, pd.NA, pd.NA, True, False, pd.NA],
        "classify_error": [pd.NA, pd.NA, pd.NA, False, False, True],
        "classify_code": [pd.NA, pd.NA, pd.NA, "C4", pd.NA, pd.NA],
        "classify_candidates": [
            pd.NA,
            pd.NA,
            pd.NA,
            [dummy_candidate_data[0]],
            [dummy_candidate_data[1]],
            pd.NA,
        ],
    }
    if keep_errors:
        expected_ids = pd.Series(data["unique_id"])
        expected_unambiguous_codes = pd.Series(
            ["L1", pd.NA, "L3", "C4", pd.NA, pd.NA]
        )
        expected_candidate_results = pd.Series(
            [
                [],  # ensures lookup classifys are empty
                [],  # ensures lookup errors are empty
                [],  # ensures lookup classifys are empty, again
                [dummy_candidate_data[0]],
                [dummy_candidate_data[1]],
                [],  # ensure classify errors are empty
            ]
        )
    else:
        # ensure errored cases are excluded from metrics_df
        expected_ids = pd.Series([1, 3, 4, 5])
        expected_unambiguous_codes = pd.Series(["L1", "L3", "C4", pd.NA])
        expected_candidate_results = pd.Series(
            [
                [],  # ensures lookup classifies are empty
                [],
                [dummy_candidate_data[0]],
                [dummy_candidate_data[1]],
            ]
        )
    return (
        pd.DataFrame(data),
        expected_ids,
        expected_unambiguous_codes,
        expected_candidate_results,
    )


@contextmanager
def get_calc_eval_metrics_mocks(
    unique_ids: pd.Series,
    unambiguous_codes: pd.Series,
    invalid_codes_during_prep: bool = False
):
    """Context manager to mock calc_eval_metrics function."""
    if not invalid_codes_during_prep:
        data = {
            "unique_id": unique_ids.tolist(),
            "model_codes": unambiguous_codes.tolist(),
            "model_codes_invalid": [set() for _ in range(len(unique_ids))],

        }
    else:
        data = {
            "unique_id": unique_ids.tolist(),
            "model_codes": unambiguous_codes.tolist(),
            "model_codes_invalid": [
                {"invalid_code"} for _ in range(len(unique_ids))
            ],
        }
    prepped_codes = pd.DataFrame(data)
    with ExitStack() as stack:
        mock_get_logger = stack.enter_context(
            patch(
                "survey_assist_eval.pipeline.api.data.get_logger",
                return_value=MagicMock()
            )
        )
        mock_prepped_codes = stack.enter_context(
            patch(
                "survey_assist_eval.pipeline.api.data.prep_model_codes",
                return_value=prepped_codes
            )
        )
        mock_calc_simple_metrics = stack.enter_context(
            patch(
                "survey_assist_eval.pipeline.api.data.calc_simple_metrics",
                return_value=SimpleMetrics(
                    ambiguity_metrics=AmbiguityMetrics(
                        precision=0.0,
                        recall=0.0,
                        f1=0.0,
                        accuracy=0.0,
                        TP=0,
                        FP=0,
                        FN=0,
                        TN=0
                    ),
                    codability_metrics=CodabilityMetrics(
                        initial_codable_prop=0.0, initial_codable_count=0
                    ),
                    initial_accuracy_metrics=AccuracyMetrics(),
                    final_accuracy_metrics=None,
                )
            )
        )
        yield {
            "get_logger": mock_get_logger,
            "prep_model_codes": mock_prepped_codes,
            "calc_simple_metrics": mock_calc_simple_metrics,
        }


@pytest.fixture
def dummy_cal_eval_perf_data() -> tuple[pd.DataFrame, int, int, int]:
    """Fixture to provide dummy data for testing calc_eval_perf function.

    Returns:
        tuple: A tuple containing a DataFrame with dummy data, the number of
            total records, the number of lookup errors, and the number of
            classify errors.
    """
    data = {
        "lookup_error": [False, False, True, True, True, False, False],
        "classify_error": [False, True, False, True, False, True, True],
    }
    return (
        pd.DataFrame(data),
        len(data["lookup_error"]),  # num records
        len([e for e in data["lookup_error"] if e]),  # num lookup errors
        len([e for e in data["classify_error"] if e])  # num classify errors
    )


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
        responses, expected_results, _ = dummy_lookup_results
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
        responses = []

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


class TestPrepDataForClassify:
    """Unit tests for the prep_data_for_classify function."""

    def test_prep_data_for_classify(self, dummy_lookup_results):
        """Test function prepares data correctly for classification."""
        _, input_df, classify_ids = dummy_lookup_results
        ids, payloads = data_module.prep_data_for_classify(input_df)

        assert isinstance(ids, list), (
            f"Expected ids to be a list, but got: {type(ids)}"
        )
        assert isinstance(payloads, list), (
            f"Expected payloads to be a list, but got: {type(payloads)}"
        )
        assert all(isinstance(payload, dict) for payload in payloads), (
            "Expected each payload to be a dictionary."
        )

        num_ids = len(ids)
        num_payloads = len(payloads)
        num_for_classify = len(classify_ids)
        assert num_ids == num_for_classify, (
            f"Expected {num_for_classify} IDs for classification, "
            f"but got {num_ids}"
        )
        assert num_payloads == num_for_classify, (
            f"Expected {num_for_classify} payloads for classification, "
            f"but got {num_payloads}"
        )
        assert ids == classify_ids, (
            f"Expected IDs for classification: {classify_ids}, "
            f"but got: {ids}"
        )

    def test_prep_data_for_classify_no_lookup_prior(
        self, dummy_lookup_results
    ):
        """Simulate no lookup prior to classification."""
        _, input_df, _ = dummy_lookup_results
        input_df.drop(columns=["lookup_classified"], inplace=True)
        with pytest.raises(
            KeyError, match="DataFrame must contain 'lookup_classified'"
        ):
            data_module.prep_data_for_classify(input_df)


class TestRecordClassifyResults:
    """Unit tests for the record_classify_results function."""

    def test_record_classify_results(
        self, dummy_lookup_results, dummy_classify_results
    ):
        """Test that the function records classify results correctly."""
        _, input_df, classify_ids = dummy_lookup_results
        responses, expected_results = dummy_classify_results

        result_df = data_module.record_classify_results(
            input_df, classify_ids, responses
        )

        assert_frame_equal(result_df, expected_results)

    def test_record_classify_results_no_responses(
        self, dummy_lookup_results
    ):
        """Test function raises ValueError when no responses are provided."""
        _, input_df, classify_ids = dummy_lookup_results
        responses = []

        with pytest.raises(ValueError, match="No classify responses provided"):
            data_module.record_classify_results(
                input_df, classify_ids, responses
            )

    def test_record_classify_results_mismatched_lengths(
        self, dummy_lookup_results
    ):
        """Test for ValueError when lengths of ids and responses mismatch."""
        _, input_df, classify_ids = dummy_lookup_results
        responses = [{"classified": True}]  # Mismatched length

        with pytest.raises(
            ValueError,
            match=(
                "Mismatch between number of classify IDs and classify """
                "responses"
            )
        ):
            data_module.record_classify_results(
                input_df, classify_ids, responses
            )


class TestCalcEvalPerf:
    """Unit tests for the calc_eval_perf function."""

    def test_calc_eval_perf(self, dummy_cal_eval_perf_data):
        """Test the function calculates evaluation performance correctly."""
        input_df, exp_records, exp_lookup_errs, exp_classify_errs = (
            dummy_cal_eval_perf_data
        )
        start = datetime.datetime.now()
        duration = 10  # seconds
        end = datetime.timedelta(seconds=duration) + start
        lookup_request_parallelism = 10
        classify_request_parallelism = 5
        perf_metrics = data_module.calc_eval_perf(
            input_df,
            start,
            end,
            lookup_request_parallelism=lookup_request_parallelism,
            classify_request_parallelism=classify_request_parallelism,
        )

        assert perf_metrics["num_records"] == exp_records, (
            f"Expected {exp_records} records, but got: "
            f"{perf_metrics['num_records']}"
        )
        assert perf_metrics["duration_seconds"] == duration, (
            f"Expected duration of {duration} seconds, but got: "
            f"{perf_metrics['duration_seconds']}"
        )
        assert perf_metrics["records_per_second"] == exp_records / duration, (
            f"Expected records per second: {exp_records / duration}, but got: "
            f"{perf_metrics['records_per_second']}"
        )
        assert perf_metrics["num_lookup_errors"] == exp_lookup_errs, (
            f"Expected {exp_lookup_errs} lookup errors, but got: "
            f"{perf_metrics['num_lookup_errors']}"
        )
        assert (
            perf_metrics["lookup_parallelism"] == lookup_request_parallelism
        ), (
            f"Expected lookup parallelism: {lookup_request_parallelism}, but "
            f"got: {perf_metrics['lookup_parallelism']}"
        )
        assert perf_metrics["lookup_error_rate"] == (
            exp_lookup_errs / exp_records
        ), (
            f"Expected lookup error rate: {exp_lookup_errs / exp_records}, but"
            f" got: {perf_metrics['lookup_error_rate']}"
        )
        assert perf_metrics["num_classify_errors"] == exp_classify_errs, (
            f"Expected {exp_classify_errs} classify errors, but got: "
            f"{perf_metrics['num_classify_errors']}"
        )
        assert (
            perf_metrics[
                "classify_parallelism"
            ] == classify_request_parallelism
        ), (
            f"Expected classify parallelism: {classify_request_parallelism}, "
            f"but got: {perf_metrics['classify_parallelism']}"
        )
        assert perf_metrics["classify_error_rate"] == (
            exp_classify_errs / exp_records
        ), (
            f"Expected classify error rate: {exp_classify_errs / exp_records},"
            f" but got: {perf_metrics['classify_error_rate']}"
        )

    def test_calc_eval_perf_with_missing_cols(self, dummy_cal_eval_perf_data):
        """Test raises KeyError when required columns are missing."""
        input_df, _, _, _ = dummy_cal_eval_perf_data
        start = datetime.datetime.now()
        end = start + datetime.timedelta(seconds=10)
        for col in ["lookup_error", "classify_error"]:
            with pytest.raises(
                KeyError, match=f"DataFrame must contain \'{col}\'"
            ):
                # explict typecast to df since removing col reverts to series
                # and has not attribute columns
                test_df = pd.DataFrame(input_df.drop(columns=[col]))
                data_module.calc_eval_perf(test_df, start, end)

    def test_calc_eval_perf_with_no_records(self, dummy_cal_eval_perf_data):
        """Test raises ValueError when DataFrame has no records."""
        input_df, _, _, _ = dummy_cal_eval_perf_data
        start = datetime.datetime.now()
        end = start + datetime.timedelta(seconds=10)
        empty_df = pd.DataFrame(columns=input_df.columns)
        with pytest.raises(ValueError, match="DataFrame is empty."):
            data_module.calc_eval_perf(empty_df, start, end)


class TestCalcEvalMetrics:
    """Unit tests for the calc_eval_metrics function."""

    # required pylint ignore for unit testing purposes
    # pylint: disable=W0212
    def test_calc_eval_metrics_prep(
        self, dummy_calc_eval_metrics_data, keep_errors
    ):
        """Test that the function prepares data correctly for evaluation."""
        (
            input_df,
            expected_ids,
            expected_unambiguous_codes,
            expected_candidate_results
        ) = dummy_calc_eval_metrics_data

        metrics_df = data_module._prep_df_for_eval(
            input_df, keep_api_errors=keep_errors
        )

        assert isinstance(metrics_df, pd.DataFrame), (
            f"Expected prep return to be a DataFrame, but got: "
            f"{type(metrics_df)}"
        )
        unique_ids = metrics_df["unique_id"]
        assert unique_ids.equals(expected_ids), (
            f"Expected unique IDs: {expected_ids.tolist()}, but got: "
            f"{unique_ids.tolist()}"
        )
        unambiguous_codes = metrics_df["unambiguous_codes"]
        assert unambiguous_codes.equals(expected_unambiguous_codes), (
            "Expected unambiguous codes: "
            f"{expected_unambiguous_codes.tolist()}, but got: "
            f"{unambiguous_codes.tolist()}"
        )
        candidate_results = metrics_df["classify_candidates"]
        assert candidate_results.equals(expected_candidate_results), (
            f"Expected candidate results: "
            f"{expected_candidate_results.tolist()}, but got: "
            f"{candidate_results.tolist()}"
        )

    @pytest.mark.parametrize("classify_type", ["sic", "soc"])
    def test_calc_eval_metrics_with_missing_cols(
        self, dummy_calc_eval_metrics_data, classify_type
    ):
        """Test raises KeyError when required columns are missing."""
        # don't need to vary keep_errors here as only col checks
        input_df, _, _, _ = dummy_calc_eval_metrics_data
        for col in input_df.columns:
            test_df = pd.DataFrame(input_df.drop(columns=[col]))
            with pytest.raises(
                KeyError, match=f"DataFrame must contain \'{col}\'"
            ):
                data_module.calc_eval_metrics(test_df, classify_type)

    @pytest.mark.parametrize("classify_type", ["sic", "soc"])
    def test_calc_eval_metrics_all_prepped_model_codes_valid(
        self, dummy_calc_eval_metrics_data, classify_type, keep_errors
    ):
        """Test function succeeds when all prepped model codes are valid."""
        (
            input_df, unique_ids, unambiguous_codes, _
        ) = dummy_calc_eval_metrics_data
        with get_calc_eval_metrics_mocks(
            unique_ids, unambiguous_codes, invalid_codes_during_prep=False
        ) as mocks:
            metrics = data_module.calc_eval_metrics(
                input_df, classify_type, keep_api_errors=keep_errors
            )

        assert not metrics["misc"]["invalid_model_codes_detected"], (
            "Expected invalid_model_codes_detected to be False when all "
            "prepped model codes are valid."
        )
        api_errors_in_metrics_calc = metrics["misc"][
            "api_errors_in_metrics_calc"
        ]
        assert api_errors_in_metrics_calc == keep_errors, (
            "Expected api_errors_in_metrics_calc to match keep_api_errors "
            f"parameter. Got {api_errors_in_metrics_calc} but expected "
            f"{keep_errors}."
        )
        assert not mocks["get_logger"].return_value.warning.called, (
            "Did not expect warning log with valid prepped model codes."
        )
        mocks["prep_model_codes"].assert_called_once_with(
            ANY,
            codes_col="unambiguous_codes",
            alt_codes_col="classify_candidates",
            code_type=classify_type
        )
        mocks["calc_simple_metrics"].assert_called_once_with(
            ANY,
            truth_col="clerical_codes",
            initial_model_col="model_codes",
            final_model_col=None,
        )

    @pytest.mark.parametrize("classify_type", ["sic", "soc"])
    def test_calc_eval_metrics_with_invalid_prepped_model_codes(
        self, dummy_calc_eval_metrics_data, classify_type, keep_errors, caplog
    ):
        """Test raises warning log message when prepped model codes invalid."""
        (
            input_df, unique_ids, unambiguous_codes, _
        ) = dummy_calc_eval_metrics_data
        with get_calc_eval_metrics_mocks(
            unique_ids, unambiguous_codes, invalid_codes_during_prep=True
        ) as mocks, caplog.at_level("WARNING"):
            metrics = data_module.calc_eval_metrics(
                input_df, classify_type, keep_api_errors=keep_errors
            )

        assert metrics["misc"]["invalid_model_codes_detected"], (
            "Expected invalid_model_codes_detected to be True when invalid "
            "prepped model codes are present."
        )
        api_errors_in_metrics_calc = metrics["misc"][
            "api_errors_in_metrics_calc"
        ]
        assert api_errors_in_metrics_calc == keep_errors, (
            "Expected api_errors_in_metrics_calc to match keep_api_errors "
            f"parameter. Got {api_errors_in_metrics_calc} but expected "
            f"{keep_errors}."
        )
        assert mocks["get_logger"].return_value.warning.called, (
            "Expected a warning log message for invalid prepped model codes."
        )
        mocks["prep_model_codes"].assert_called_once_with(
            ANY,
            codes_col="unambiguous_codes",
            alt_codes_col="classify_candidates",
            code_type=classify_type
        )
        mocks["calc_simple_metrics"].assert_called_once_with(
            ANY,
            truth_col="clerical_codes",
            initial_model_col="model_codes",
            final_model_col=None,
        )
