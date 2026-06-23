"""Unit tests for ApiEvaluator data functionality."""

import pandas as pd
import pytest

from survey_assist_eval.pipeline.api import data as data_module

# turning off black for this file: set max line length to match PEP8 (79 chars)
# for improved readability
# fmt: off


@pytest.fixture
def dummy_input_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Fixture to provide dummy input test data for API evaluation."""
    data = {
        "unique_id": [1, 2],
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
        "clerical_codes": ["1234", "5678"],
    }
    expected_org_descriptions = [
        "Contractor",
        "Secondary education",
        "",
    ]
    return pd.DataFrame(data), pd.Series(expected_org_descriptions)


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
