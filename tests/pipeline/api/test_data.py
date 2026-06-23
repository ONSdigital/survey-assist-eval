"""Unit tests for ApiEvaluator data functionality."""

from survey_assist_eval.pipeline.api import data as data_module

# turning off black for this file: set max line length to match PEP8 (79 chars)
# for improved readability
# fmt: off


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

    def test_build_org_description_with_unknown_values(self):
        """Test that unknown values (-9) are removed from the end string."""
        result = data_module._build_org_description(  # pylint: disable=W0212
            "TestCompany", "-9", "Ltd."
        )
        assert result == "TestCompanyLtd."

    def test_build_org_description_with_non_string_inputs(self):
        """Test that non-string inputs are ignored."""
        result = data_module._build_org_description(  # pylint: disable=W0212
            "TestCompany", 123, None, "Ltd."
        )
        assert result == "TestCompanyLtd."
