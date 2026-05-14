"""Test lookup soc_code_list and associated helper functions in code_standard."""

# pylint: disable=C0116

from survey_assist_eval.data_cleaning.soc_code_list import (
    VALID_SOC_CODES,
    generate_valid_soc_codes,
)


def test_valid_soc_codes_notnone():
    assert isinstance(VALID_SOC_CODES, set), "VALID_SOC_CODES should be a set"
    assert len(VALID_SOC_CODES) > 0, "VALID_SOC_CODES should not be empty"


def test_valid_soc_codes_contains_prefixes():
    # map len to codes
    code_len = [len(code) for code in VALID_SOC_CODES]
    len_counts = {length: code_len.count(length) for length in set(code_len)}
    # SOC 2020 has codes of length 1 to 4 (major groups to unit groups)
    assert len_counts.keys() == {
        1,
        2,
        3,
        4,
    }, "VALID_SOC_CODES should contain codes of length 1 to 4"


def test_valid_soc_codes_no_invalid_entries():
    # SOC codes are purely numeric - no section letters like SIC
    assert not any(
        c.isalpha() for c in VALID_SOC_CODES
    ), "VALID_SOC_CODES should not contain letter-based section codes"
    # No 5-digit codes (SOC 2020 uses 4-digit unit groups at most)
    assert not any(
        len(c) > 4 for c in VALID_SOC_CODES
    ), "VALID_SOC_CODES should not contain codes longer than 4 digits"


def test_generate_valid_soc_codes_basic():
    code_list = ("1111", "2123")
    valid_codes = generate_valid_soc_codes(code_list)

    assert isinstance(valid_codes, set), "Valid codes is not a set"
    # Should include all prefixes of "1111"
    assert "1" in valid_codes, "Missing prefix 1"
    assert "11" in valid_codes, "Missing prefix 11"
    assert "111" in valid_codes, "Missing prefix 111"
    assert "1111" in valid_codes, "Missing code 1111"
    # Should include all prefixes of "2123"
    assert "2" in valid_codes, "Missing prefix 2"
    assert "21" in valid_codes, "Missing prefix 21"
    assert "212" in valid_codes, "Missing prefix 212"
    assert "2123" in valid_codes, "Missing code 2123"


def test_generate_valid_soc_codes_no_section_letters():
    # SOC codes are purely numeric - no section letters unlike SIC
    code_list = ("1111", "2111")
    valid_codes = generate_valid_soc_codes(code_list)
    assert not any(c.isalpha() for c in valid_codes), "SOC codes should be numeric only"


def test_generate_valid_soc_codes_empty():
    assert not generate_valid_soc_codes(())
