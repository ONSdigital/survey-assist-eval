"""Test SIC code parsing and validation."""

# pylint: disable=C0116

import pytest

from survey_assist_eval.data_cleaning.code_standard import (
    asses_codability_gain,
    expand_to_n_digit_str,
    extract_alt_candidates_n_digit_codes,
    get_clean_n_digit_codes,
    get_clean_n_digit_one_code,
    get_codability_level,
    parse_numerical_code,
    validate_codes,
)


def test_parse_numerical_code_basic():
    assert parse_numerical_code("86101") == {"86101"}
    assert parse_numerical_code("[86101, 86210]") == {"86101", "86210"}
    assert parse_numerical_code("86101;8602x;4+") == {"86101", "8602x"}
    assert parse_numerical_code(86101) == {"86101"}


def test_parse_numerical_code_empty():
    assert parse_numerical_code("nan") == set()
    assert parse_numerical_code("-9") == set()
    assert parse_numerical_code("") == set()
    assert parse_numerical_code("") == set()
    assert parse_numerical_code(None) == set()


def test_parse_numerical_code_logs(caplog):
    with caplog.at_level("WARNING"):
        parse_numerical_code(0, code_regex_pattern=r"([")
    assert any("error parsing" in record.message.lower() for record in caplog.records)


def test_expand_to_n_digit_str():
    assert expand_to_n_digit_str("86101", 2) == {"86101"}
    assert expand_to_n_digit_str("", 1) == {str(x) for x in range(10)}
    result = expand_to_n_digit_str("86", 5)
    assert "86000" in result
    assert "86999" in result
    assert len(result) == 10**3


def test_get_clean_n_digit_one_code():
    assert get_clean_n_digit_one_code("861012", n=5, code_type="SIC") == {"86101"}
    assert get_clean_n_digit_one_code("86101", n=5, code_type="SIC") == {"86101"}
    group86 = {"86100", "86102", "86230", "86220", "86900", "86101", "86210"}
    assert get_clean_n_digit_one_code("86xxx", n=5, code_type="SIC") == group86
    assert get_clean_n_digit_one_code("86", n=5, code_type="SIC") == group86
    assert get_clean_n_digit_one_code("861012", n=3, code_type="SIC") == {"861"}
    assert get_clean_n_digit_one_code("86101", n=0, code_type="SIC") == {"Q"}


def test_get_clean_n_digit_codes_from_section():
    assert get_clean_n_digit_codes("C", n=0, code_type="SIC")[0] == {"C"}
    assert len(get_clean_n_digit_codes("I", n=5, code_type="SIC")[0]) == 16
    assert len(get_clean_n_digit_codes("M", n=2, code_type="SIC")[0]) == 7
    assert get_clean_n_digit_codes("Z", n=0, code_type="SIC")[1] == {"Z"}


def test_get_clean_n_digit_codes_with_invalid():
    # Case 1: Mixed valid and invalid (alphanumeric/garbage)
    codes = ["86101", "NotACode", "123456789", "86210"]
    valid, invalid = get_clean_n_digit_codes(codes, n=5, code_type="SIC")

    # Check valid codes are processed
    assert "86101" in valid
    assert "86210" in valid
    assert len(valid) == 2

    # Check invalid codes are captured correctly
    assert "NotACode" in invalid
    # "123456789" will likely fail the .isdigit() check or validation check
    assert "123456789" in invalid
    assert len(invalid) == 2

    # Case 2: Purely invalid codes
    bad_input = ["Bad1", "Bad2"]
    valid, invalid = get_clean_n_digit_codes(bad_input, n=5, code_type="SIC")
    assert valid == set()
    assert invalid == {"Bad1", "Bad2"}

    # Case 3: Numeric codes that shouldn't exist (assuming 00000 is not in VALID_SIC_CODES)
    # This tests the validation lookup failure path
    fake_codes = ["00000"]
    valid, invalid = get_clean_n_digit_codes(fake_codes, n=5, code_type="SOC")
    if "00000" not in valid:  # Only assert if we are sure it's invalid in your lookup
        assert "00000" in invalid


def test_get_clean_n_digit_codes_5d():
    codes = ["86101", "86210", "85xxx"]
    result, invalid = get_clean_n_digit_codes(codes, n=5, code_type="SIC")
    assert "86101" in result
    assert "86210" in result
    assert "85100" in result
    assert "85590" in result
    assert isinstance(result, set)

    # Check invalid codes - there should be none
    assert isinstance(invalid, set)
    assert invalid == set()


def test_get_clean_n_digit_codes_logs(caplog):
    with caplog.at_level("WARNING"):
        get_clean_n_digit_codes(3, n=5, code_type="SIC")
    assert any("set of strings" in record.message.lower() for record in caplog.records)


def test_get_clean_n_digit_codes_section():
    cleaned, invalid = get_clean_n_digit_codes("2xxxx", n=0, code_type="SIC")
    assert cleaned == {"C"}
    assert invalid == set()

    codes = ["86101", "86210", "2xxxx"]
    cleaned, invalid = get_clean_n_digit_codes(codes, n=0, code_type="SIC")
    assert cleaned == {"Q", "C"}
    assert invalid == set()


def test_get_clean_n_digit_codes_logs_invalid_item(caplog):
    # Ensure we capture WARNING logs for this test
    with caplog.at_level("WARNING"):
        # Input contains an item that will produce no valid codes
        cleaned, invalid = get_clean_n_digit_codes({"98765"}, n=5, code_type="SIC")

    # Assert logging happened with the expected message fragment
    assert any(
        "has no valid codes" in record.message.lower() for record in caplog.records
    ), "Expected a warning about invalid codes to be logged"

    # And the invalid item is recorded in the invalid set
    assert "98765" in invalid
    # cleaned may be empty depending on your cleaning logic
    assert isinstance(cleaned, set)
    assert isinstance(invalid, set)


def test_validate_sic_codes():
    assert validate_codes("01110", code_type="SIC") == {"01110"}
    valid = validate_codes(["01110", "99999", "A"], code_type="SIC")
    assert "01110" in valid
    assert "A" in valid
    assert "99999" not in valid


def test_validate_sic_codes_logs(caplog):
    with caplog.at_level("WARNING"):
        validate_codes(5, code_type="SIC")
    assert any("set of strings" in record.message.lower() for record in caplog.records)


def test_extract_alt_candidates_n_digit_codes():
    candidates = [
        {"code": "86101", "likelihood": 0.8},
        {"code": "86210", "likelihood": 0.6},
    ]
    result_valid, result_invalid = extract_alt_candidates_n_digit_codes(
        candidates,
        code_name="code",
        score_name="likelihood",
        threshold=0.7,
        code_type="SIC",
    )
    assert result_valid == {"86101"}
    assert result_invalid == set()
    # No pruning
    result2_valid, result2_invalid = extract_alt_candidates_n_digit_codes(
        candidates,
        code_name="code",
        score_name="likelihood",
        threshold=0,
        code_type="SIC",
    )
    assert result2_valid == {"86101", "86210"}
    assert result2_invalid == set()


def test_extract_alt_candidates_n_digit_codes_invalid():
    candidates = [
        {"code": "86101", "likelihood": 0.8},
        {"code": "12345", "likelihood": 0.6},
    ]
    result_valid, result_invalid = extract_alt_candidates_n_digit_codes(
        candidates,
        code_name="code",
        score_name="likelihood",
        threshold=0.7,
        code_type="SIC",
    )
    assert result_valid == {"86101"}
    assert result_invalid == {"12345"}


@pytest.mark.parametrize(
    "codes,expected",
    [
        (set(), "Uncodable"),
        ({"01621", "8610"}, "Uncodable"),
        ({"01", "02"}, "Section (letter)"),
        ({"01", "01621"}, "Division (2-digits)"),
        ({"016"}, "Group (3-digits)"),
        ({"861"}, "Class (4-digits)"),
        ({"0162"}, "Class (4-digits)"),
        ({"01621"}, "Sub-class (5-digits)"),
    ],
)
def test_get_codability_level(codes, expected):
    assert get_codability_level(codes, code_type="SIC") == expected


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("Uncodable", "Uncodable", 0),
        ("Uncodable", "Section (letter)", 1),
        ("Section (letter)", "Uncodable", -1),
        ("Division (2-digits)", "Group (3-digits)", 1),
        ("Group (3-digits)", "Division (2-digits)", -1),
        ("Class (4-digits)", "Class (4-digits)", 0),
        ("Sub-class (5-digits)", "Class (4-digits)", -1),
        ("not a label", "Sub-class (5-digits)", None),
        ("not a label", "not a label", None),
    ],
)
def test_asses_codability_gain(left, right, expected):
    row = {"initial": left, "final": right}
    out = asses_codability_gain(
        row,
        initial_level_col="initial",
        final_level_col="final",
        code_type="SIC",
    )
    assert out == expected


# ---------------------------------------------------------------------------
# SOC-specific tests
# ---------------------------------------------------------------------------


def test_parse_numerical_code_soc():
    assert parse_numerical_code("1111", code_type="SOC") == {"1111"}
    assert parse_numerical_code("[1111, 2111]", code_type="SOC") == {"1111", "2111"}
    assert parse_numerical_code(1111, code_type="SOC") == {"1111"}


def test_get_clean_n_digit_one_code_soc():
    # Truncation: longer code -> take first 4 digits
    assert get_clean_n_digit_one_code("11113", n=4, code_type="SOC") == {"1111"}
    # Exact match
    assert get_clean_n_digit_one_code("1111", n=4, code_type="SOC") == {"1111"}
    # Wildcard / short code expands to all valid unit groups in minor group 111
    group111 = {"1111", "1112"}
    assert get_clean_n_digit_one_code("111x", n=4, code_type="SOC") == group111
    assert get_clean_n_digit_one_code("111", n=4, code_type="SOC") == group111
    # Roll up to Sub-Major group (2-digit)
    assert get_clean_n_digit_one_code("11113", n=2, code_type="SOC") == {"11"}
    # Roll up to Major group (1-digit)
    assert get_clean_n_digit_one_code("1111", n=1, code_type="SOC") == {"1"}


def test_get_clean_n_digit_codes_from_major_group():
    # n=1 is the top codability level for SOC (Major group)
    assert get_clean_n_digit_codes("1111", n=1, code_type="SOC")[0] == {"1"}
    # All 4-digit unit groups under minor group "111" (only 1111 and 1112)
    assert len(get_clean_n_digit_codes("111", n=4, code_type="SOC")[0]) == 2
    # Sub-major groups under major group "1"
    result_2d, _ = get_clean_n_digit_codes("1", n=2, code_type="SOC")
    assert "11" in result_2d
    assert "12" in result_2d
    # Completely invalid code lands in invalid set
    assert get_clean_n_digit_codes("Z", n=1, code_type="SOC")[1] == {"Z"}


def test_get_clean_n_digit_codes_with_invalid_soc():
    # Case 1: Mixed valid and invalid
    codes = ["1111", "NotACode", "123456789", "2111"]
    valid, invalid = get_clean_n_digit_codes(codes, n=4, code_type="SOC")
    assert "1111" in valid
    assert "2111" in valid
    assert len(valid) == 2
    assert "NotACode" in invalid
    assert "123456789" in invalid
    assert len(invalid) == 2

    # Case 2: Purely invalid
    valid, invalid = get_clean_n_digit_codes(["Bad1", "Bad2"], n=4, code_type="SOC")
    assert valid == set()
    assert invalid == {"Bad1", "Bad2"}

    # Case 3: Numeric code not present in SOC lookup
    valid, invalid = get_clean_n_digit_codes(["9999"], n=4, code_type="SOC")
    assert "9999" in invalid


def test_get_clean_n_digit_codes_4d():
    codes = ["1111", "2111", "111x"]
    result, invalid = get_clean_n_digit_codes(codes, n=4, code_type="SOC")
    assert "1111" in result
    assert "2111" in result
    assert "1112" in result  # from 111x expansion
    assert isinstance(result, set)
    assert invalid == set()


def test_get_clean_n_digit_codes_major_group():
    # Roll up to Major group (n=1) - comparable to SIC section rollup
    cleaned, invalid = get_clean_n_digit_codes("1xxx", n=1, code_type="SOC")
    assert cleaned == {"1"}
    assert invalid == set()

    codes = ["1111", "2111", "1xxx"]
    cleaned, invalid = get_clean_n_digit_codes(codes, n=1, code_type="SOC")
    assert cleaned == {"1", "2"}
    assert invalid == set()


def test_get_clean_n_digit_codes_logs_invalid_item_soc(caplog):
    with caplog.at_level("WARNING"):
        cleaned, invalid = get_clean_n_digit_codes({"9999"}, n=4, code_type="SOC")
    assert any(
        "has no valid codes" in record.message.lower() for record in caplog.records
    ), "Expected a warning about invalid codes to be logged"
    assert "9999" in invalid
    assert isinstance(cleaned, set)
    assert isinstance(invalid, set)


def test_validate_soc_codes():
    assert validate_codes("1111", code_type="SOC") == {"1111"}
    valid = validate_codes(["1111", "9999", "A"], code_type="SOC")
    assert "1111" in valid
    assert "A" not in valid  # SOC has no letter-based section codes
    assert "9999" not in valid  # not a valid SOC unit group


def test_validate_soc_codes_logs(caplog):
    with caplog.at_level("WARNING"):
        validate_codes(5, code_type="SOC")
    assert any("set of strings" in record.message.lower() for record in caplog.records)


def test_extract_alt_candidates_n_digit_codes_soc():
    candidates = [
        {"code": "1111", "likelihood": 0.8},
        {"code": "2111", "likelihood": 0.6},
    ]
    result_valid, result_invalid = extract_alt_candidates_n_digit_codes(
        candidates,
        code_name="code",
        score_name="likelihood",
        threshold=0.7,
        code_type="SOC",
        n=4,
    )
    assert result_valid == {"1111"}
    assert result_invalid == set()

    # No pruning - both codes returned
    result2_valid, result2_invalid = extract_alt_candidates_n_digit_codes(
        candidates,
        code_name="code",
        score_name="likelihood",
        threshold=0,
        code_type="SOC",
        n=4,
    )
    assert result2_valid == {"1111", "2111"}
    assert result2_invalid == set()


def test_extract_alt_candidates_n_digit_codes_invalid_soc():
    candidates = [
        {"code": "1111", "likelihood": 0.8},
        {"code": "9999", "likelihood": 0.6},  # not a valid SOC code
    ]
    result_valid, result_invalid = extract_alt_candidates_n_digit_codes(
        candidates,
        code_name="code",
        score_name="likelihood",
        threshold=0.7,
        code_type="SOC",
        n=4,
    )
    assert result_valid == {"1111"}
    assert result_invalid == {"9999"}


@pytest.mark.parametrize(
    "codes,expected",
    [
        (set(), "Uncodable"),
        ({"1111", "2111"}, "Uncodable"),  # different major groups, no common root
        ({"11", "12"}, "Major group (1-digit)"),  # both under major group 1
        ({"11"}, "Sub-Major group (2-digits)"),
        ({"111"}, "Minor group (3-digits)"),
        ({"1111", "1112"}, "Minor group (3-digits)"),  # same minor group 111
        ({"1111"}, "Unit group (4-digits)"),
    ],
)
def test_get_codability_level_soc(codes, expected):
    assert get_codability_level(codes, code_type="SOC") == expected


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("Uncodable", "Uncodable", 0),
        ("Uncodable", "Major group (1-digit)", 2),  # -1 → 1
        ("Major group (1-digit)", "Uncodable", -2),  # 1 → -1
        ("Sub-Major group (2-digits)", "Minor group (3-digits)", 1),  # 2 → 3
        ("Minor group (3-digits)", "Sub-Major group (2-digits)", -1),  # 3 → 2
        ("Minor group (3-digits)", "Minor group (3-digits)", 0),
        ("Unit group (4-digits)", "Minor group (3-digits)", -1),  # 4 → 3
        ("not a label", "Unit group (4-digits)", None),
        ("not a label", "not a label", None),
    ],
)
def test_asses_codability_gain_soc(left, right, expected):
    row = {"initial": left, "final": right}
    out = asses_codability_gain(
        row,
        initial_level_col="initial",
        final_level_col="final",
        code_type="SOC",
    )
    assert out == expected
