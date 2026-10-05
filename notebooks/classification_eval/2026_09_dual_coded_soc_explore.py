#!/usr/bin/env python3
"""Dual-Coded SOC/SIC Analysis
Convert occupation and industry classifications from dual-coded datasets.
"""

# pylint: disable=invalid-name,too-many-lines

# %%
import logging
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score

from survey_assist_eval.data_cleaning.code_standard import (
    SIC_CODABILITY_LEVELS,
    SOC_CODABILITY_LEVELS,
    get_clean_n_digit_codes,
    get_codability_level,
)
from survey_assist_eval.data_cleaning.prep_data import (
    prep_clerical_codes,
    prep_model_codes,
)
from survey_assist_eval.evaluation.metrics import calc_simple_metrics

# SETUP: Pandas Display Options

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", 10)

# %%
# Update file paths and column names based on your actual data.

# File paths
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
CLERICALLY_CODED_SOC_PATH = (
    f"gs://{bucket_name}/evaluation-pipeline/original_datasets/sic_2k/"
    "comparison_soc_2k.xlsx"
)
TWO_K_PATH = (
    f"gs://{bucket_name}/evaluation-pipeline/original_datasets/sic_2k/"
    "sic_2k_test_data.parquet"
)

TWO_K_ID_COL = "unique_id"
TWO_K_CLERICAL_CODES = "clerical_codes"  # List of candidate codes

# Survey Assist's own SOC output for this same subset
SA_DATA_PATH = (
    f"gs://{bucket_name}/evaluation-pipeline/dual_coded_2k/soc/" "STG2.parquet"
)  # Pipeline run by Peter
SA_SOC_ID_COL = "Unique_identifier"
SA_SOC_CODES_COL = "initial_code"
SA_SOC_ALT_CODES_COL = "alt_soc_candidates"
SA_SOC_LIKELIHOOD_COL = "initial_likelihood"

# Survey Assist's own SIC output for this same subset
SA_SIC_DATA_PATH = (
    f"gs://{bucket_name}/evaluation-pipeline/dual_coded_2k/sic/" "STG2.parquet"
)
SA_SIC_ID_COL = "Unique_identifier"
SA_SIC_CODES_COL = "initial_code"

SIGNIFICANCE_LEVEL = 0.05


def show_value_counts(
    df: pd.DataFrame, column: str, label: str, top_n: int = 10
) -> None:
    """Show top N value counts for a column."""
    if column not in df.columns:
        print(f"⚠️  Column '{column}' not found in {label}")
        return

    print(f"\n{label} :: '{column}' (top {top_n} values):")
    try:
        counts = df[column].value_counts(dropna=False).head(top_n)
        print(counts.to_string())
    except TypeError:
        # Handle unhashable types (lists, arrays, etc.)
        print(
            "Column contains unhashable values (lists/arrays). "
            "Flattening and showing unique codes:"
        )
        all_codes = []
        for item in df[column].dropna():
            if isinstance(item, list | tuple | np.ndarray):
                all_codes.extend(item)
            else:
                all_codes.append(item)
        counts = pd.Series(all_codes).value_counts().head(top_n)
        print(counts.to_string())


def check_id_overlap(dataset_a: tuple, dataset_b: tuple) -> None:
    """Compare IDs between two datasets.

    Each argument is a (df, id_col, name) tuple.
    """
    df_a, id_col_a, name_a = dataset_a
    df_b, id_col_b, name_b = dataset_b

    if id_col_a not in df_a.columns:
        print(f"⚠️  Column '{id_col_a}' not found in {name_a}")
        return
    if id_col_b not in df_b.columns:
        print(f"⚠️  Column '{id_col_b}' not found in {name_b}")
        return

    ids_a = set(df_a[id_col_a].dropna().astype(str))
    ids_b = set(df_b[id_col_b].dropna().astype(str))
    common = ids_a & ids_b
    only_a = ids_a - ids_b
    only_b = ids_b - ids_a

    print(f"\n{'='*70}")
    print(f"ID Overlap: {name_a}.{id_col_a} vs {name_b}.{id_col_b}")
    print(f"{'='*70}")
    print(f"{name_a}: {len(ids_a):>6} unique IDs")
    print(f"{name_b}: {len(ids_b):>6} unique IDs")
    print(f"Common:  {len(common):>6}")
    print(f"Only in {name_a}: {len(only_a):>6}")
    print(f"Only in {name_b}: {len(only_b):>6}")

    if only_a:
        print(f"\nExamples from {name_a}: {list(only_a)[:5]}")
    if only_b:
        print(f"Examples from {name_b}: {list(only_b)[:5]}")

    overlap_pct = 100 * len(common) / max(len(ids_a), len(ids_b))
    print(f"\n✓ Overlap: {overlap_pct:.1f}%")


# %%
# DATA LOADING

print("Loading data...")
clerically_coded_soc_sheets = pd.read_excel(
    CLERICALLY_CODED_SOC_PATH, sheet_name=None, dtype=str
)
print(
    f"✓ Loaded {len(clerically_coded_soc_sheets)} sheet(s): "
    f"{list(clerically_coded_soc_sheets.keys())}"
)

# Load the configured sheet
# Define the sheet name explicitly from the workbook
# Matches the key in clerically_coded_soc_sheets
CLERICALLY_CODED_SOC_SHEET = "Comparisons"
clerically_coded_soc_df = clerically_coded_soc_sheets[CLERICALLY_CODED_SOC_SHEET]
two_k_df = pd.read_parquet(TWO_K_PATH, dtype_backend="numpy_nullable")

# Drop fully-empty junk columns left over from the Excel export (e.g. "Unnamed: 10").
empty_cols = [
    col
    for col in clerically_coded_soc_df.columns
    if clerically_coded_soc_df[col].isna().all()
]
if empty_cols:
    print(f"Dropping empty column(s) from {CLERICALLY_CODED_SOC_SHEET}: {empty_cols}")
    clerically_coded_soc_df = clerically_coded_soc_df.drop(columns=empty_cols)

# Replace the two clerical coders with generic labels
clerically_coded_soc_df = clerically_coded_soc_df.rename(
    columns={
        "Carol": "Coder1",
        "Carol_Comments": "Coder1_Comments",
        "Lynne": "Coder2",
        "Lynne_Comments": "Coder2_Comments",
    }
)
# Replace '6321' with '6231' in the clerical SOC code columns only
code_cols = [
    c
    for c in ("Coder1", "Coder2", "Final code")
    if c in clerically_coded_soc_df.columns
]
clerically_coded_soc_df[code_cols] = clerically_coded_soc_df[code_cols].replace(
    {"6321": "6231"}
)

# %%
# Clean every clerical SOC code column once, treat them as full-length SOC codes for rest

CLERICALLY_CODED_SOC_ID_COL = "Unique_identifier"
CLERICALLY_CODED_SOC_CODER1_COL = "Coder1"
CLERICALLY_CODED_SOC_CODER2_COL = "Coder2"
FINAL_CODE_COL = "Final code"

UNCODABLE_LABEL = "Uncodable"
SOC_FULL_DIGITS = max(digits for digits, _label in SOC_CODABILITY_LEVELS)

# get_clean_n_digit_codes warns on every unparseable cell
_code_standard_logger = logging.getLogger(
    "survey_assist_eval.data_cleaning.code_standard"
)
_previous_log_level = _code_standard_logger.level

unrecognised_values: set = set()


def clean_soc_code(raw: object) -> str:
    """Return the full-length clean SOC code for a coder's cell."""
    if pd.isna(raw):
        return UNCODABLE_LABEL
    raw_str = str(raw).strip()
    if not raw_str:
        return UNCODABLE_LABEL
    cleaned, invalid = get_clean_n_digit_codes(
        [raw_str], n=SOC_FULL_DIGITS, code_type="SOC"
    )
    if invalid and raw_str.lower() != "uncodable":
        unrecognised_values.add(raw_str)
    return next(iter(cleaned)) if len(cleaned) == 1 else UNCODABLE_LABEL


_code_standard_logger.setLevel(logging.ERROR)

for col in (CLERICALLY_CODED_SOC_CODER1_COL, CLERICALLY_CODED_SOC_CODER2_COL):
    clerically_coded_soc_df[col] = clerically_coded_soc_df[col].apply(clean_soc_code)
# Final code is only filled where coders disagreed - keep blanks blank
if FINAL_CODE_COL in clerically_coded_soc_df.columns:
    has_final = (
        clerically_coded_soc_df[FINAL_CODE_COL]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    clerically_coded_soc_df.loc[has_final, FINAL_CODE_COL] = (
        clerically_coded_soc_df.loc[has_final, FINAL_CODE_COL].apply(clean_soc_code)
    )

if unrecognised_values:
    print(
        f"\n⚠️  {len(unrecognised_values)} distinct coder value(s) were neither a "
        "valid SOC code nor 'uncodable' - check these for typos, they are "
        "currently being folded into the Uncodable category:"
    )
    print(sorted(unrecognised_values)[:20])

# %%
# ============================================================================
# CLERICALLY CODED SOC - PREVIEW & PROFILE
# ============================================================================

print(f"\n{'='*70}")
print("CLERICALLY CODED SOC PREVIEW: Comparisons Sheet")
print(f"{'='*70}")
print(f"\n📋 Columns ({len(clerically_coded_soc_df.columns)} total):")
for i, col in enumerate(clerically_coded_soc_df.columns, 1):
    print(f"  {i:2}. {col}")

print(
    f"\n📊 Shape: {clerically_coded_soc_df.shape[0]} rows x "
    f"{clerically_coded_soc_df.shape[1]} columns"
)
# print(clerically_coded_soc_df.head(3).to_string())

# %%
# 2k.parquet - PREVIEW & PROFILE

print(f"\n{'='*70}")
print("2k.parquet PREVIEW")
print(f"{'='*70}")
print(f"\n📋 Columns ({len(two_k_df.columns)} total):")
for i, col in enumerate(two_k_df.columns, 1):
    print(f"  {i:2}. {col}")

print(f"\n📊 Shape: {two_k_df.shape[0]} rows x {two_k_df.shape[1]} columns")
# print(two_k_df.head(3).to_string())


# %%
# CODE COLUMN VALUE SAMPLES
# Showing value distributions for configured SIC/SOC columns.

# Also used in later cells
TWO_K_SIC_COL = "sic2007_employee"
TWO_K_SIC_IND_COL = "sic_ind1"

print(f"\n{'='*70}")
print("CLERICALLY CODED SOC CODE VALUES")
print(f"{'='*70}")

show_value_counts(
    clerically_coded_soc_df,
    CLERICALLY_CODED_SOC_CODER1_COL,
    "clerically_coded_soc :: Coder 1",
    top_n=15,
)
show_value_counts(
    clerically_coded_soc_df,
    CLERICALLY_CODED_SOC_CODER2_COL,
    "clerically_coded_soc :: Coder 2",
    top_n=15,
)

# %%
# MERGE CLERICAL + 2k INTO ONE DATAFRAME

merged = clerically_coded_soc_df.merge(
    two_k_df,
    left_on=CLERICALLY_CODED_SOC_ID_COL,
    right_on=TWO_K_ID_COL,
    how="left",
    suffixes=("", "_2k"),
    validate="one_to_one",
)
is_matched = merged[TWO_K_ID_COL].notna()
print(f"\nNumber of matching unique identifiers: {is_matched.sum()}")
print(f"Number of non-matching unique identifiers: {(~is_matched).sum()}")

# Check the job title / description text agrees between the two files for each ID.
# (clerical column, 2k column) - confirm the 2k names against the column listing above.
TEXT_COL_PAIRS = {
    "job title": ("soc2020_job_title_main_job", "soc2020_job_title"),
    "job description": (
        "soc2020_job_description_main_job",
        "soc2020_job_description",
    ),
    "industry": ("sic2007_employed_main_job", TWO_K_SIC_COL),
}


def _normalise_text(s: pd.Series) -> pd.Series:
    """Lowercase, trim and collapse whitespace so formatting differences don't count."""
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


print(f"\n{'='*70}")
print("JOB TITLE / DESCRIPTION MATCH (clerical vs 2k, matched IDs only)")
print(f"{'='*70}")
for text_type, (clerical_col, two_k_col) in TEXT_COL_PAIRS.items():
    if clerical_col not in merged.columns or two_k_col not in merged.columns:
        print(
            f"⚠️  Skipping {text_type} check - '{clerical_col}' or '{two_k_col}' not found"
        )
        continue
    text_match = _normalise_text(merged[clerical_col]) == _normalise_text(
        merged[two_k_col]
    )
    n_mismatch = (is_matched & ~text_match).sum()
    print(
        f"{text_type.capitalize()} mismatches: {n_mismatch} of {is_matched.sum()} matched IDs"
    )
    if n_mismatch:
        print(
            merged.loc[
                is_matched & ~text_match,
                [CLERICALLY_CODED_SOC_ID_COL, clerical_col, two_k_col],
            ]
            .head(5)
            .to_string(index=False)
        )

print(f"\n{'='*70}")
print("2k.parquet CODE VALUES")
print(f"{'='*70}")

show_value_counts(two_k_df, TWO_K_SIC_COL, "2k.parquet :: SIC codes", top_n=15)
# show_value_counts(two_k_df, TWO_K_SIC_IND_COL, "2k.parquet :: SIC indicators", top_n=15)
show_value_counts(
    two_k_df, TWO_K_CLERICAL_CODES, "2k.parquet :: Clerical codes", top_n=10
)

# %%
# INTER-RATER RELIABILITY: CODER AGREEMENT ANALYSIS
# Compute inter-rater reliability metrics (Cohen's kappa, % agreement),
# at each SOC digit level (1/2/3/4-digit), treating "uncodable" as a
# genuine category rather than missing data.

SOC_DIGIT_LEVELS = sorted(
    {digits for digits, _label in SOC_CODABILITY_LEVELS if digits > 0}
)


def soc_label_at_digits(code: str, n_digits: int) -> str:
    """Truncate an already-clean full-length SOC code to n digits."""
    if code == UNCODABLE_LABEL:
        return UNCODABLE_LABEL
    return code[:n_digits]


print(f"\n{'='*70}")
print("INTER-RATER RELIABILITY: Coder1 vs Coder2 (by SOC digit level)")
print(f"{'='*70}")
print(f"Total records: {len(clerically_coded_soc_df)}")

digit_level_summary = []
coder1_full = pd.Series(dtype=object)
coder2_full = pd.Series(dtype=object)

for soc_digit_count in SOC_DIGIT_LEVELS:
    coder1_labels = clerically_coded_soc_df[CLERICALLY_CODED_SOC_CODER1_COL].apply(
        lambda x, n=soc_digit_count: soc_label_at_digits(x, n_digits=n)
    )
    coder2_labels = clerically_coded_soc_df[CLERICALLY_CODED_SOC_CODER2_COL].apply(
        lambda x, n=soc_digit_count: soc_label_at_digits(x, n_digits=n)
    )

    n_agree = (coder1_labels == coder2_labels).sum()
    pct_agree = 100 * n_agree / len(clerically_coded_soc_df)
    kappa = cohen_kappa_score(coder1_labels, coder2_labels)

    digit_level_summary.append(
        {
            "digits": soc_digit_count,
            "n_records": len(clerically_coded_soc_df),
            "n_agree": int(n_agree),
            "pct_agree": round(pct_agree, 1),
            "cohens_kappa": round(kappa, 4),
        }
    )

    if soc_digit_count == max(SOC_DIGIT_LEVELS):
        # keep the full 4-digit labels around for the "Agree" cross-check below
        coder1_full = coder1_labels
        coder2_full = coder2_labels

digit_level_df = pd.DataFrame(digit_level_summary).sort_values(
    "digits", ascending=False
)
print("\nAgreement by SOC digit level (uncodable treated as its own category):")
print(digit_level_df.to_string(index=False))

# Cross-check against the clerical team's own row-level "Agree" flag, if present.
if "Agree" in clerically_coded_soc_df.columns:
    our_agree = coder1_full == coder2_full
    their_agree = (
        clerically_coded_soc_df["Agree"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )
    mismatches = (our_agree != their_agree).sum()
    print(
        f"\nCross-check vs sheet's own 'Agree' column: {mismatches} of "
        f"{len(clerically_coded_soc_df)} rows disagree with our recomputed agreement "
        "(investigate before trusting the figures above if this is large)."
    )

# %%
# ID OVERLAP CHECK
# Check if both files cover the same records.

check_id_overlap(
    (
        clerically_coded_soc_df,
        CLERICALLY_CODED_SOC_ID_COL,
        f"clerically_coded_soc :: {CLERICALLY_CODED_SOC_SHEET}",
    ),
    (two_k_df, TWO_K_ID_COL, "2k.parquet"),
)

# %%
# SIC / SOC CODABILITY RELATIONSHIP


SIC_LEVEL_ORDER = [label for _digits, label in SIC_CODABILITY_LEVELS]
SOC_LEVEL_ORDER = [label for _digits, label in SOC_CODABILITY_LEVELS]


def clerical_codes_to_set(raw: object) -> set:
    """Convert a 2k parquet `clerical_codes` cell to a set of code strings."""
    if raw is None:
        return set()
    try:
        if len(raw) == 0:
            return set()
    except TypeError:
        return set()
    return {str(code) for code in raw}


# Computed on the merged dataframe; rows with no 2k match stay NaN.
merged["sic_codability_level"] = (
    merged[TWO_K_CLERICAL_CODES]
    .apply(
        lambda codes: get_codability_level(
            clerical_codes_to_set(codes), code_type="SIC"
        )
    )
    .where(merged[TWO_K_ID_COL].notna())
)

# Finest SOC digit level at which Coder1 and Coder2 agree, per row.
soc_codability = pd.Series(UNCODABLE_LABEL, index=merged.index)
already_resolved = pd.Series(False, index=merged.index)
for soc_digit_count in sorted(SOC_DIGIT_LEVELS, reverse=True):
    coder1_labels = merged[CLERICALLY_CODED_SOC_CODER1_COL].apply(
        lambda x, digits=soc_digit_count: soc_label_at_digits(x, n_digits=digits)
    )
    coder2_labels = merged[CLERICALLY_CODED_SOC_CODER2_COL].apply(
        lambda x, digits=soc_digit_count: soc_label_at_digits(x, n_digits=digits)
    )
    either_uncodable = (coder1_labels == UNCODABLE_LABEL) | (
        coder2_labels == UNCODABLE_LABEL
    )
    agree = (coder1_labels == coder2_labels) & ~either_uncodable
    level_label = next(
        label for digits, label in SOC_CODABILITY_LEVELS if digits == soc_digit_count
    )
    newly_resolved = agree & ~already_resolved
    soc_codability[newly_resolved] = level_label
    already_resolved = already_resolved | newly_resolved
merged["soc_codability_level"] = soc_codability


n_unmatched = merged["sic_codability_level"].isna().sum()
if n_unmatched:
    print(f"⚠️  {n_unmatched} rows failed to join to the 2k parquet - check ID formats.")

print(f"\n{'='*70}")
print("SIC vs SOC CODABILITY")
print(f"{'='*70}")

sic_cat = pd.Categorical(
    merged["sic_codability_level"], categories=SIC_LEVEL_ORDER, ordered=True
)
soc_cat = pd.Categorical(
    merged["soc_codability_level"], categories=SOC_LEVEL_ORDER, ordered=True
)
codability_crosstab = pd.crosstab(sic_cat, soc_cat, dropna=False)
codability_crosstab.index.name = "SIC codability"
codability_crosstab.columns.name = "SOC codability (dual-coding proxy)"
print("\nCross-tab of SIC codability vs SOC codability:")
print(codability_crosstab.to_string())

sic_uncodable = merged["sic_codability_level"] == "Uncodable"
soc_uncodable = merged["soc_codability_level"] == "Uncodable"
print(
    f"\nSIC uncodable: {sic_uncodable.mean():.1%} ({sic_uncodable.sum()} of {len(merged)})"
)
print(
    f"SOC uncodable (dual-coding proxy): {soc_uncodable.mean():.1%} "
    f"({soc_uncodable.sum()} of {len(merged)})"
)
print(
    f"Both uncodable: {(sic_uncodable & soc_uncodable).mean():.1%} "
    f"({(sic_uncodable & soc_uncodable).sum()})"
)
print(
    f"SIC uncodable only (SOC still codable): {(sic_uncodable & ~soc_uncodable).sum()}"
)
print(
    f"SOC uncodable only (SIC still codable): {(~sic_uncodable & soc_uncodable).sum()}"
)

non_empty_rows = codability_crosstab.sum(axis=1) > 0
non_empty_cols = codability_crosstab.sum(axis=0) > 0
contingency = codability_crosstab.loc[non_empty_rows, non_empty_cols]
if contingency.shape[0] > 1 and contingency.shape[1] > 1:
    codability_chi2, codability_p_value, codability_dof, _expected = chi2_contingency(
        contingency
    )
    print(
        f"\nChi-square test for independence: chi2={codability_chi2:.2f}, "
        f"dof={codability_dof}, p={codability_p_value:.4g}"
    )
    if codability_p_value < SIGNIFICANCE_LEVEL:
        print(
            "=> SIC and SOC codability appear related (reject independence at 5%): "
            "cases that are hard to code for one tend to be hard to code for the other."
        )
    else:
        print(
            "=> No significant evidence of a relationship between SIC and SOC "
            "codability at the 5% level."
        )
else:
    print(
        "\nNot enough variation in one of the two codability levels to run a chi-square test."
    )

# %%
# PATTERN INSIGHTS: DISAGREEMENT BY INDUSTRY (SIC SECTION) AND OCCUPATION
# Where does Coder1/Coder2 disagreement concentrate? "Disagree" here means
# the two coders did not assign the same full 4-digit SOC label, with
# Uncodable/Uncodable counted as agreement - the same criterion used for the
# digit=4 row in the reliability table above.

SECTION_MIN_N = 15  # groups smaller than this are noisy - shown but flagged


def primary_soc_code(row: pd.Series) -> object:
    """Best single SOC code for a row: the adjudicated Final code where
    available (i.e. where the coders disagreed), otherwise Coder1's code.
    """
    final = row.get(FINAL_CODE_COL)
    if pd.notna(final) and str(final).strip():
        return final
    return row[CLERICALLY_CODED_SOC_CODER1_COL]


merged["soc_major_group_digit"] = merged.apply(
    lambda row, n=1: soc_label_at_digits(primary_soc_code(row), n_digits=n),
    axis=1,
)

SOC_MAJOR_GROUP_TITLES = {
    "1": "1 Managers, Directors and Senior Officials",
    "2": "2 Professional Occupations",
    "3": "3 Associate Professional Occupations",
    "4": "4 Administrative and Secretarial Occupations",
    "5": "5 Skilled Trades Occupations",
    "6": "6 Caring, Leisure and Other Service Occupations",
    "7": "7 Sales and Customer Service Occupations",
    "8": "8 Process, Plant and Machine Operatives",
    "9": "9 Elementary Occupations",
}
merged["soc_major_group"] = (
    merged["soc_major_group_digit"].map(SOC_MAJOR_GROUP_TITLES).fillna("Uncodable")
)

merged["disagree"] = (
    merged[CLERICALLY_CODED_SOC_CODER1_COL] != merged[CLERICALLY_CODED_SOC_CODER2_COL]
)

EXAMPLE_COLS = [
    CLERICALLY_CODED_SOC_ID_COL,
    "soc2020_job_title_main_job",
    "soc2020_job_description_main_job",
    CLERICALLY_CODED_SOC_CODER1_COL,
    CLERICALLY_CODED_SOC_CODER2_COL,
    FINAL_CODE_COL,
]


def disagreement_by(group_col: str, title: str, what: str) -> None:
    """Disagreement rate per group, chi-square test and worst-group examples."""
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    stats = (
        merged.groupby(group_col)["disagree"]
        .agg(n="size", n_disagree="sum")
        .assign(disagree_rate=lambda d: (d["n_disagree"] / d["n"]).round(3))
        .sort_values("disagree_rate", ascending=False)
    )
    stats["low_sample_lt_15"] = stats["n"] < SECTION_MIN_N
    print(stats.to_string())

    ct = pd.crosstab(merged[group_col], merged["disagree"])
    if ct.shape[0] > 1:
        chi2, p_value, dof, _expected = chi2_contingency(ct)
        print(
            f"\nChi-square test (disagreement rate vs {what}): "
            f"chi2={chi2:.2f}, dof={dof}, p={p_value:.4g}"
        )
        if p_value < SIGNIFICANCE_LEVEL:
            print(
                f"=> Disagreement rate varies significantly by {what} "
                f"(p<{SIGNIFICANCE_LEVEL})."
            )
        else:
            print(
                f"=> No significant evidence that disagreement rate varies by {what} "
                f"(p>={SIGNIFICANCE_LEVEL})."
            )

    eligible = stats[~stats["low_sample_lt_15"]]
    if not eligible.empty:
        worst = eligible.index[0]
        print(f"\nExample disagreements in worst {what} group ({worst}):")
        print(
            merged[(merged[group_col] == worst) & merged["disagree"]][EXAMPLE_COLS]
            .head(5)
            .to_string(index=False)
        )


disagreement_by("sic_section", "DISAGREEMENT BY SIC SECTION (INDUSTRY)", "industry")
disagreement_by(
    "soc_major_group", "DISAGREEMENT BY OCCUPATION (SOC MAJOR GROUP)", "occupation"
)

# %%
# SURVEY ASSIST SHARED HELPERS (used by both the SOC and SIC comparisons)

SA_CODABILITY_CONFIDENCE_THRESHOLD = 0.8


def _max_candidate_likelihood(candidates: object) -> float:
    """Highest 'likelihood' across a row's alt_{sic,soc}_candidates, or NaN."""
    if not isinstance(candidates, list | tuple | np.ndarray):
        return float("nan")
    likelihoods = [
        candidate.get("likelihood")
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("likelihood") is not None
    ]
    return max(likelihoods) if likelihoods else float("nan")


def blank_low_confidence_initial_code(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # noqa: PLR0913
    df: pd.DataFrame,
    codes_col: str,
    alt_codes_col: str,
    likelihood_col: str | None,
    label: str,
    apply_threshold: bool = True,
) -> pd.DataFrame:
    """Blank `codes_col` wherever its likelihood is below
    SA_CODABILITY_CONFIDENCE_THRESHOLD or missing (i.e. not unambiguously codable).

    The likelihood is read from `likelihood_col` when present (one-prompt
    pipeline), otherwise from the best `alt_codes_col` candidate likelihood.

    When apply_threshold=False (e.g., for SIC), codes are not blanked based on
    confidence thresholds; only returned as-is.
    """
    if not apply_threshold:
        # For code types like SIC with different strategies, return as-is
        return df

    if likelihood_col is not None and likelihood_col in df.columns:
        confidence = pd.to_numeric(df[likelihood_col], errors="coerce")
        source = likelihood_col
    else:
        confidence = df[alt_codes_col].apply(_max_candidate_likelihood)
        source = alt_codes_col
    print(
        f"{label} likelihood source: {source} "
        f"({confidence.isna().sum()} of {len(df)} missing)"
    )
    has_initial_code = df[codes_col].fillna("").astype(str).str.strip().ne("")
    # Missing likelihood = not confident
    low_confidence = confidence.isna() | (
        confidence < SA_CODABILITY_CONFIDENCE_THRESHOLD
    ).fillna(False)
    not_unambiguously_codable = has_initial_code & low_confidence
    print(
        f"Blanking {not_unambiguously_codable.sum()} of {len(df)} {label} "
        f"initial_code value(s) with likelihood below "
        f"{SA_CODABILITY_CONFIDENCE_THRESHOLD} or missing (not unambiguously codable)."
    )
    df.loc[not_unambiguously_codable, codes_col] = ""
    return df


def soc_major_group_label(code_set: set) -> str:
    """SOC major group title for a single-code set, else Ambiguous/Uncodable."""
    if len(code_set) != 1:
        return "Ambiguous/Uncodable"
    return SOC_MAJOR_GROUP_TITLES.get(next(iter(code_set))[:1], "Uncodable")


def run_sa_comparison(  # noqa: PLR0913  # pylint: disable=too-many-arguments,too-many-locals
    *,
    code_type: str,
    sa_path: str,
    sa_id_col: str,
    codes_col: str,
    alt_codes_col: str,
    likelihood_col: str | None,
    digit_levels: list[int],
    build_truth,  # n -> DataFrame[unique_id, clerical_codes]
    group_label,  # set -> str, for the distribution table
    group_name: str,
    sort_distribution: bool = False,
    apply_threshold: bool = True,
) -> None:
    """Compare Survey Assist initial codes with clerical truth by digit level."""
    title = f"SURVEY ASSIST {code_type} PERFORMANCE COMPARISON"
    try:
        sa = pd.read_parquet(sa_path, dtype_backend="numpy_nullable")
    except FileNotFoundError:
        print(f"\n{'='*70}")
        print(f"{title} - SKIPPED")
        print(f"{'='*70}")
        print(
            f"'{sa_path}' not found. Update the SA_{code_type} path/column "
            f"constants in the CONFIGURATION section once Survey Assist's "
            f"{code_type} output for this subset is available, then re-run."
        )
        return
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")

    # Standardise column names
    sa = sa.rename(columns={sa_id_col: TWO_K_ID_COL})
    sa[TWO_K_ID_COL] = sa[TWO_K_ID_COL].astype(str)
    sa = blank_low_confidence_initial_code(
        sa,
        codes_col,
        alt_codes_col,
        likelihood_col=likelihood_col,
        label=f"Survey Assist {code_type}",
        apply_threshold=apply_threshold,
    )

    # Run performance evaluation with INITIAL_CODE only
    top = max(digit_levels)
    summary = []
    full = pd.DataFrame()

    for n in sorted(digit_levels, reverse=True):
        truth = build_truth(n)
        model = prep_model_codes(
            sa,
            codes_col=codes_col,
            alt_codes_col=None,
            code_type=code_type,
            digits=n,
            out_col="model_codes",
        )
        combined = truth.merge(model, on="unique_id", how="inner")
        n_missing = len(truth) - len(combined)
        if n_missing:
            print(
                f"⚠️  {n_missing} clerical rows had no matching Survey Assist "
                f"{code_type} record (ID mismatch) - excluded from this comparison."
            )
        metrics = calc_simple_metrics(
            combined,
            truth_col="clerical_codes",
            initial_model_col="model_codes",
            final_model_col=None,
        )
        summary.append(
            {
                "digits": n,
                "n_records": len(combined),
                "f1": round(metrics.ambiguity_metrics.f1, 4),
                "sa_codability": round(
                    metrics.codability_metrics.initial_codable_prop, 4
                ),
                "MM_accuracy": round(
                    metrics.initial_accuracy_metrics.accuracy_mm_total, 4
                ),
                "OO_accuracy": round(
                    metrics.initial_accuracy_metrics.accuracy_oo_unambiguous, 4
                ),
            }
        )
        if n == top:
            print(metrics.report_metrics())
            full = combined

    print(f"\nSurvey Assist vs clerical truth, by {code_type} digit level:")
    print(pd.DataFrame(summary).to_string(index=False))

    print(f"\n{'='*70}")
    print(
        f"POST-HOC ANALYSIS: Model's Pick vs Clerical Truth ({top}-digit {code_type})"
    )
    print(f"{'='*70}")

    matches = full.apply(
        lambda row: len(row["model_codes"]) == 1
        and (row["model_codes"] <= row["clerical_codes"]),
        axis=1,
    )
    print("\nModel's initial_code vs clerical truth:")
    print(
        f"  Model pick matches clerical truth: {matches.sum()} of {len(full)} "
        f"({100 * matches.mean():.1f}%)"
    )
    print(f"  Model pick does not match: {(~matches).sum()}")

    # Distribution comparison
    dist = (
        pd.concat(
            [
                full["clerical_codes"]
                .apply(group_label)
                .value_counts(normalize=True)
                .rename("Clerical truth"),
                full["model_codes"]
                .apply(group_label)
                .value_counts(normalize=True)
                .rename("Survey Assist"),
            ],
            axis=1,
        )
        .fillna(0)
        .mul(100)
        .round(1)
    )
    if sort_distribution:
        dist = dist.sort_index()
    print(f"\n{group_name} distribution, clerical truth vs Survey Assist (%):")
    print(dist.to_string())


# %%
# SURVEY ASSIST PERFORMANCE COMPARISON - SOC

truth_input_df = merged[["unique_id"]].copy()
truth_input_df["unique_id"] = truth_input_df["unique_id"].astype(str)
# Already-clean full-length codes (or the Uncodable sentinel)
truth_input_df["consensus_code"] = merged.apply(primary_soc_code, axis=1)

run_sa_comparison(
    code_type="SOC",
    sa_path=SA_DATA_PATH,
    sa_id_col=SA_SOC_ID_COL,
    codes_col=SA_SOC_CODES_COL,
    alt_codes_col=SA_SOC_ALT_CODES_COL,
    likelihood_col=SA_SOC_LIKELIHOOD_COL,
    digit_levels=SOC_DIGIT_LEVELS,
    build_truth=lambda n: prep_clerical_codes(
        truth_input_df,
        clerical_col="consensus_code",
        code_type="SOC",
        digits=n,
        out_col="clerical_codes",
    ),
    group_label=soc_major_group_label,
    group_name="SOC major group",
)

# %%
# SURVEY ASSIST PERFORMANCE COMPARISON - SIC

SIC_DIGIT_LEVELS = sorted(
    {digits for digits, _label in SIC_CODABILITY_LEVELS if digits > 0}
)

# UK SIC 2007 sections by division (first 2 digits)
_SIC_SECTION_RANGES = [
    ("Agriculture, Forestry, and Fishing", 1, 9),
    ("Mining", 10, 14),
    ("Construction", 15, 17),
    ("Manufacturing", 20, 39),
    ("Transportation, Communications", 40, 49),
    ("Wholesale Trade", 50, 51),
    ("Retail Trade", 52, 59),
    ("Finance, Insurance, and Real Estate", 60, 67),
    ("Services", 70, 89),
    ("Public Administration", 91, 97),
    ("Nonclassifiable Establishments", 99, 99),
]


def sic_section_from_code(code: str) -> str:
    """Map a clean SIC code to its SIC 2007 section letter."""
    try:
        division = int(str(code)[:2])
    except ValueError:
        return "Uncodable"
    for section, low, high in _SIC_SECTION_RANGES:
        if low <= division <= high:
            return section
    return "Uncodable"


def sic_truth_at_digits(raw: object, n_digits: int) -> set:
    """Clean the 2k `clerical_codes` candidate list to a set of n-digit SIC codes."""
    codes = clerical_codes_to_set(raw)
    if not codes:
        return set()
    cleaned, _invalid = get_clean_n_digit_codes(
        sorted(codes), n=n_digits, code_type="SIC"
    )
    return set(cleaned)


def sic_section_label(code_set: set) -> str:
    """SIC section letter for a single-code set, else Ambiguous/Uncodable."""
    if len(code_set) != 1:
        return "Ambiguous/Uncodable"
    return sic_section_from_code(next(iter(code_set)))


# SIC clerical truth comes from the 2k parquet, so only matched rows have it
sic_truth_input_df = merged.loc[is_matched, [TWO_K_ID_COL, TWO_K_CLERICAL_CODES]].copy()
sic_truth_input_df[TWO_K_ID_COL] = sic_truth_input_df[TWO_K_ID_COL].astype(str)


def build_sic_truth(n: int) -> pd.DataFrame:
    """SIC clerical truth as sets of n-digit codes, keyed by unique_id."""
    return pd.DataFrame(
        {
            "unique_id": sic_truth_input_df[TWO_K_ID_COL],
            "clerical_codes": sic_truth_input_df[TWO_K_CLERICAL_CODES].apply(
                lambda raw: sic_truth_at_digits(raw, n_digits=n)
            ),
        }
    )


run_sa_comparison(
    code_type="SIC",
    sa_path=SA_SIC_DATA_PATH,
    sa_id_col=SA_SIC_ID_COL,
    codes_col=SA_SIC_CODES_COL,
    alt_codes_col=None,
    likelihood_col=None,
    digit_levels=SIC_DIGIT_LEVELS,
    build_truth=build_sic_truth,
    group_label=sic_section_label,
    group_name="SIC section",
    sort_distribution=True,
    apply_threshold=False,
)

print("\n✓ Analysis complete!")
