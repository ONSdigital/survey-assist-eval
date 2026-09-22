#!/usr/bin/env python3
"""Dual-Coded SOC/SIC Analysis
Convert occupation and industry classifications from dual-coded datasets.
"""

# %%
import logging
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score

from survey_assist_eval.data_cleaning.code_standard import (
    SIC_CODABILITY_LEVELS,
    SOC_CODABILITY_LEVELS,
    get_clean_n_digit_codes,
    get_codability_level,
)

# ============================================================================
# SETUP: Pandas Display Options
# ============================================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", 10)

# %%
# Update file paths and column names based on your actual data.

# File paths
DATA_DIR = Path("data")
NEW_DATA_PATH = DATA_DIR / "evaluation-pipeline_original_datasets_sic_2k_comparison_soc_2k.xlsx"
TWO_K_PATH = DATA_DIR / "evaluation-pipeline_original_datasets_sic_2k_sic_2k_test_data.parquet"

TWO_K_CLERICAL_CODES = "clerical_codes"        # List of candidate codes

# Survey Assist's own SOC output for this same subset
SA_DATA_PATH = DATA_DIR / "evaluation-pipeline_yavuz_soc_STG2.parquet" # Pipeline run by Peter
SA_ID_COL = "Unique_identifier"
SA_CODES_COL = "initial_code"
SA_ALT_CODES_COL = "alt_soc_candidates"

SIGNIFICANCE_LEVEL = 0.05


def show_value_counts(df: pd.DataFrame, col: str, label: str, top_n: int = 10) -> None:
    """Show top N value counts for a column."""
    if col not in df.columns:
        print(f"⚠️  Column '{col}' not found in {label}")
        return

    print(f"\n{label} :: '{col}' (top {top_n} values):")
    try:
        counts = df[col].value_counts(dropna=False).head(top_n)
        print(counts.to_string())
    except TypeError:
        # Handle unhashable types (lists, arrays, etc.)
        print("Column contains unhashable values (lists/arrays). Flattening and showing unique codes:")
        all_codes = []
        for item in df[col].dropna():
            if isinstance(item, list | tuple):
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
# ============================================================================
# DATA LOADING
# ============================================================================

print("Loading data...")
new_data_sheets = pd.read_excel(NEW_DATA_PATH, sheet_name=None, dtype=str)
print(f"✓ Loaded {len(new_data_sheets)} sheet(s): {list(new_data_sheets.keys())}")

# Load the configured sheet
# Define the sheet name explicitly from the workbook
NEW_DATA_SHEET = "Comparisons"  # matches the key in new_data_sheets
new_data_df = new_data_sheets[NEW_DATA_SHEET]
two_k_df = pd.read_parquet(TWO_K_PATH, dtype_backend='numpy_nullable')

# Drop fully-empty junk columns left over from the Excel export (e.g. "Unnamed: 10").
empty_cols = [col for col in new_data_df.columns if new_data_df[col].isna().all()]
if empty_cols:
    print(f"Dropping empty column(s) from {NEW_DATA_SHEET}: {empty_cols}")
    new_data_df = new_data_df.drop(columns=empty_cols)

# Anonymise the two clerical coders immediately on load
new_data_df = new_data_df.rename(columns={
    "Carol": "Coder1",
    "Carol_Comments": "Coder1_Comments",
    "Lynne": "Coder2",
    "Lynne_Comments": "Coder2_Comments",
})

# %%
# ============================================================================
# NEW DATA PREVIEW & PROFILE
# ============================================================================

print(f"\n{'='*70}")
print("NEW DATA PREVIEW: Comparisons Sheet")
print(f"{'='*70}")
print(f"\n📋 Columns ({len(new_data_df.columns)} total):")
for i, col in enumerate(new_data_df.columns, 1):
    print(f"  {i:2}. {col}")

print(f"\n📊 Shape: {new_data_df.shape[0]} rows x {new_data_df.shape[1]} columns")
# print(new_data_df.head(3).to_string())

# %%
# ============================================================================
# 2k.parquet - PREVIEW & PROFILE
# ============================================================================

print(f"\n{'='*70}")
print("2k.parquet PREVIEW")
print(f"{'='*70}")
print(f"\n📋 Columns ({len(two_k_df.columns)} total):")
for i, col in enumerate(two_k_df.columns, 1):
    print(f"  {i:2}. {col}")

print(f"\n📊 Shape: {two_k_df.shape[0]} rows x {two_k_df.shape[1]} columns")
# print(two_k_df.head(3).to_string())



# %%
# ============================================================================
# CODE COLUMN VALUE SAMPLES
# ============================================================================
# Showing value distributions for configured SIC/SOC columns.

# Match the actual column names in the Comparisons sheet
NEW_DATA_ID_COL = "Unique_identifier"
NEW_DATA_CODER1_COL = "Coder1"
NEW_DATA_CODER2_COL = "Coder2"

# Also used in later cells
TWO_K_ID_COL = "unique_id"
TWO_K_SIC_COL = "sic2007_employee"
TWO_K_SIC_IND_COL = "sic_ind1"

print(f"\n{'='*70}")
print("NEW DATA CODE VALUES")
print(f"{'='*70}")

show_value_counts(new_data_df, NEW_DATA_CODER1_COL, "new_data :: Coder 1", top_n=15)
show_value_counts(new_data_df, NEW_DATA_CODER2_COL, "new_data :: Coder 2", top_n=15)

# Check ID matching
mask = new_data_df.Unique_identifier.isin(two_k_df.unique_id)
print(f"\nNumber of matching unique identifiers: {mask.sum()}")
print(f"Number of non-matching unique identifiers: {(~mask).sum()}")

print(f"\n{'='*70}")
print("2k.parquet CODE VALUES")
print(f"{'='*70}")

show_value_counts(two_k_df, TWO_K_SIC_COL, "2k.parquet :: SIC codes", top_n=15)
# show_value_counts(two_k_df, TWO_K_SIC_IND_COL, "2k.parquet :: SIC indicators", top_n=15)
show_value_counts(two_k_df, TWO_K_CLERICAL_CODES, "2k.parquet :: Clerical codes", top_n=10)

# %%
# ============================================================================
# INTER-RATER RELIABILITY: CODER AGREEMENT ANALYSIS
# ============================================================================
# Compute inter-rater reliability metrics (Cohen's kappa, % agreement),
# at each SOC digit level (1/2/3/4-digit), treating "uncodeable" as a
# genuine category rather than missing data.


UNCODEABLE_LABEL = "Uncodable"
SOC_DIGIT_LEVELS = sorted({digits for digits, _label in SOC_CODABILITY_LEVELS if digits > 0})

# get_clean_n_digit_codes logs a warning for every cell it can't parse as a
# code (e.g. "uncodeable" itself) - that's expected here and would be very
# noisy at ~90 occurrences x 4 digit levels, so quiet it for this section.
_code_standard_logger = logging.getLogger(
    "survey_assist_eval.data_cleaning.code_standard"
)
_previous_log_level = _code_standard_logger.level
_code_standard_logger.setLevel(logging.ERROR)

try:
    def soc_label_at_digits(raw: object, n: int, unrecognised: set) -> str:
        """Return the n-digit SOC code for a single coder's cell, or the
        Uncodeable sentinel if the cell is blank, 'uncodeable', or otherwise
        not a valid SOC code (any such value is also recorded in
        `unrecognised` so it can be surfaced separately from genuine
        uncodeable calls).
        """
        if pd.isna(raw):
            return UNCODEABLE_LABEL
        raw_str = str(raw).strip()
        if not raw_str:
            return UNCODEABLE_LABEL

        cleaned, invalid = get_clean_n_digit_codes([raw_str], n=n, code_type="SOC")
        if invalid and raw_str.lower() not in {"uncodeable", "uncodable"}:
            unrecognised.add(raw_str)
        if len(cleaned) == 1:
            return next(iter(cleaned))
        # empty (invalid/uncodeable) or, unexpectedly, >1 candidate
        return UNCODEABLE_LABEL

    print(f"\n{'='*70}")
    print("INTER-RATER RELIABILITY: Coder1 vs Coder2 (by SOC digit level)")
    print(f"{'='*70}")
    print(f"Total records: {len(new_data_df)}")

    digit_level_summary = []
    unrecognised_values: set = set()

    for n in SOC_DIGIT_LEVELS:
        coder1_labels = new_data_df[NEW_DATA_CODER1_COL].apply(
            soc_label_at_digits, n=n, unrecognised=unrecognised_values
        )
        coder2_labels = new_data_df[NEW_DATA_CODER2_COL].apply(
            soc_label_at_digits, n=n, unrecognised=unrecognised_values
        )

        n_agree = (coder1_labels == coder2_labels).sum()
        pct_agree = 100 * n_agree / len(new_data_df)
        kappa = cohen_kappa_score(coder1_labels, coder2_labels)

        digit_level_summary.append(
            {
                "digits": n,
                "n_records": len(new_data_df),
                "n_agree": int(n_agree),
                "pct_agree": round(pct_agree, 1),
                "cohens_kappa": round(kappa, 4),
            }
        )

        if n == max(SOC_DIGIT_LEVELS):
            # keep the full 4-digit labels around for the "Agree" cross-check below
            coder1_full = coder1_labels
            coder2_full = coder2_labels

    digit_level_df = pd.DataFrame(digit_level_summary).sort_values(
        "digits", ascending=False
    )
    print("\nAgreement by SOC digit level (uncodeable treated as its own category):")
    print(digit_level_df.to_string(index=False))

    if unrecognised_values:
        print(
            f"\n⚠️  {len(unrecognised_values)} distinct coder value(s) were neither a "
            "valid SOC code nor 'uncodeable' - check these for typos, they are "
            "currently being folded into the Uncodeable category:"
        )
        print(sorted(unrecognised_values)[:20])

    # Cross-check against the clerical team's own row-level "Agree" flag, if present.
    if "Agree" in new_data_df.columns:
        our_agree = coder1_full == coder2_full
        their_agree = new_data_df["Agree"].astype(str).str.strip().str.lower().isin(
            {"true", "1", "yes", "y"}
        )
        mismatches = (our_agree != their_agree).sum()
        print(
            f"\nCross-check vs sheet's own 'Agree' column: {mismatches} of "
            f"{len(new_data_df)} rows disagree with our recomputed agreement "
            "(investigate before trusting the figures above if this is large)."
        )
finally:
    _code_standard_logger.setLevel(_previous_log_level)

# %%
# ============================================================================
# ID OVERLAP CHECK
# ============================================================================
# Check if both files cover the same records.

check_id_overlap(
    new_data_df, NEW_DATA_ID_COL, f"new_data :: {NEW_DATA_SHEET}",
    two_k_df, TWO_K_ID_COL, "2k.parquet"
)

# %%
# ============================================================================
# SIC / SOC CODABILITY RELATIONSHIP
# ============================================================================


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


_code_standard_logger.setLevel(logging.ERROR)
try:
    two_k_df["sic_codability_level"] = two_k_df[TWO_K_CLERICAL_CODES].apply(
        lambda codes: get_codability_level(clerical_codes_to_set(codes), code_type="SIC")
    )

    # Finest SOC digit level at which Coder1 and Coder2 agree, per row.
    soc_codability = pd.Series(UNCODEABLE_LABEL, index=new_data_df.index)
    already_resolved = pd.Series(False, index=new_data_df.index)
    for n in sorted(SOC_DIGIT_LEVELS, reverse=True):
        coder1_labels = new_data_df[NEW_DATA_CODER1_COL].apply(
            soc_label_at_digits, n=n, unrecognised=set()
        )
        coder2_labels = new_data_df[NEW_DATA_CODER2_COL].apply(
            soc_label_at_digits, n=n, unrecognised=set()
        )
        either_uncodeable = (coder1_labels == UNCODEABLE_LABEL) | (
            coder2_labels == UNCODEABLE_LABEL
        )
        agree = (coder1_labels == coder2_labels) & ~either_uncodeable
        level_label = next(label for digits, label in SOC_CODABILITY_LEVELS if digits == n)
        newly_resolved = agree & ~already_resolved
        soc_codability[newly_resolved] = level_label
        already_resolved = already_resolved | newly_resolved
    new_data_df["soc_codability_level"] = soc_codability
finally:
    _code_standard_logger.setLevel(_previous_log_level)

merged = new_data_df.merge(
    two_k_df[[TWO_K_ID_COL, "sic_section", "sic_codability_level"]],
    left_on=NEW_DATA_ID_COL,
    right_on=TWO_K_ID_COL,
    how="left",
)
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
print(f"\nSIC uncodable: {sic_uncodable.mean():.1%} ({sic_uncodable.sum()} of {len(merged)})")
print(
    f"SOC uncodable (dual-coding proxy): {soc_uncodable.mean():.1%} "
    f"({soc_uncodable.sum()} of {len(merged)})"
)
print(
    f"Both uncodable: {(sic_uncodable & soc_uncodable).mean():.1%} "
    f"({(sic_uncodable & soc_uncodable).sum()})"
)
print(f"SIC uncodable only (SOC still codable): {(sic_uncodable & ~soc_uncodable).sum()}")
print(f"SOC uncodable only (SIC still codable): {(~sic_uncodable & soc_uncodable).sum()}")

non_empty_rows = codability_crosstab.sum(axis=1) > 0
non_empty_cols = codability_crosstab.sum(axis=0) > 0
contingency = codability_crosstab.loc[non_empty_rows, non_empty_cols]
if contingency.shape[0] > 1 and contingency.shape[1] > 1:
    chi2, p_value, dof, _expected = chi2_contingency(contingency)
    print(
        f"\nChi-square test for independence: chi2={chi2:.2f}, dof={dof}, p={p_value:.4g}"
    )
    if p_value < SIGNIFICANCE_LEVEL:
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
    print("\nNot enough variation in one of the two codability levels to run a chi-square test.")

# %%
# ============================================================================
# PATTERN INSIGHTS: DISAGREEMENT BY INDUSTRY (SIC SECTION) AND OCCUPATION
# ============================================================================
# Where does Coder1/Coder2 disagreement concentrate? "Disagree" here means
# the two coders did not land on the exact same 4-digit SOC unit group
# (i.e. soc_codability_level is anything other than "Unit group (4-digits)"),
# the same criterion used for the digit=4 row in the reliability table above.

SECTION_MIN_N = 15  # groups smaller than this are noisy - shown but flagged

_code_standard_logger.setLevel(logging.ERROR)
try:
    def primary_soc_code(row: pd.Series) -> object:
        """Best single SOC code for a row: the adjudicated Final code where
        available (i.e. where the coders disagreed), otherwise Coder1's code
        (arbitrary - Coder1 and Coder2 agree on ~97% of rows so it barely
        matters which one is used as the "primary" occupation label).
        """
        final = row.get("Final code")
        if pd.notna(final) and str(final).strip():
            return final
        return row[NEW_DATA_CODER1_COL]

    merged["soc_major_group_digit"] = merged.apply(
        lambda row: soc_label_at_digits(primary_soc_code(row), 1, set()), axis=1
    )
finally:
    _code_standard_logger.setLevel(_previous_log_level)

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
merged["soc_major_group"] = merged["soc_major_group_digit"].map(SOC_MAJOR_GROUP_TITLES).fillna(
    "Uncodable"
)

merged["disagree"] = merged["soc_codability_level"] != "Unit group (4-digits)"

print(f"\n{'='*70}")
print("DISAGREEMENT BY SIC SECTION (INDUSTRY)")
print(f"{'='*70}")
section_stats = (
    merged.groupby("sic_section")["disagree"]
    .agg(n="size", n_disagree="sum")
    .assign(disagree_rate=lambda d: (d["n_disagree"] / d["n"]).round(3))
    .sort_values("disagree_rate", ascending=False)
)
section_stats["low_sample_lt_15"] = section_stats["n"] < SECTION_MIN_N
print(section_stats.to_string())

section_ct = pd.crosstab(merged["sic_section"], merged["disagree"])
if section_ct.shape[0] > 1:
    chi2, p_value, dof, _expected = chi2_contingency(section_ct)
    print(
        f"\nChi-square test (disagreement rate vs SIC section): "
        f"chi2={chi2:.2f}, dof={dof}, p={p_value:.4g}"
    )
    print(
        f"=> Disagreement rate varies significantly by industry (p<{SIGNIFICANCE_LEVEL})."
        if p_value < SIGNIFICANCE_LEVEL
        else f"=> No significant evidence that disagreement rate varies by industry (p>={SIGNIFICANCE_LEVEL})."
    )

print(f"\n{'='*70}")
print("DISAGREEMENT BY OCCUPATION (SOC MAJOR GROUP)")
print(f"{'='*70}")
major_group_stats = (
    merged.groupby("soc_major_group")["disagree"]
    .agg(n="size", n_disagree="sum")
    .assign(disagree_rate=lambda d: (d["n_disagree"] / d["n"]).round(3))
    .sort_values("disagree_rate", ascending=False)
)
major_group_stats["low_sample_lt_15"] = major_group_stats["n"] < SECTION_MIN_N
print(major_group_stats.to_string())

major_group_ct = pd.crosstab(merged["soc_major_group"], merged["disagree"])
if major_group_ct.shape[0] > 1:
    chi2, p_value, dof, _expected = chi2_contingency(major_group_ct)
    print(
        f"\nChi-square test (disagreement rate vs SOC major group): "
        f"chi2={chi2:.2f}, dof={dof}, p={p_value:.4g}"
    )
    print(
        f"=> Disagreement rate varies significantly by occupation group (p<{SIGNIFICANCE_LEVEL})."
        if p_value < SIGNIFICANCE_LEVEL
        else f"=> No significant evidence that disagreement rate varies by occupation group (p>={SIGNIFICANCE_LEVEL})."
    )

# A handful of concrete example disagreements from the worst-performing
# section and major group, so the numbers above can be read alongside what
# the actual job titles/descriptions/comments look like.
EXAMPLE_COLS = [
    NEW_DATA_ID_COL,
    "soc2020_job_title_main_job",
    "soc2020_job_description_main_job",
    NEW_DATA_CODER1_COL,
    NEW_DATA_CODER2_COL,
    "Final code",
]
eligible_sections = section_stats[~section_stats["low_sample_lt_15"]]
if not eligible_sections.empty:
    worst_section = eligible_sections.index[0]
    print(f"\nExample disagreements in worst SIC section ({worst_section}):")
    print(
        merged[(merged["sic_section"] == worst_section) & merged["disagree"]][EXAMPLE_COLS]
        .head(5)
        .to_string(index=False)
    )

eligible_groups = major_group_stats[~major_group_stats["low_sample_lt_15"]]
if not eligible_groups.empty:
    worst_group = eligible_groups.index[0]
    print(f"\nExample disagreements in worst SOC major group ({worst_group}):")
    print(
        merged[(merged["soc_major_group"] == worst_group) & merged["disagree"]][EXAMPLE_COLS]
        .head(5)
        .to_string(index=False)
    )

# %%
# ============================================================================
# SURVEY ASSIST PERFORMANCE COMPARISON
# ============================================================================

if not SA_DATA_PATH.exists():
    print(f"\n{'='*70}")
    print("SURVEY ASSIST PERFORMANCE COMPARISON - SKIPPED")
    print(f"{'='*70}")
    print(
        f"'{SA_DATA_PATH}' not found. Update SA_DATA_PATH (and SA_CODES_COL/"
        "SA_ALT_CODES_COL if needed) in the CONFIGURATION section once "
        "Survey Assist's SOC output for this subset is available, then re-run."
    )
else:
    from survey_assist_eval.data_cleaning.prep_data import (
        prep_clerical_codes,
        prep_model_codes,
    )
    from survey_assist_eval.evaluation.metrics import calc_simple_metrics

    print(f"\n{'='*70}")
    print("SURVEY ASSIST PERFORMANCE COMPARISON")
    print(f"{'='*70}")

    sa_df = pd.read_parquet(SA_DATA_PATH, dtype_backend='numpy_nullable')
    if SA_ID_COL != "unique_id":
        sa_df = sa_df.rename(columns={SA_ID_COL: "unique_id"})
    # Match dtype with truth_input_df's "unique_id" (built from two_k_df, also
    # read with dtype_backend='numpy_nullable') so the merge below can't
    # silently under-match on a string-dtype mismatch between the two files.
    sa_df["unique_id"] = sa_df["unique_id"].astype(str)

    # ========================================================================
    # Run performance evaluation with INITIAL_CODE only
    # ========================================================================

    truth_input_df = merged[["unique_id"]].copy()
    truth_input_df["unique_id"] = truth_input_df["unique_id"].astype(str)
    truth_input_df["consensus_code"] = merged.apply(primary_soc_code, axis=1)

    _code_standard_logger.setLevel(logging.ERROR)
    try:
        digit_perf_summary = []
        for n in sorted(SOC_DIGIT_LEVELS, reverse=True):
            truth_codes_df = prep_clerical_codes(
                truth_input_df,
                clerical_col="consensus_code",
                code_type="SOC",
                digits=n,
                out_col="clerical_codes",
            )

            # STANDARD prep_model_codes
            model_codes_df = prep_model_codes(
                sa_df,
                codes_col=SA_CODES_COL,
                alt_codes_col=SA_ALT_CODES_COL,
                code_type="SOC",
                digits=n,
                out_col="model_codes",
            )

            combined = truth_codes_df.merge(model_codes_df, on="unique_id", how="inner")
            n_unmatched_sa = len(truth_codes_df) - len(combined)
            if n_unmatched_sa:
                print(
                    f"⚠️  {n_unmatched_sa} clerical rows had no matching Survey Assist "
                    "record (ID mismatch) - excluded from this comparison."
                )

            metrics = calc_simple_metrics(
                combined,
                truth_col="clerical_codes",
                initial_model_col="model_codes",
                final_model_col=None,
            )
            digit_perf_summary.append(
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
            if n == max(SOC_DIGIT_LEVELS):
                print(metrics.report_metrics())
                full_digit_combined = combined

        print("\nSurvey Assist vs clerical truth, by SOC digit level:")
        print(pd.DataFrame(digit_perf_summary).to_string(index=False))

        # ====================================================================
        # POST-HOC ANALYSIS: Does the model's single pick exactly match the
        # clerical truth at the 4-digit level? (full_digit_combined is fixed
        # at the finest digit level, captured above.) Note this is the same
        # comparison as the "OO_accuracy"/"MM_accuracy" columns in the
        # digits=4 row of the table above, just shown here as raw counts
        # instead of a rate - there is no separate SOC "shortlist" to check
        # against, since Coder1/Coder2 each record a single code, not a list
        # of candidates (unlike the 2k parquet's SIC clerical_codes).
        # ====================================================================

        print(f"\n{'='*70}")
        print("POST-HOC ANALYSIS: Model's Pick vs Clerical Truth (4-digit)")
        print(f"{'='*70}")

        comparison = full_digit_combined.copy()
        comparison["model_pick_matches_truth"] = comparison.apply(
            lambda row: len(row["model_codes"]) == 1
            and (row["model_codes"] <= row["clerical_codes"]),
            axis=1,
        )

        print("\nModel's initial_code vs clerical truth:")
        print(f"  Model pick matches clerical truth: {comparison['model_pick_matches_truth'].sum()} of {len(comparison)} ({100*comparison['model_pick_matches_truth'].mean():.1f}%)")
        print(f"  Model pick does not match: {(~comparison['model_pick_matches_truth']).sum()}")

        # Distribution comparison
        def _major_group_label(code_set: set) -> str:
            if len(code_set) != 1:
                return "Ambiguous/Uncodable"
            return SOC_MAJOR_GROUP_TITLES.get(next(iter(code_set))[:1], "Uncodable")

        truth_major_dist = (
            full_digit_combined["clerical_codes"].apply(_major_group_label).value_counts(
                normalize=True
            )
        )
        sa_major_dist = (
            full_digit_combined["model_codes"].apply(_major_group_label).value_counts(
                normalize=True
            )
        )
        dist_compare = (
            pd.concat(
                [
                    truth_major_dist.rename("Clerical truth"),
                    sa_major_dist.rename("Survey Assist"),
                ],
                axis=1,
            )
            .fillna(0)
            .mul(100)
            .round(1)
        )
        print("\nSOC major group distribution, clerical truth vs Survey Assist (%):")
        print(dist_compare.to_string())

    finally:
        _code_standard_logger.setLevel(_previous_log_level)

print("\n✓ Analysis complete!")
