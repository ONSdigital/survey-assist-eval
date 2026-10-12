#!/usr/bin/env python3
"""Dual-Coded SOC/SIC Analysis
Convert occupation and industry classifications from dual-coded datasets.
"""

# pylint: disable=invalid-name,too-many-locals

# %%
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score

from survey_assist_eval.data_cleaning.code_standard import (
    SIC_CODABILITY_LEVELS,
    SIC_EXPECTED_CODE_LENGTH,
    SOC_CODABILITY_LEVELS,
    SOC_EXPECTED_CODE_LENGTH,
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
# Define source file paths for clerical and Survey Assist coded data

# File paths
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")

clerically_coded_soc_path = (
    f"gs://{bucket_name}/evaluation-pipeline/original_datasets/sic_2k/"
    "comparison_soc_2k.xlsx"
)
clerically_coded_sic_path = (
    f"gs://{bucket_name}/evaluation-pipeline/original_datasets/sic_2k/"
    "sic_2k_test_data.parquet"
)
sa_coded_soc_path = (
    f"gs://{bucket_name}/evaluation-pipeline/dual_coded_2k/soc/" "STG2.parquet"
)
sa_coded_sic_path = (
    f"gs://{bucket_name}/evaluation-pipeline/dual_coded_2k/sic/" "STG2.parquet"
)

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
            if isinstance(item, list | tuple | set | np.ndarray):
                all_codes.extend(item)
            else:
                all_codes.append(item)
        counts = pd.Series(all_codes).value_counts().head(top_n)
        print(counts.to_string())


# %%
# DATA LOADING
print("Loading data...")
clerically_coded_soc_df = pd.read_excel(
    clerically_coded_soc_path, sheet_name="Comparisons", dtype=str
).rename(columns={"Unique_identifier": "unique_id"})

clerically_coded_sic_df = pd.read_parquet(
    clerically_coded_sic_path, dtype_backend="numpy_nullable"
)

# Survey Assist outputs - loaded once here and reused by every later cell
sa_coded_soc_df = pd.read_parquet(
    sa_coded_soc_path, dtype_backend="numpy_nullable"
).rename(columns={"Unique_identifier": "unique_id"})
sa_coded_sic_df = pd.read_parquet(
    sa_coded_sic_path, dtype_backend="numpy_nullable"
).rename(columns={"Unique_identifier": "unique_id"})

# %%
# Clean up clerically coded SOC dataframe
empty_cols = [
    col
    for col in clerically_coded_soc_df.columns
    if clerically_coded_soc_df[col].isna().all()
]
if empty_cols:
    print(f"Dropping empty column(s) from Comparisons: {empty_cols}")
    clerically_coded_soc_df = clerically_coded_soc_df.drop(columns=empty_cols)

# Replace the two clerical coders with generic labels
clerically_coded_soc_df = clerically_coded_soc_df.rename(
    columns={
        "Carol": "coder1",
        "Carol_Comments": "coder1_Comments",
        "Lynne": "coder2",
        "Lynne_Comments": "coder2_Comments",
        "Final code": "final_code",
    }
)


# %%
# Clean every clerical SOC code column once, treat them as full-length SOC codes for rest
def clean_soc_code(raw: object) -> set[str]:
    """Return the full-length clean SOC code for a coder's cell."""
    if pd.isna(raw) or (
        raw in ["uncodeable", "Uncodeable", "0000"]
    ):  # surpress invalid codes logging
        return set()
    cleaned, _ = get_clean_n_digit_codes(
        raw, n=SOC_EXPECTED_CODE_LENGTH, code_type="SOC"
    )
    return cleaned


for col in (
    "coder1",
    "coder2",
    "final_code",
):
    clerically_coded_soc_df[f"{col}_clean"] = (
        clerically_coded_soc_df[col]
        .replace({"6321": "6231"})  # Correct typo
        .apply(clean_soc_code)
    )

coder1_clean_col = "coder1_clean"
coder2_clean_col = "coder2_clean"
final_clean_col = "final_code_clean"
clerically_coded_soc_df["clerical_agreement_soc"] = (
    clerically_coded_soc_df[coder1_clean_col]
    == clerically_coded_soc_df[coder2_clean_col]
)
clerically_coded_soc_df.loc[
    clerically_coded_soc_df["clerical_agreement_soc"], final_clean_col
] = clerically_coded_soc_df.loc[
    clerically_coded_soc_df["clerical_agreement_soc"], coder1_clean_col
]


# %%
# CLERICALLY CODED SOC - PREVIEW & PROFILE

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
# CLERICALLY CODED SIC - PREVIEW & PROFILE


print(f"\n{'='*70}")
print("CLERICALLY CODED SIC PREVIEW")
print(f"{'='*70}")
print(f"\n📋 Columns ({len(clerically_coded_sic_df.columns)} total):")
for i, col in enumerate(clerically_coded_sic_df.columns, 1):
    print(f"  {i:2}. {col}")

print(
    f"\n📊 Shape: {clerically_coded_sic_df.shape[0]} rows x "
    f"{clerically_coded_sic_df.shape[1]} columns"
)
# print(clerically_coded_sic_df.head(3).to_string())

print(f"\n{'='*70}")
print("SIC CLERICAL CODE VALUES")
print(f"{'='*70}")

clerically_coded_sic_df["clerical_codes"] = clerically_coded_sic_df[
    "clerical_codes"
].apply(set)
show_value_counts(
    clerically_coded_sic_df,
    "clerical_codes",
    "clerically_coded_sic_df.parquet :: SIC codes",
    top_n=15,
)


# %%
# CODE COLUMN VALUE SAMPLES
# Showing value distributions for configured SIC/SOC columns.
print(f"\n{'='*70}")
print("CLERICALLY CODED SOC CODE VALUES")
print(f"{'='*70}")

show_value_counts(
    clerically_coded_soc_df,
    coder1_clean_col,
    "clerically_coded_soc :: Coder 1",
    top_n=15,
)
show_value_counts(
    clerically_coded_soc_df,
    coder2_clean_col,
    "clerically_coded_soc :: Coder 2",
    top_n=15,
)

# %%
# MERGE CLERICAL + 2k INTO ONE DATAFRAME
clerically_coded_merged = clerically_coded_soc_df.merge(
    clerically_coded_sic_df,
    on="unique_id",
    how="left",
    suffixes=("", "_2k"),
    validate="one_to_one",
    indicator=True,
)
is_matched = clerically_coded_merged.pop("_merge").eq("both")
print(f"\nNumber of matching unique identifiers: {is_matched.sum()}")
print(f"Number of non-matching unique identifiers: {(~is_matched).sum()}")
print(
    f"Number of SIC clerically coded records excluded: "
    f"{clerically_coded_sic_df.shape[0] - is_matched.sum()}"
)

# Check the job title / description text agrees between the two files for each ID.
# (clerical column, 2k column) - confirm the 2k names against the column listing above.
text_col_pairs = {
    "job title": ("soc2020_job_title_main_job", "soc2020_job_title"),
    "job description": (
        "soc2020_job_description_main_job",
        "soc2020_job_description",
    ),
    "industry1": ("sic2007_employed_main_job", "sic2007_employee"),
    "industry2": ("sic2007_self_employed_main_job", "sic2007_self_employed"),
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
for text_type, (soc_col, sic_col) in text_col_pairs.items():
    if (
        soc_col not in clerically_coded_merged.columns
        or sic_col not in clerically_coded_merged.columns
    ):
        print(f"⚠️  Skipping {text_type} check - '{soc_col}' or '{sic_col}' not found")
        continue
    text_match = _normalise_text(clerically_coded_merged[soc_col]) == _normalise_text(
        clerically_coded_merged[sic_col]
    )
    n_mismatch = (is_matched & ~text_match).sum()
    print(
        f"{text_type.capitalize()} mismatches: {n_mismatch} of {is_matched.sum()} matched IDs"
    )
    if n_mismatch:
        print(
            clerically_coded_merged.loc[
                is_matched & ~text_match,
                ["unique_id", soc_col, sic_col],
            ]
            .head(5)
            .to_string(index=False)
        )


# %%
# INTER-RATER RELIABILITY: CODER AGREEMENT ANALYSIS
# Compute inter-rater reliability metrics (Cohen's kappa, % agreement),
# at each SOC digit level (1/2/3/4-digit), treating "uncodable" as a
# genuine category rather than missing data.

soc_digit_levels = sorted(
    {digits for digits, _label in SOC_CODABILITY_LEVELS if digits >= 0}
)
sic_digit_levels = sorted(
    {digits for digits, _label in SIC_CODABILITY_LEVELS if digits >= 0}
)

print(f"\n{'='*70}")
print("INTER-RATER RELIABILITY: Coder1 vs Coder2 (by SOC digit level)")
print(f"{'='*70}")
print(f"Total records: {len(clerically_coded_merged)}")

digit_level_summary = []
coder1_full = pd.Series(dtype=object)
coder2_full = pd.Series(dtype=object)

for soc_digit_count in soc_digit_levels:
    coder1_labels = clerically_coded_merged[coder1_clean_col].apply(
        lambda x, n=soc_digit_count: next(
            iter(get_clean_n_digit_codes(x, n=n, code_type="soc")[0]), "uncodable"
        )
    )
    coder2_labels = clerically_coded_merged[coder2_clean_col].apply(
        lambda x, n=soc_digit_count: next(
            iter(get_clean_n_digit_codes(x, n=n, code_type="soc")[0]), "uncodable"
        )
    )

    n_agree = (coder1_labels == coder2_labels).sum()
    pct_agree = 100 * n_agree / len(clerically_coded_merged)
    kappa = cohen_kappa_score(coder1_labels, coder2_labels)

    digit_level_summary.append(
        {
            "digits": soc_digit_count,
            "n_records": len(clerically_coded_merged),
            "n_agree": int(n_agree),
            "pct_agree": round(pct_agree, 1),
            "cohens_kappa": round(kappa, 4),
        }
    )

    if soc_digit_count == max(soc_digit_levels):
        # keep the full 4-digit labels around for the "Agree" cross-check below
        coder1_full = coder1_labels
        coder2_full = coder2_labels

digit_level_df = pd.DataFrame(digit_level_summary).sort_values(
    "digits", ascending=False
)
print("\nAgreement by SOC digit level (uncodable treated as its own category):")
print(digit_level_df.to_string(index=False))

# Cross-check against the clerical team's own row-level "Agree" flag, if present.
if "Agree" in clerically_coded_merged.columns:
    our_agree = coder1_full == coder2_full
    their_agree = (
        clerically_coded_merged["Agree"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )
    mismatches = (our_agree != their_agree).sum()
    print(
        f"\nCross-check vs sheet's own 'Agree' column: {mismatches} of "
        f"{len(clerically_coded_merged)} rows disagree with our recomputed agreement "
        "(investigate before trusting the figures above if this is large)."
    )

# %%
# SIC / SOC CODABILITY RELATIONSHIP
# Computed on the merged dataframe; rows with no 2k match stay NaN.
clerically_coded_merged["clerical_sic_codability_level"] = clerically_coded_merged[
    "clerical_codes"
].apply(lambda codes: get_codability_level(codes, code_type="SIC"))
clerically_coded_merged["clerical_codes_soc"] = clerically_coded_merged.apply(
    lambda row: row[coder1_clean_col].union(row[coder2_clean_col]),
    axis=1,
)
clerically_coded_merged["clerical_soc_codability_level"] = clerically_coded_merged[
    "clerical_codes_soc"
].apply(lambda codes: get_codability_level(codes, code_type="SOC"))

print(f"\n{'='*70}")
print("SIC vs SOC CODABILITY")
print(f"{'='*70}")

sic_level_order = [label for _digits, label in SIC_CODABILITY_LEVELS]
soc_level_order = [label for _digits, label in SOC_CODABILITY_LEVELS]
sic_cat = pd.Categorical(
    clerically_coded_merged["clerical_sic_codability_level"],
    categories=sic_level_order,
    ordered=True,
)
soc_cat = pd.Categorical(
    clerically_coded_merged["clerical_soc_codability_level"],
    categories=soc_level_order,
    ordered=True,
)
codability_crosstab = pd.crosstab(sic_cat, soc_cat, dropna=False)
codability_crosstab.index.name = "SIC codability"
codability_crosstab.columns.name = "SOC codability (dual-coding proxy)"
print("\nCross-tab of SIC codability vs SOC codability:")
print(codability_crosstab.to_string())

sic_uncodable = clerically_coded_merged["clerical_sic_codability_level"] == "Uncodable"
soc_uncodable = clerically_coded_merged["clerical_soc_codability_level"] == "Uncodable"
print(
    f"\nSIC uncodable: {sic_uncodable.mean():.1%} ("
    f"{sic_uncodable.sum()} of {len(clerically_coded_merged)})"
)
print(
    f"SOC uncodable (dual-coding proxy): {soc_uncodable.mean():.1%} "
    f"({soc_uncodable.sum()} of {len(clerically_coded_merged)})"
)
print(
    f"Both uncodable: {(sic_uncodable & soc_uncodable).mean():.1%} "
    f"({(sic_uncodable & soc_uncodable).sum()} of {len(clerically_coded_merged)})"
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


clerically_coded_merged["soc_major_group_digit"] = clerically_coded_merged[
    final_clean_col
].apply(
    lambda code, n=1: next(
        iter(get_clean_n_digit_codes(code, n=1, code_type="soc")[0]), "uncodable"
    )
)

soc_major_group_titles = {
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
clerically_coded_merged["soc_major_group"] = (
    clerically_coded_merged["soc_major_group_digit"]
    .map(soc_major_group_titles)
    .fillna("Uncodable")
)

example_cols = [
    "unique_id",
    "soc2020_job_title_main_job",
    "soc2020_job_description_main_job",
    coder1_clean_col,
    coder2_clean_col,
    final_clean_col,
]


def disagreement_by(
    merged: pd.DataFrame,
    group_col: str,
    title: str,
    what: str,
    agreement_col: str = "clerical_agreement_soc",
) -> None:
    """Disagreement rate per group, chi-square test and worst-group examples."""
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    stats = (
        merged.groupby(group_col)[agreement_col]
        .agg(n="size", n_disagree=lambda x: (~x).sum())
        .assign(disagree_rate=lambda d: (d["n_disagree"] / d["n"]).round(3))
        .sort_values("disagree_rate", ascending=False)
    )
    stats["low_sample_lt_15"] = stats["n"] < SECTION_MIN_N
    print(stats.to_string())

    ct = pd.crosstab(merged[group_col], ~merged[agreement_col])
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
            merged[(merged[group_col] == worst) & ~merged[agreement_col]][example_cols]
            .head(5)
            .to_string(index=False)
        )


disagreement_by(
    clerically_coded_merged,
    "sic_section",
    "DISAGREEMENT BY SIC SECTION (INDUSTRY)",
    "industry",
)
disagreement_by(
    clerically_coded_merged,
    "soc_major_group",
    "DISAGREEMENT BY OCCUPATION (SOC MAJOR GROUP)",
    "occupation",
)

# %%
# SURVEY ASSIST SHARED HELPERS (used by both the SOC and SIC comparisons)

SA_CODABILITY_CONFIDENCE_THRESHOLD = 0.8


def blank_low_confidence_initial_code(
    df: pd.DataFrame,
    codes_col: str = "initial_code",
    likelihood_col: str = "initial_likelihood",
) -> pd.DataFrame:
    """Blank `codes_col` wherever its likelihood is below
    SA_CODABILITY_CONFIDENCE_THRESHOLD or missing (i.e. not unambiguously codable).

    The likelihood is read from `likelihood_col` when present (one-prompt
    pipeline), otherwise from the best `alt_codes_col` candidate likelihood.
    """
    df = df.copy()
    if likelihood_col not in df.columns:
        raise ValueError(
            f"Likelihood column '{likelihood_col}' not found in DataFrame."
        )
    #  check the likelihood is numering and no missingness
    if not pd.api.types.is_numeric_dtype(df[likelihood_col]):
        raise ValueError(f"Likelihood column '{likelihood_col}' must be numeric.")
    if df[likelihood_col].isna().any():
        raise ValueError(
            f"Likelihood column '{likelihood_col}' contains missing values."
        )

    has_initial_code = df[codes_col].fillna("").astype(str).str.strip().ne("")
    low_confidence = df[likelihood_col] < SA_CODABILITY_CONFIDENCE_THRESHOLD
    not_unambiguously_codable = has_initial_code & low_confidence
    print(
        f"Blanking {not_unambiguously_codable.sum()} of {len(df)} "
        f"initial_code value(s) with likelihood below "
        f"{SA_CODABILITY_CONFIDENCE_THRESHOLD} or missing (not unambiguously codable)."
    )
    df.loc[not_unambiguously_codable, codes_col] = ""
    return df


# Apply the confidence threshold once to SOC; SIC output is used as is
sa_coded_soc_df = blank_low_confidence_initial_code(sa_coded_soc_df)


def run_sa_comparison(
    sa: pd.DataFrame,
    clerical: pd.DataFrame,
    code_type: str,
    sort_distribution: bool = False,
) -> pd.DataFrame:
    """Compare prepared Survey Assist initial codes with clerical truth by digit level."""
    title = f"SURVEY ASSIST {code_type} PERFORMANCE COMPARISON"
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")

    # Run performance evaluation with INITIAL_CODE only
    top = SIC_EXPECTED_CODE_LENGTH if code_type == "SIC" else SOC_EXPECTED_CODE_LENGTH
    digit_levels = sic_digit_levels if code_type == "SIC" else soc_digit_levels
    clerical_col = "clerical_codes" if code_type == "SIC" else "clerical_codes_soc"
    alt_codes_col = "alt_sic_candidates" if code_type == "SIC" else "alt_soc_candidates"
    high_level_name = "SOC major group" if code_type == "SOC" else "SIC section"

    summary = []
    full = pd.DataFrame()
    high_level = pd.DataFrame()

    for n in sorted(digit_levels, reverse=True):
        truth = prep_clerical_codes(
            clerical,
            clerical_col=clerical_col,
            code_type=code_type,
            digits=n,
            out_col="clerical_codes",
        )
        model = prep_model_codes(
            sa,
            alt_codes_col=alt_codes_col,
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
        if n == min(digit_levels):
            high_level = combined

    print(f"\nSurvey Assist vs clerical truth, by {code_type} digit level:")
    print(pd.DataFrame(summary).to_string(index=False))

    # Distribution comparison
    dist = (
        pd.concat(
            [
                high_level["clerical_codes"]
                .map(lambda x: next(iter(x), "Uncodable"))
                .value_counts(normalize=True)
                .rename("Clerical truth"),
                high_level["model_codes"]
                .map(lambda x: next(iter(x), "Uncodable"))
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
    print(f"\n{high_level_name} distribution, clerical truth vs Survey Assist (%):")
    print(dist.to_string())

    return full


# %%
# SURVEY ASSIST PERFORMANCE COMPARISON - SOC

full_soc_comparison_df = run_sa_comparison(
    sa=sa_coded_soc_df,
    clerical=clerically_coded_merged,
    code_type="SOC",
)

# %%
# SURVEY ASSIST PERFORMANCE COMPARISON - SIC

full_sic_comparison_df = run_sa_comparison(
    sa=sa_coded_sic_df,
    clerical=clerically_coded_merged,
    code_type="SIC",
    sort_distribution=True,
)

print("\n✓ Analysis complete!")

# %%
# CROSS-MODEL ERROR CORRELATION ANALYSIS
# Question: when Survey Assist is wrong in SOC, is it more likely wrong in SIC?

print(f"\n{'='*70}")
print("CROSS-MODEL ERROR CORRELATION ANALYSIS")
print(f"{'='*70}")


def score_records(model_codes: set, clerical_codes: set) -> str:
    """Per-record match."""
    if model_codes == clerical_codes:
        return "exact_match"
    if model_codes & clerical_codes:
        return "partial_match"
    return "mismatch"


# SOC: adjudicated consensus code vs Survey Assist (low-confidence codes blanked)
full_soc_comparison_df["soc_scores"] = full_soc_comparison_df.apply(
    lambda row: score_records(row["model_codes"], row["clerical_codes"]), axis=1
)


# SIC: 2k clerical candidate codes vs Survey Assist (used as is, no threshold)
full_sic_comparison_df["sic_scores"] = full_sic_comparison_df.apply(
    lambda row: score_records(row["model_codes"], row["clerical_codes"]), axis=1
)

cross_analysis_df = full_soc_comparison_df[["unique_id", "soc_scores"]].merge(
    full_sic_comparison_df[["unique_id", "sic_scores"]]
)

print(
    f"\nCombined dataset: {len(cross_analysis_df)} records with both SOC and SIC "
    f"results (SOC {SOC_EXPECTED_CODE_LENGTH}-digit, SIC {SIC_EXPECTED_CODE_LENGTH}-digit)"
)


def report_cross_errors(  # pylint: disable=too-many-locals
    df: pd.DataFrame, label: str
) -> None:
    """Contingency table, conditional error rates, effect sizes and tests."""
    print(f"\n{'-'*70}")
    print(f"{label} (n={len(df)})")
    print(f"{'-'*70}")

    score_levels = ["exact_match", "partial_match", "mismatch"]
    table3 = pd.crosstab(df["soc_scores"], df["sic_scores"]).reindex(
        index=score_levels, columns=score_levels, fill_value=0
    )
    print("\nContingency table:")
    print(table3.assign(All=table3.sum(axis=1)).to_string())
    print("\nRow percentages (given SOC result, % of SIC results):")
    print((table3.div(table3.sum(axis=1), axis=0) * 100).round(1).to_string())

    non_exact_levels = ["partial_match", "mismatch"]
    both_correct = table3.loc["exact_match", "exact_match"]
    both_wrong = table3.loc[non_exact_levels, non_exact_levels].to_numpy().sum()
    soc_only_wrong = table3.loc[non_exact_levels, "exact_match"].sum()
    sic_only_wrong = table3.loc["exact_match", non_exact_levels].sum()

    n = len(df)
    print("\nError pattern distribution:")

    for name, count in [
        ("Both exact match", both_correct),
        ("Both wrong", both_wrong),
        ("SOC only wrong", soc_only_wrong),
        ("SIC only wrong", sic_only_wrong),
    ]:

        print(f"  {name + ':':<16}{count:5d} ({100 * count / n:5.1f}%)")

    n_soc_wrong = both_wrong + soc_only_wrong
    n_soc_correct = sic_only_wrong + both_correct

    if n_soc_wrong == 0 or n_soc_correct == 0:

        print("\nSOC is all correct or all wrong - cannot compare SIC error rates.")

        return

    p_sic_wrong_given_soc_wrong = both_wrong / n_soc_wrong
    p_sic_wrong_given_soc_correct = sic_only_wrong / n_soc_correct

    print("\nConditional error probabilities:")
    print(f"  P(SIC wrong | SOC wrong)   = {p_sic_wrong_given_soc_wrong:.1%}")
    print(f"  P(SIC wrong | SOC correct) = {p_sic_wrong_given_soc_correct:.1%}")
    sic_wrong_count = table3.loc[:, non_exact_levels].to_numpy().sum()
    print(f"  P(SIC wrong) overall       = {sic_wrong_count / n:.1%}")
    print("\nEffect size:")

    risk_difference = p_sic_wrong_given_soc_wrong - p_sic_wrong_given_soc_correct

    print(f"  Risk difference = {risk_difference:+.1%} points")

    soc_wrong = df["soc_scores"] != "exact_match"
    sic_wrong = df["sic_scores"] != "exact_match"
    if soc_wrong.nunique() > 1 and sic_wrong.nunique() > 1:
        phi = np.corrcoef(soc_wrong, sic_wrong)[0, 1]
        print(f"  Phi correlation (exact vs non-exact) = {phi:.3f}")

    if p_sic_wrong_given_soc_correct > 0:

        relative_risk = p_sic_wrong_given_soc_wrong / p_sic_wrong_given_soc_correct

        print(
            f"  Relative risk   = {relative_risk:.2f}x "
            f"(SIC is {abs(relative_risk - 1):.0%} "
            f"{'more' if relative_risk >= 1 else 'less'} likely to be wrong "
            "when SOC is wrong than when SOC is correct)"
        )

    chi2, chi2_p, dof, expected = chi2_contingency(table3)

    print("\nTests for independence:")
    print(f"  Chi-square: chi2={chi2:.2f}, dof={dof}, p={chi2_p:.4g}")

    if expected.min() < 5:  # noqa: PLR2004
        print(
            "  (Expected count < 5 in some cells - chi-square result may be unreliable.)"
        )

    if chi2_p >= SIGNIFICANCE_LEVEL:
        print(
            f"=> No significant association between SOC and SIC errors "
            f"(p >= {SIGNIFICANCE_LEVEL})."
        )

    else:
        print(
            f"=> SOC and SIC errors are significantly associated (p < "
            f"{SIGNIFICANCE_LEVEL})."
        )


report_cross_errors(cross_analysis_df, "ALL RECORDS")

# %%
