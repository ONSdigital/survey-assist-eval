"""Analyse quality of classification for free text responses subset of Blaise SAYT pilot survey."""

# pylint: disable=C0103,R0801,C0301

# %%
import os

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from survey_assist_utils.logging import get_logger

from survey_assist_eval.data_cleaning.code_standard import SIC_EXPECTED_CODE_LENGTH
from survey_assist_eval.data_cleaning.prep_data import (
    get_clean_n_digit_codes,
    prep_clerical_codes,
)
from survey_assist_eval.evaluation.metrics import (
    calc_simple_metrics,
)

# %%
# Load environment variables and set up logging
DATADROP_NUM = 1

load_dotenv()
bucket_name = os.getenv("PREPROD_DATA_BUCKET_NAME")
if not bucket_name:
    raise ValueError("PREPROD_DATA_BUCKET_NAME environment variable not set")

logger = get_logger("sayt_blaise_pilot_sic_classification")

input_data_xlsx = (
    f"gs://{bucket_name}/2026-08-tlfs-sayt-free-text/TLFS_drop_1_coded_data_for_SA.xlsx",
    f"gs://{bucket_name}/2026-08-tlfs-sayt-free-text/???.xlsx",
)[DATADROP_NUM - 1]

output_dir = "data/plots/sayt_analysis"
os.makedirs(output_dir, exist_ok=True)

logger.info(
    "Analysing SIC classification from SAYT Blaise pilot survey",
    input_data=input_data_xlsx,
    output_dir=output_dir,
)


# %%
# Load the dataset (coded by three methods: clerical, CIMS, and SurveyAssist)
df_sayt_coded = pd.read_excel(input_data_xlsx, sheet_name="all_data", dtype=str)
df_sayt_coded["unique_id"] = (
    "tlfs_"
    + df_sayt_coded["case_person_id"].astype(str)
    + "_"
    + (df_sayt_coded.index + 2).astype(str)
)

clerical_msk = ~df_sayt_coded["Clerical_code_mj"].isna()
cims_msk = ~df_sayt_coded["sic2007_main_job_five_digit"].isin(["-9", -9, "-8", -8])
sa_msk = (
    ~df_sayt_coded["initial_code"].isna() | ~df_sayt_coded["alt_sic_candidates"].isna()
)
logger.info(
    "Number of records by coding method.",
    clerical=str(clerical_msk.sum()),
    cims=str(cims_msk.sum()),
    sa=str(sa_msk.sum()),
)

not_overlapping = df_sayt_coded[
    (clerical_msk | cims_msk | sa_msk) & ~(clerical_msk & cims_msk & sa_msk)
]
logger.info(
    "Number of manually coded records not overlapping between coding methods.",
    count=str(not_overlapping.shape[0]),
)
print(
    df_sayt_coded[
        (clerical_msk | cims_msk | sa_msk) & ~(clerical_msk & cims_msk & sa_msk)
    ]
)
# records not covered by all three coding methods are missing sa codes
# we supressed some input records, so take a subset based on sa codes present

df_sub = df_sayt_coded[sa_msk & clerical_msk & cims_msk].copy()

# %%
# Standardise codes
clerical = prep_clerical_codes(df_sub, clerical_col="Clerical_code_mj")
msk = clerical.clerical_codes_invalid.map(len) > 0

fix_pairs = [
    ("46702", "46720"),
    ("72202", "72200"),
    ("477x", "477xx"),
    ("16320", "16230"),
]
logger.info(
    "Applying manual fix to records with invalid clerical codes before fixing.",
    count=str(msk.sum()),
    suggested_fix=str(fix_pairs),
)
df_sub["clerical_fix"] = df_sub["Clerical_code_mj"]
for old, new in fix_pairs:
    df_sub.loc[msk, "clerical_fix"] = df_sub.loc[msk, "clerical_fix"].str.replace(
        old, new
    )
df_sub["clerical_codes"] = prep_clerical_codes(df_sub, clerical_col="clerical_fix")[
    "clerical_codes"
]


df_sub["cims_combine"] = "-9"
for num_dig, word in enumerate(["two", "three", "four", "five"], start=2):
    msk = ~df_sub[f"sic2007_main_job_{word}_digit"].isin(["-9", -9, "-1", -1, "-8", -8])
    df_sub.loc[msk, "cims_combine"] = df_sub.loc[
        msk, f"sic2007_main_job_{word}_digit"
    ] + "x" * (SIC_EXPECTED_CODE_LENGTH - num_dig)
df_sub["cims_codes"] = prep_clerical_codes(df_sub, clerical_col="cims_combine")[
    "clerical_codes"
]

df_sub["sa_combine"] = df_sub["initial_code"].fillna("") + df_sub[
    "alt_sic_candidates"
].fillna("")
df_sub["sa_codes"] = prep_clerical_codes(df_sub, clerical_col="sa_combine")[
    "clerical_codes"
]

# %%
# Calculate performance at different number of digits
DIGIT_LEVELS = [0, 2, 3, 4, 5]
for col in ["clerical_codes", "cims_codes", "sa_codes"]:
    for digits in DIGIT_LEVELS:
        df_sub[f"{col}_to_{digits}digits"] = df_sub[col].apply(
            lambda x, n=digits: get_clean_n_digit_codes(x, n=n, code_type="SIC")[0]
        )


eval_metrics = {}
for digits in DIGIT_LEVELS:
    print(f"Processing codes to {digits} digits..")

    eval_metrics[(digits, "SurveyAssist_cc")] = calc_simple_metrics(
        df_sub,
        truth_col=f"clerical_codes_to_{digits}digits",
        initial_model_col=f"sa_codes_to_{digits}digits",
        final_model_col=None,
    )
    eval_metrics[(digits, "CIMS_cc")] = calc_simple_metrics(
        df_sub,
        truth_col=f"clerical_codes_to_{digits}digits",
        initial_model_col=f"cims_codes_to_{digits}digits",
        final_model_col=None,
    )
    eval_metrics[(digits, "Clerical_cc")] = calc_simple_metrics(
        df_sub,
        truth_col=f"clerical_codes_to_{digits}digits",
        initial_model_col=f"clerical_codes_to_{digits}digits",
        final_model_col=None,
    )

plot_df = pd.DataFrame(
    [
        {
            "digits": str(k[0]) if k[0] > 0 else "S",
            "method": k[1].split("_")[0],
            "Codability": v.codability_metrics.initial_codable_prop,
            "F1": v.ambiguity_metrics.f1,
            "Precision": v.ambiguity_metrics.precision,
            "Recall": v.ambiguity_metrics.recall,
            "Accuracy": v.ambiguity_metrics.accuracy,
            "Confusion Matrix": (
                f"TP={v.ambiguity_metrics.TP}, FP={v.ambiguity_metrics.FP}, FN={v.ambiguity_metrics.FN}, TN={v.ambiguity_metrics.TN}"
            ),
            "OO Accuracy": (
                v.initial_accuracy_metrics.accuracy_oo_unambiguous,
                v.initial_accuracy_metrics.matches_oo,
                v.ambiguity_metrics.TN,
            ),
            "OM Accuracy": (
                v.initial_accuracy_metrics.accuracy_om_unambiguous,
                v.initial_accuracy_metrics.matches_om,
                v.ambiguity_metrics.FP + v.ambiguity_metrics.TN,
            ),
            "MO Accuracy": (
                v.initial_accuracy_metrics.accuracy_mo_unambiguous,
                v.initial_accuracy_metrics.matches_mo,
                v.ambiguity_metrics.FN + v.ambiguity_metrics.TN,
            ),
            "MM Accuracy": (
                v.initial_accuracy_metrics.accuracy_mm_total,
                v.initial_accuracy_metrics.matches_mm,
                v.initial_accuracy_metrics.total_records,
            ),
        }
        for k, v in eval_metrics.items()
    ]
)
# remove dummy clerical_cc performance
msk = plot_df["method"] == "Clerical"
cols = set(plot_df.columns).difference(["digits", "method", "Codability"])
plot_df.loc[msk, list(cols)] = None


# %%
# Melt results dataframe and plot ambiguity decision metrics
plot_df_f1 = plot_df.melt(
    id_vars=["digits", "method"],
    value_vars=["Codability", "Precision", "Recall", "F1", "Accuracy"],
    var_name="metrics",
    value_name="value",
)

fig = px.line(
    plot_df_f1,
    x="digits",
    y="value",
    color="method",
    facet_col="metrics",
    title="Ambiguity Decision Metrics by Number of Digits and Method",
    markers=True,
    template="simple_white",
)
# drop first part of facet annotation
for i in fig.layout.annotations:
    i.text = i.text.split("=")[-1].capitalize()
# display y axes as percentages and remove axis title
fig.update_yaxes(tickformat=".0%", title_text="", showgrid=True, gridcolor="lightgrey")

# add text to footnote
fig.update_layout(margin={"b": 130})
fig.add_annotation(
    text=(
        """
Codability: Percentage of records identified as unambiguous by either the model or clerical coders.<br>
Precision: Among cases flagged as ambiguous by the model, the percentage that are truly ambiguous.<br>
Recall: Among all truly ambiguous cases, the percentage correctly identified by the model.<br>
F1: The harmonic mean of precision and recall.<br>
Accuracy: Overall percentage of correct codability/ambiguity decisions.
"""
    ),
    align="left",
    xref="paper",
    yref="paper",
    x=-0.08,
    y=-0.45,
    showarrow=False,
    font={"size": 10},
)
fig.update_layout(height=500, width=1000)
fig.write_html(f"{output_dir}/sayt_sic_ambiguity_decision_metrics.html")
fig.show()


# %%
# Melt results dataframe and plot matching accuracy metrics
plot_df_accu = plot_df.melt(
    id_vars=["digits", "method"],
    value_vars=["OO Accuracy", "OM Accuracy", "MO Accuracy", "MM Accuracy"],
    var_name="metrics",
    value_name="value_tuple",
)
# drop NAs in Clerical itself
plot_df_accu = plot_df_accu[plot_df_accu["method"] != "Clerical"]

# unwrap tuple into three columns
plot_df_accu[["accu_value", "matches", "total"]] = pd.DataFrame(
    plot_df_accu["value_tuple"].tolist(), index=plot_df_accu.index
)


# %%
# Treat uncodable fairly (make sure we are not penalizing CIMS for records that are uncodable)
# SurveyAssist always returns some candidates while clerical and CIMS may mark records as uncodable.
# The overall trend in accuracy comparison doesn't change when these are excluded or treated differently,
# but to give a fair comparison, we will consider CIMS uncodable records as extra matches for the MM Accuracy.

cims_cc_uncodable_match = sum(
    (df_sub.cims_codes == set()) & (df_sub.clerical_codes == set())
)

msk = (plot_df_accu["metrics"] == "MM Accuracy") & (plot_df_accu["method"] == "CIMS")
plot_df_accu.loc[msk, "matches"] += cims_cc_uncodable_match
plot_df_accu.loc[msk, "accu_value"] = (
    plot_df_accu.loc[msk, "matches"] / plot_df_accu.loc[msk, "total"]
)


# %%
fig = px.line(
    plot_df_accu,
    x="digits",
    y="accu_value",
    color="method",
    facet_col="metrics",
    title="Classification Accuracy Metrics by Number of Digits and Method",
    markers=True,
    template="simple_white",
    hover_data={"matches": True, "total": True, "accu_value": ":.2%"},
)
# drop first part of facet annotation
for i in fig.layout.annotations:
    i.text = i.text.split("=")[1]
# display y axes as percentages and remove axis title
fig.update_yaxes(tickformat=".0%", title_text="", showgrid=True, gridcolor="lightgrey")

# add text to footnote
fig.update_layout(margin={"b": 125})
fig.add_annotation(
    text=(
        """
OO: One-to-One match on a subset where the clerical label as well as the model's label are not ambiguous.<br>
OM: One-to-Many match on a subset where the clerical label is not ambiguous. (Is the clerical label in the model's shortlist?)<br>
MO: Many-to-One match on a subset where the model is not ambiguous. (Is the model's label in the clerical label shortlist?)<br>
MM: Many-to-Many match on the full set. (Is there any overlap between the clerical label's and model's shortlists?)
"""
    ),
    align="left",
    xref="paper",
    yref="paper",
    x=-0.08,
    y=-0.42,
    showarrow=False,
    font={"size": 10},
)
fig.update_layout(height=500, width=770)

fig.write_html(f"{output_dir}/sayt_sic_matching_accuracy_metrics.html")
fig.show()

# %%
