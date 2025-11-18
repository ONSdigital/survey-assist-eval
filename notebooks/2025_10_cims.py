"""WIP evaluation of CIMS SIC code predictions vs semantic search and SurveyAssist initial model."""

# pylint: disable=C0301,C0103,R0801

# %%
import logging
import os

import dotenv
import pandas as pd
import plotly.express as px

from survey_assist_utils.data_cleaning.prep_data import (
    prep_clerical_codes,
    prep_model_codes,
)
from survey_assist_utils.data_cleaning.sic_codes import get_clean_n_digit_one_code
from survey_assist_utils.evaluation.metrics import (
    calc_simple_metrics,
)

# %%
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

bucket_prefix = dotenv.get_key(".env", "BUCKET_PREFIX")
if not bucket_prefix:
    raise ValueError("BUCKET_PREFIX not found in .env file. Please set it.")

output_folder = "data/temp"  # set to None if no output saving is needed

if output_folder:
    os.makedirs(output_folder, exist_ok=True)


# %%
# load clerical data
clerical_col = "sic_ind_occ"
clerical_file = f"{bucket_prefix}original_datasets/TLFS_evaluation_data/TLFS_evaluation_data_IT5.csv"
clerical_4plus_file = (
    f"{bucket_prefix}original_datasets/TLFS_evaluation_data/Codes_for_4_plus_IT5.csv"
)

cc_raw_df = pd.read_csv(clerical_file)
cc_4plus_df = pd.read_csv(clerical_4plus_file)

msk = cc_4plus_df[clerical_col].notna() & ~(
    cc_4plus_df[clerical_col].isin(["", -9, "-9"])
)
cc_4plus_df = cc_4plus_df[msk].reset_index(drop=True)

# %%
# read cims data
cims_file = f"{bucket_prefix}CIMS/raw_data_stage_2_2025_09_16_16_15.xlsx"
cims_df = pd.read_excel(cims_file, sheet_name="Sheet1").rename(
    columns={"tlfs_id": "unique_id"}
)
cims_df = cims_df.drop_duplicates(subset=["unique_id"]).reset_index(drop=True)
logger.info("CIMS data shape: %s", cims_df.shape)


# %%
# prepare semantic search data
sem_file = f"{bucket_prefix}two_prompt_pipeline/2025_10_tlfs_it5_gemini25/STG1.parquet"
sem_df = pd.read_parquet(sem_file)
logger.info("Semantic search data shape: %s", sem_df.shape)


# %%
# read classifai data
classifai_file = f"{bucket_prefix}CIMS/2025_11_tlfs_it5_classifai.parquet"
class_df = pd.read_parquet(classifai_file)
# class_df = class_df[class_df['classifai_results'].map( lambda x: x['code']!='*')]
for i in class_df.index:
    if class_df.loc[i, "classifai_results"]["code"] == "*":
        class_df.loc[i, "classifai_results"]["distance"] = 10
logger.info("Classifai data shape: %s", class_df.shape)


# %%
# load initial survey assitst predictions for reference
sa_model_file = "/Users/ivaspakulova/Documents/python/_surveyassist/sic-classification-utils/data/tlfs/STG2.parquet"
sa_model_df = pd.read_parquet(sa_model_file)
# as it is interim file, find the last nonempty initial_code
msk = sa_model_df["alt_sic_candidates"].notna() & ~(
    sa_model_df["alt_sic_candidates"].apply(len) == 0
)
sa_model_df = sa_model_df.loc[: sa_model_df[msk].index[-1]]
logger.info("SurveyAssist model data shape: %s", sa_model_df.shape)

# %%
# subset to common IDs (sa_df and cims are both subsets of semantic==cc)
ID_SUBSET = (
    set(sa_model_df["unique_id"].unique())
    .intersection(cims_df["unique_id"].unique())
    .intersection(class_df["unique_id"].unique())
)
logger.info("Common ID subset size: %s", len(ID_SUBSET))
cc_raw_df = cc_raw_df[cc_raw_df["unique_id"].isin(ID_SUBSET)].reset_index(drop=True)
cc_4plus_df = cc_4plus_df[cc_4plus_df["unique_id"].isin(ID_SUBSET)].reset_index(
    drop=True
)
cims_df = cims_df[cims_df["unique_id"].isin(ID_SUBSET)].reset_index(drop=True)
sem_df = sem_df[sem_df["unique_id"].isin(ID_SUBSET)].reset_index(drop=True)
sa_df = sa_model_df[sa_model_df["unique_id"].isin(ID_SUBSET)].reset_index(drop=True)
class_df = class_df[class_df["unique_id"].isin(ID_SUBSET)].reset_index(drop=True)

# %%
default_split = "residential_care"  #'residential_care' # other options: 'all', 'sic_section', 'residential_care'


def label_subsets(
    in_df: pd.DataFrame, subset_type: str = default_split, min_section_size: int = 1000
) -> pd.Series:
    """Label subsets based on unique_id prefixes or specific conditions."""
    if subset_type.lower() == "all":
        return pd.Series(["All"] * len(in_df), index=in_df.index)

    if subset_type.lower() == "sic_section":
        vc = in_df["sic_section"].value_counts()
        rare_sections = vc[vc < min_section_size].index.tolist()
        return in_df["sic_section"].replace(rare_sections, "Other")

    out_subsets = in_df["unique_id"].map(lambda x: x[:2])
    if subset_type.lower() == "residential_care":
        care_msk = (in_df["sic_ind1"] == "87300") | (in_df["sic_ind_occ1"] == "87300")
        out_subsets.loc[care_msk] = "RC"

    return out_subsets


# %%
# top candidate performance by threshold on distance
digits_set = [0, 2, 3, 4, 5]
info_cols = [
    "unique_id",
    "sic_ind_occ1",
    "sic2007_employee",
    "sic2007_self_employed",
    "soc2020_job_title",
    "soc2020_job_description",
]
top_match_metrics = {}
cc_codable = pd.DataFrame()
DISTANCE_SCALER = 1000  # to bring distances to some easy to display values

for DIGITS in digits_set:
    logger.info("--- Evaluating %d-digit match ---", DIGITS)

    # prep clerical coding (2nd iteration, ground truth):
    cc_df = prep_clerical_codes(
        cc_raw_df, cc_4plus_df, clerical_col=clerical_col, digits=DIGITS
    )
    cc_df["subset"] = label_subsets(cc_raw_df)

    # prep semantic search model :
    sem_df["top_distance"] = sem_df["semantic_search_results"].apply(
        lambda x: x[0]["distance"] * DISTANCE_SCALER if len(x) > 0 else None
    )
    sem_df["top_candidate"] = sem_df["semantic_search_results"].apply(
        lambda x, digits=DIGITS: (
            get_clean_n_digit_one_code(x[0]["code"], digits) if len(x) > 0 else None
        )
    )
    sem_df["top_description"] = sem_df["semantic_search_results"].apply(
        lambda x: x[0]["title"] if len(x) > 0 else None
    )
    # prep classifai data
    class_df["top_distance"] = class_df["classifai_results"].apply(
        lambda x: x["distance"] * DISTANCE_SCALER if x else None
    )
    class_df["top_candidate"] = class_df["classifai_results"].apply(
        lambda x, digits=DIGITS: (get_clean_n_digit_one_code(x["code"], digits))
    )
    class_df["top_description"] = class_df["classifai_results"].apply(
        lambda x: x["title"] if x else None
    )

    # prep CIMS data
    code_col = f"predictionClass_SIC07_stage_2_{max(2, DIGITS)}"
    conf_col = f"confidence_SIC07_stage_2_{max(2, DIGITS)}"
    msk = cims_df[code_col].isna()
    # stuck on deterministic codes
    cims_df.loc[msk, code_col] = cims_df.loc[msk, "predicted_SIC07_determ_and_stage_2"]
    cims_df.loc[msk, conf_col] = 1
    cims_df["top_candidate"] = cims_df[code_col].apply(
        lambda x, digits=DIGITS: get_clean_n_digit_one_code(
            str(x).replace(".0", ""), digits
        )
    )
    cims_df["top_distance"] = (1 - cims_df[conf_col]) * DISTANCE_SCALER

    combined_df = (
        cims_df[["unique_id", code_col, conf_col, "top_candidate", "top_distance"]]
        .merge(cc_df, on="unique_id", how="outer")
        .merge(
            sem_df[
                [
                    *info_cols,
                    "semantic_search_results",
                    "top_candidate",
                    "top_distance",
                    "top_description",
                ]
            ],
            on="unique_id",
            how="outer",
            suffixes=("_CIMS", ""),
        )
        .merge(
            class_df[
                [
                    "unique_id",
                    "classifai_results",
                    "top_candidate",
                    "top_distance",
                    "top_description",
                ]
            ],
            on="unique_id",
            how="outer",
            suffixes=("_semantic", "_classifai"),
        )
        .copy()
    )

    # cumulative eval of top candidate based on threshold
    for method in ["semantic", "CIMS", "classifai"]:
        for subset in combined_df["subset"].unique():
            eval_df = (
                combined_df[combined_df["subset"] == subset]
                .copy()
                .rename(
                    columns={
                        f"top_candidate_{method}": "top_candidate",
                        f"top_distance_{method}": "top_distance",
                        f"top_description_{method}": "top_description",
                    }
                )
                .sort_values(by="top_distance", ascending=True)
                .reset_index(drop=True)
            )
            eval_df["top_in_cc"] = eval_df.apply(
                lambda row: len(row["top_candidate"] & row["clerical_codes"]) > 0,
                axis=1,
            )
            eval_df["clerical_unambig"] = eval_df["clerical_codes"].map(
                lambda x: len(x) == 1
            )

            # subset to CC unambiguous cases
            unambig_df = (
                eval_df[eval_df["clerical_unambig"]].copy().reset_index(drop=True)
            )
            cc_codable = pd.concat(
                [
                    cc_codable,
                    pd.DataFrame(
                        {
                            "digits": [str(DIGITS) if DIGITS > 0 else "S"],
                            "method": [method],
                            "subset": [subset],
                            "codability": [len(unambig_df) / len(eval_df)],
                        }
                    ),
                ],
                ignore_index=True,
            )
            logger.info(
                "Total clerical codes unambiguous: %.2f%%",
                len(unambig_df) / len(eval_df) * 100,
            )

            # calculate proportion and accuracy at different thresholds on distance
            for df in [eval_df, unambig_df]:
                df["subset_total"] = range(1, len(df) + 1)
                df["match_count"] = df["top_in_cc"].cumsum()
                df["codability"] = df["subset_total"] / len(df)
                df["accuracy"] = df["match_count"] / df["subset_total"]
            for i in eval_df.index:
                # could be done by reverse cumsum but it is then fiddly with the deduplication
                thr = eval_df.loc[i, "top_distance"]
                thr_msk = eval_df["top_distance"] > thr
                eval_df.loc[i, "TP"] = (~eval_df.loc[thr_msk, "clerical_unambig"]).sum()
                if thr_msk.sum() > 0:
                    eval_df.loc[i, "precision"] = eval_df.loc[i, "TP"] / thr_msk.sum()
            if not eval_df["clerical_unambig"].all():
                eval_df["recall"] = eval_df["TP"] / (~eval_df["clerical_unambig"]).sum()

            eval_df = eval_df.drop_duplicates(
                subset=["top_distance"], keep="last"
            ).reset_index(drop=True)

            unambig_df = unambig_df.drop_duplicates(
                subset=["top_distance"], keep="last"
            )

            # store for plotting
            top_match_metrics[
                (str(DIGITS) if DIGITS > 0 else "S", method, "OO", subset)
            ] = unambig_df.copy()
            top_match_metrics[
                (str(DIGITS) if DIGITS > 0 else "S", method, "MO", subset)
            ] = eval_df.copy()


# %%
sa_df = pd.DataFrame()
simulate_lookup = (
    True  # set to True to simulate knowledge base lookup from semantic search results
)
LOOKUP_MATCH_THRESHOLD = 10

for DIGITS in digits_set:

    cc_df = prep_clerical_codes(
        cc_raw_df, cc_4plus_df, clerical_col=clerical_col, digits=DIGITS
    )
    cc_df["subset"] = label_subsets(cc_raw_df)

    sa_model_df["initial_code2"] = sa_model_df["initial_code"]
    if simulate_lookup:
        kb_msk = sa_model_df["semantic_search_results"].map(
            lambda x: (
                x[0]["distance"] < LOOKUP_MATCH_THRESHOLD / DISTANCE_SCALER
                if len(x) > 0
                else False
            )
        )
        sa_model_df.loc[kb_msk, "initial_code2"] = sa_model_df.loc[
            kb_msk, "semantic_search_results"
        ].map(lambda x: x[0]["code"])
    model_prompt2 = prep_model_codes(
        sa_model_df,
        digits=DIGITS,
        codes_col="initial_code2",
        out_col="sa_initial_codes",
        threshold=0,
    )
    combined_dataframe_m2 = model_prompt2.merge(cc_df, on="unique_id", how="inner")

    for subset in combined_dataframe_m2["subset"].unique():

        eval_metr = calc_simple_metrics(
            combined_dataframe_m2[combined_dataframe_m2["subset"] == subset]
        )
        sa_df = pd.concat(
            [
                sa_df,
                pd.DataFrame(
                    {
                        "digits": [str(DIGITS) if DIGITS > 0 else "S"] * 2,
                        "method": ["SurveyAssist"] * 2,
                        "match_type": ["OO", "MO"],
                        "subset": [subset] * 2,
                        "accuracy": [
                            eval_metr.initial_accuracy_metrics.accuracy_oo_unambiguous,
                            eval_metr.initial_accuracy_metrics.accuracy_mo_unambiguous,
                        ],
                        "codability": [
                            eval_metr.codability_metrics.initial_codable_prop
                        ]
                        * 2,
                        "precision": [eval_metr.ambiguity_metrics.precision] * 2,
                        "recall": [eval_metr.ambiguity_metrics.recall] * 2,
                    }
                ),
            ],
            ignore_index=True,
        ).reset_index(drop=True)


# %%
# prepare data for plotting codability vs accuracy curves
plot_df = pd.DataFrame(
    [
        {
            "digits": key[0],
            "method": key[1],
            "match_type": key[2],
            "subset": key[3],
            "codability": row["codability"],
            "distance_threshold": round(row.top_distance, 3),
            "accuracy": row["accuracy"],
            "match_count": row["match_count"],
            "subset_total": row["subset_total"],
            "precision": row.get("precision", None),
            "recall": row.get("recall", None),
        }
        for key, df in top_match_metrics.items()
        for _, row in df.iterrows()
    ]
)

for DIGITS in plot_df["digits"].unique():
    fig = px.line(
        plot_df[
            (plot_df["codability"] > 1 / 10) & (plot_df["digits"] == DIGITS)
        ],  # remove initial small sample variation
        x="codability",
        y="accuracy",
        color="method",
        facet_col="match_type",
        facet_row="subset",
        title=f"""Codability vs Accuracy of top candidates <br>(above parametrised threshold, <b>{
            str(DIGITS)+"-digits" if DIGITS!='S' else "Section level"}</b> match)""",
        template="simple_white",
        hover_data={
            "distance_threshold": True,
            "match_count": True,
            "subset_total": True,
        },
    )
    # Add vline to all subplots/facets
    subsets = plot_df["subset"].unique()
    for fac_row, subset in enumerate(subsets):
        x_val = cc_codable[
            (cc_codable["digits"] == str(DIGITS)) & (cc_codable["subset"] == subset)
        ]["codability"].values[0]
        fig.add_vline(
            x=x_val,
            line={"color": "navy", "width": 2},
            line_dash="dot",
            annotation_text="Clerical codability",
            annotation_position="bottom right",
            annotation_font_size=10,
            row=len(subsets) - fac_row,
        )

    # add dots for SA model by facets
    for fac_col, match_type in enumerate(["OO", "MO"]):
        for fac_row, subset in enumerate(subsets):
            msk = (
                (sa_df["digits"] == DIGITS)
                & (sa_df["match_type"] == match_type)
                & (sa_df["subset"] == subset)
            )
            fig.add_scatter(
                x=sa_df[msk]["codability"],
                y=sa_df[msk]["accuracy"],
                mode="markers",
                marker={"size": 10, "color": "navy", "symbol": "x"},
                name="SurveyAssist",
                col=fac_col + 1,
                row=len(subsets) - fac_row,
                showlegend=(fac_col == 0) & (fac_row == 0),
            )

    # display y axes as percentages and remove axis title
    fig.update_yaxes(tickformat=".0%", showgrid=True, gridcolor="lightgrey")
    fig.update_xaxes(
        tickformat=".0%"
    )  # , title_text="Codability (prop. above threshold)")

    # add text to footnote
    fig.update_layout(margin={"b": 100})
    fig.add_annotation(
        text=(
            """
    OO: One-to-One match on a subset where the true label as well as the model's label are not ambiguous.<br>
    MO: Many-to-One match on a subset where the model is not ambiguous. (Is the model's label in the true label shortlist?)<br>
    """
        ),
        align="left",
        xref="paper",
        yref="paper",
        x=-0.08,
        y=-0.5 / len(subsets),
        showarrow=False,
        font={"size": 10},
    )
    fig.update_layout(height=200 + 200 * len(subsets), width=770)
    fig.show()

    if output_folder:
        fig.to_html(
            f"{output_folder}/2025-10_semantic_top_candidate_accuracy_codability_{DIGITS}-digits.html"
        )

# %%

for DIGITS in plot_df["digits"].unique():
    fig = px.line(
        plot_df[
            (plot_df["match_type"] == "MO") & (plot_df["digits"] == DIGITS)
        ],  # remove initial small sample variation
        x="recall",
        y="precision",
        color="method",
        facet_row="subset",
        title=f"""Precision vs Recall using parametrised threshold (<b>{
            str(DIGITS)+"-digits" if DIGITS!='S' else "Section level"}</b> match)""",
        template="simple_white",
        hover_data={
            "distance_threshold": True,
            "match_count": True,
            "subset_total": True,
        },
    )

    subsets = plot_df["subset"].unique()
    # add dots for SA model by facets
    for fac_row, subset in enumerate(subsets):
        msk = (
            (sa_df["digits"] == DIGITS)
            & (sa_df["match_type"] == "MO")
            & (sa_df["subset"] == subset)
        )
        fig.add_scatter(
            x=sa_df[msk]["recall"],
            y=sa_df[msk]["precision"],
            mode="markers",
            marker={"size": 10, "color": "navy", "symbol": "x"},
            name="SurveyAssist",
            col=1,
            row=len(subsets) - fac_row,
            showlegend=(fac_row == 0),
        )

    # display y axes as percentages and remove axis title
    fig.update_yaxes(tickformat=".0%", showgrid=True, gridcolor="lightgrey")
    fig.update_xaxes(
        tickformat=".0%"
    )  # , title_text="Codability (prop. above threshold)")

    # add text to footnote
    fig.update_layout(margin={"b": 100})
    fig.add_annotation(
        text=(
            """
        Precision: Among cases flagged as ambiguous by the model, the percentage that are truly ambiguous.<br>
        Recall: Among all truly ambiguous cases, the percentage correctly identified by the model.<br>
        """
        ),
        align="left",
        xref="paper",
        yref="paper",
        x=-0.08,
        y=-0.5 / len(subsets),
        showarrow=False,
        font={"size": 10},
    )
    fig.update_layout(height=200 + 200 * len(subsets), width=620)
    fig.show()

    if output_folder:
        fig.to_html(
            f"{output_folder}/2025-10_semantic_top_candidate_precision_recall_{DIGITS}-digits.html"
        )


# %%
field_len = {}
for field in [
    "sic2007_employee",
    "sic2007_self_employed",
    "soc2020_job_title",
    "soc2020_job_description",
]:
    lens = cc_raw_df[field].map(
        lambda x: None if (x in ["-9", -9, "X"]) else len(str(x))
    )
    # calculate stats and min/max
    field_len[field] = {
        "min": int(lens.min()),
        "5%": int(lens.quantile(0.05)),
        "25%": int(lens.quantile(0.25)),
        "mean": int(lens.mean()),
        "median": int(lens.median()),
        "75%": int(lens.quantile(0.75)),
        "95%": int(lens.quantile(0.95)),
        "max": int(lens.max()),
    }
field_len_df = pd.DataFrame(field_len).T
logger.info("Field length statistics:\n%s", field_len_df)

# %%
msk = (
    (combined_df["subset"] == "EV")
    & (combined_df["clerical_codes"] != combined_df["top_candidate_classifai"])
    & (combined_df["top_distance_classifai"] < LOOKUP_MATCH_THRESHOLD)
    & (combined_df["clerical_codes"].map(len) == 1)
)
combined_df[msk][
    [
        "unique_id",
        "soc2020_job_title",
        "soc2020_job_description",
        "sic2007_employee",
        "sic2007_self_employed",
        "clerical_codes",
        "top_candidate_classifai",
        "top_distance_classifai",
        "top_description_classifai",
        "top_candidate_semantic",
        "top_distance_semantic",
        "top_description_semantic",
    ]
].sort_values(by="top_distance_classifai").drop_duplicates(
    subset=["sic2007_employee", "sic2007_self_employed"]
).head(
    20
)


# %% {markdown}
# Explanation for classifai differencies and problems
# - there are entries like 'CHILDCARE', 'CHARITY', 'RETAIL' in knowledge base that are uncodable/wrong
# - not using job title and description fields for matching, only SIC titles (e.g. dentist hyegienist vs dentistry)


# %%
msk = (
    (combined_df["subset"] == "EV")
    & (combined_df["clerical_codes"] != combined_df["top_candidate_classifai"])
    & (combined_df["clerical_codes"].map(len) == 1)
    # & (combined_df['clerical_codes']==combined_df['top_candidate_CIMS'])
)
combined_df[msk][
    [
        "unique_id",  #'soc2020_job_title', 'soc2020_job_description',
        "sic2007_employee",
        "sic2007_self_employed",
        "clerical_codes",
        "top_candidate_classifai",
        "top_distance_classifai",
        "top_description_classifai",
        "top_candidate_CIMS",
        "top_distance_CIMS",
    ]
].sort_values(by="top_distance_classifai").drop_duplicates(
    subset=["sic2007_employee", "sic2007_self_employed"]
).head(
    20
)

# %%
