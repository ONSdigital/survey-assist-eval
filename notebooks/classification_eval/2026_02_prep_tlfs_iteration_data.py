"""Notebook to prepare data for TLFS iteration evaluation.

It loads clerical coding data, SurveyAssist outputs, and CIMS outputs from the
configured evaluation bucket.
The bucket prefix is read from the .env file, where it should be stored as
EVALUATION_BUCKET, i.e.:
EVALUATION_BUCKET = "gs://<bucket-name>/"

Disabled check for too long lines (f strings) and variables names (uppercase for constants)
"""

# pylint: disable=C0301,C0103

# %%
import dotenv
import pandas as pd
from survey_assist_utils import get_logger

from survey_assist_eval.data_cleaning.prep_data import (
    prep_clerical_codes,
    prep_model_codes,
)
from survey_assist_eval.data_cleaning.sic_codes import (
    get_clean_n_digit_codes,
    parse_numerical_code,
)

# %%
logger = get_logger(__name__)

bucket_prefix = dotenv.get_key(".env", "EVALUATION_BUCKET")
if not bucket_prefix:
    raise ValueError("EVALUTION_BUCKET not found in .env file. Please set it.")

work_folder = f"{bucket_prefix}evaluation-pipeline/two_prompt_pipeline/2026_03_tlfs_it11_gemini25_europe_west9/"

cc_data_folder = (
    f"{bucket_prefix}evaluation-pipeline/original_datasets/TLFS_evaluation_data/"
)

cims_folder = f"{bucket_prefix}evaluation-pipeline/CIMS/"

# %%
# Load clerical source files used to build the comparison dataset.
clerical_file = f"{cc_data_folder}TLFS_evaluation_data_IT11.csv"
clerical_4plus_file = f"{cc_data_folder}Codes_for_4_plus_IT11_v3.csv"
# clerical_errors_file = f"{cc_data_folder}tlfs_it9_invalid_code_correction_to_47110.csv"
cc_df = pd.read_csv(clerical_file)
cc_4plus_df = pd.read_csv(clerical_4plus_file)

# %%
clerical_codes_df = prep_clerical_codes(
    cc_df, cc_4plus_df, digits=5, out_col="cc_initial_codes"
)

print(clerical_codes_df[clerical_codes_df["cc_initial_codes_invalid"].apply(len) > 0])

# %%
# Correct known clerical typos before rebuilding the cleaned code columns.
fix_lookup = {
    "43391": "43991",
    "74000": "74100",
    "86xx": "86xxx",
    "96060": "96020",
    "5611x": "4511x",
    "89291": "81291",
}
for col in ["sic_ind_occ1", "sic_ind_occ2", "sic_ind_occ3"]:
    cc_df[col] = cc_df[col].replace(fix_lookup)

fix_lookup4plus = {
    "41101;41201;42xxx;43xxx;71xxx": "41100;41201;42xxx;43xxx;71xxx",
    "43330;43339;43910": "43330;43390,43910;",
    "53103": "56103",
    "37xxx;38xxx;41202;42110;42990;43210;45200;49xxx;50xxx;522xx;56102;56290;62xxx;68xxx;6910x;0xxx;711xx;71200;72xxx;749xx;75000;79909;812xx;81300;82xxx;84xxx;85xxx;86xxx;87xxx;88xxx;90040;91xxx;93xxx": "37xxx;38xxx;41202;42110;42990;43210;45200;49xxx;50xxx;522xx;56102;56290;62xxx;68xxx;6910x;70xxx;711xx;71200;72xxx;749xx;75000;79909;812xx;81300;82xxx;84xxx;85xxx;86xxx;87xxx;88xxx;90040;91xxx;93xxx",
}
cc_4plus_df["sic_ind_occ"] = cc_4plus_df["sic_ind_occ"].replace(fix_lookup4plus)

# Backfill unresolved 4+ cases from the base clerical columns when no 4+ file row exists.
msk = (cc_df.sic_ind_occ1 == "4+") & ~cc_df.unique_id.isin(cc_4plus_df.unique_id)
for cod_num in ["1", "2", "3"]:
    cc_df.loc[msk, f"sic_ind_occ{cod_num}"] = cc_df.loc[msk, f"sic_ind{cod_num}"]

# %%
# Rebuild the cleaned clerical codes after applying the manual corrections above.
clerical_codes_df["cc_initial_codes"] = prep_clerical_codes(
    cc_df, cc_4plus_df, digits=5, out_col="cc_initial_codes"
)["cc_initial_codes"]


# %%
stg3_file = f"{work_folder}STG3.parquet"
stg3_df = pd.read_parquet(stg3_file)

# %%
model_df = prep_model_codes(stg3_df, digits=5, out_col="sa_without_kb_initial_codes")

# %%
knowledge_base_file = f"{bucket_prefix}sic_knowledgebase/sic_knowledge_base_utf8.csv"
kb_df = pd.read_csv(knowledge_base_file)
kb_df["sa_initial_codes"] = kb_df["label"].apply(
    lambda x: get_clean_n_digit_codes(parse_numerical_code(x), n=5)[0]
)
print(kb_df["sa_initial_codes"].apply(len).value_counts())
kb_df = kb_df[kb_df["sa_initial_codes"].apply(len) == 1].reset_index(drop=True).copy()

# %%
stg3_df["description"] = stg3_df["sic2007_employee"]

for df in [stg3_df, kb_df]:
    df["clean_descr"] = df["description"].str.lower()
    # Normalise descriptions so exact text matches can be used for the KB merge.
    df["clean_descr"] = df["clean_descr"].str.replace(r"[^a-z0-9 ]", "", regex=True)
    df["clean_descr"] = (
        df["clean_descr"].str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df.drop(columns=["description"], inplace=True)


# %%
combined_df = (
    stg3_df.merge(
        model_df,
        on="unique_id",
        how="left",
    )
    .merge(
        clerical_codes_df,
        on="unique_id",
        how="left",
    )
    .merge(kb_df[["sa_initial_codes", "clean_descr"]], how="left", on="clean_descr")
)

combined_df["kb_used"] = combined_df["sa_initial_codes"].notna()
combined_df.loc[~combined_df["kb_used"], "sa_initial_codes"] = combined_df.loc[
    ~combined_df["kb_used"], "sa_without_kb_initial_codes"
]

# %%
combined_df.to_parquet(
    f"{work_folder}sa_cc_combined.parquet",
    index=False,
)
# %%
cims_df = pd.read_excel(
    f"{cims_folder}TLFS_IT11_raw_data_stage_1_SIC07_multidig_lr_2_stage_20260402-2152_2026_04_13_13_57.xlsx"
)


# %%
mock_cims_df = pd.DataFrame(
    {
        "unique_id": cims_df["tlfs_id"],
        "sic_ind_occ1": cims_df["predicted_SIC07"].map(
            lambda x: x + "x" * (5 - len(x)) if pd.notna(x) else x
        ),
    }
)
cims_codes = prep_clerical_codes(
    mock_cims_df,
    digits=5,
    out_col="cims_initial_codes",
)
cims_codes["cims_code"] = cims_df["predicted_SIC07"]
cims_codes["cims_confidence"] = cims_df["confidence_SIC07"]

# %%
cims_combined_df = combined_df.merge(
    cims_codes[["unique_id", "cims_initial_codes", "cims_code", "cims_confidence"]],
    on="unique_id",
    how="inner",
    indicator=True,
)

# %%
cims_combined_df.to_parquet(
    f"{work_folder}sa_cc_cims_combined.parquet",
    index=False,
)
# %%
