"""Work in progress notebook to visualize metrics for different models.

It loads specific clerical coding data and model outputs from bucket.
The bucket name and folder (on line 32) can be manually entered or it is read from
the .env file, where it should be stored as BUCKET_PREFIX variable, i.e.:
BUCKET_PREFIX = "gs://<bucket-name>/<folder>/"

Disabled check for too long lines (f strings) and variables names (uppercase for constants)
"""

# pylint: disable=C0301,C0103,R0801

# %%
import dotenv
import pandas as pd

from survey_assist_utils.data_cleaning.prep_data import (
    prep_clerical_codes,
    prep_model_codes,
)
from survey_assist_utils.data_cleaning.sic_codes import (
    get_clean_n_digit_codes,
    parse_numerical_code,
)

# %%
bucket_prefix = dotenv.get_key(".env", "BUCKET_PREFIX")
api_data_bucket = dotenv.get_key(".env", "API_DATA_BUCKET")
if not bucket_prefix:
    raise ValueError("BUCKET_PREFIX not found in .env file. Please set it.")

# %%
# load clerical data
clerical_file = f"{bucket_prefix}original_datasets/TLFS_evaluation_data/TLFS_evaluation_data_IT9.csv"
clerical_4plus_file = (
    f"{bucket_prefix}original_datasets/TLFS_evaluation_data/Codes_for_4_plus_IT9.csv"
)
cc_df = pd.read_csv(clerical_file)
cc_4plus_df = pd.read_csv(clerical_4plus_file)

# %%
clerical_codes_df = prep_clerical_codes(
    cc_df, cc_4plus_df, digits=5, out_col="cc_initial_codes"
)

# %%
# fix typos:
fix_lookup = {
    "43391": "43991",
    "74000": "74xxx",
    "86xx": "86xxx",
    "96060": "960xx",
    "5611x": "4511x",
}
for col in ["sic_ind_occ1", "sic_ind_occ2", "sic_ind_occ3"]:
    cc_df[col] = cc_df[col].replace(fix_lookup)

fix_loookup4plus = {
    "41101;41201;42xxx;43xxx;71xxx": "41xxx;42xxx;43xxx;71xxx",
    "43330;43339;43910": "43330;43390,43910;",
    "53103": "5310x",
}
cc_4plus_df["sic_ind_occ"] = cc_4plus_df["sic_ind_occ"].replace(fix_loookup4plus)

clerical_codes_df["cc_initial_codes"] = prep_clerical_codes(
    cc_df, cc_4plus_df, digits=5, out_col="cc_initial_codes"
)["cc_initial_codes"]


# %%
stg2_file = f"{bucket_prefix}two_prompt_pipeline/2026_02_tlfs_it9_gemini25/STG2.parquet"
stg2_df = pd.read_parquet(stg2_file)

# %%
model_df = prep_model_codes(stg2_df, digits=5, out_col="sa_initial_codes")

# %%
knowledge_base_file = f"{api_data_bucket}api_config/data/sic_knowledge_base_utf8.csv"
kb_df = pd.read_csv(knowledge_base_file)
kb_df["kb_initial_codes"] = kb_df["label"].apply(
    lambda x: get_clean_n_digit_codes(parse_numerical_code(x), n=5)[0]
)
print(kb_df["kb_initial_codes"].apply(len).value_counts())
kb_df = kb_df[kb_df["kb_initial_codes"].apply(len) == 1].reset_index(drop=True).copy()

# %%
stg2_df["description"] = stg2_df["sic2007_employee"]

for df in [stg2_df, kb_df]:
    df["clean_descr"] = df["description"].str.lower()
    # remove weird characters
    df["clean_descr"] = df["clean_descr"].str.replace(r"[^a-z0-9 ]", "", regex=True)
    # remove multiple spaces
    df["clean_descr"] = (
        df["clean_descr"].str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df.drop(columns=["description"], inplace=True)


# %%
combined_df = (
    stg2_df.merge(
        model_df,
        on="unique_id",
        how="left",
    )
    .merge(
        clerical_codes_df,
        on="unique_id",
        how="left",
    )
    .merge(kb_df[["kb_initial_codes", "clean_descr"]], how="left", on="clean_descr")
)

combined_df["kb_used"] = combined_df["kb_initial_codes"].notna()
combined_df.loc[~combined_df["kb_used"], "kb_initial_codes"] = combined_df.loc[
    ~combined_df["kb_used"], "sa_initial_codes"
]


# %%
combined_df.to_parquet(
    f"{bucket_prefix}two_prompt_pipeline/2026_02_tlfs_it9_gemini25/sa_cc_combined.parquet",
    index=False,
)
# %%
