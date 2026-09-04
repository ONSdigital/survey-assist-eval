"""Build `sic_kb_for_sayt.csv` for the SAYT suggester.

The notebook reads following bucket sources:
- `evaluation-pipeline/SAYT/Lookup_IT2.csv` for reviewed SAYT lookup terms, it2
- `evaluation-pipeline/SAYT/Lookup_IT3_Final.csv` for reviewed SAYT lookup terms, it3.
- `evaluation-pipeline/SAYT/Lookup_IT4_Build.xlsx` for reviewed SAYT lookup terms, it4.
- `sic_knowledgebase/sic_kb_for_classifai.csv` for the wider SIC search-text base.
- `sic_knowledgebase/sic_rephrased.csv` for reworded SIC descriptions.

It normalises SIC codes to 5 digits, creates candidate `display_text` labels from
the reviewed lookup and rephrased descriptions, removes duplicate or very similar
labels within the same code, then combines search terms from all three sources.
Each remaining `search_text` is matched to the most similar `display_text` for the
same SIC code using `sentence-transformers/all-MiniLM-L6-v2` embeddings.

The output is written to `sic_kb_for_sayt.csv` with columns `code`,
`search_text`, and `display_text`.

Expects `EVALUATION_BUCKET_NAME` to be set, loaded from `.env`.
"""

# pylint: disable=invalid-name, duplicate-code

# %%
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from survey_assist_embed_core.adapters.classifai.vectoriser import build_vectoriser
from survey_assist_utils.logging import get_logger

from survey_assist_eval.data_cleaning.code_standard import (
    SIC_EXPECTED_CODE_LENGTH,
    get_clean_n_digit_codes,
)

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")

OUTPUT_DIR = "data/sayt"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

logger = get_logger(__name__)
logger.info("Location specs", bucket_name=bucket_name, output_dir=OUTPUT_DIR)


# %%
def get_one_clean_n_digit_code(
    x: str, n=SIC_EXPECTED_CODE_LENGTH, code_type: str = "sic", expand: bool = False
) -> str | set[str]:
    """Return the one clean n-digit code if there is exactly one,
    otherwise return the original string.

    Args:
        x (str): The input code string.
        n (int, optional): The expected number of digits in the code.
            Defaults to SIC_EXPECTED_CODE_LENGTH.
        code_type (str, optional): The type of code. Defaults to "sic".
        expand (bool, optional): Whether to return all possible clean codes if
            there are multiple. Defaults to False.

    Returns:
        The clean n-digit code if exactly one exists, otherwise the original
        string or all possible codes if expand is True.
    """
    x = x.rstrip("0")
    codes = get_clean_n_digit_codes(x, n, code_type=code_type)[0]
    if len(codes) == 1:
        return next(iter(codes))
    if expand:
        return codes
    return x


# %%
sayt_df = {}
sayt_df["it3"] = pd.read_csv(
    f"gs://{bucket_name}/evaluation-pipeline/SAYT/Lookup_IT3_Final.csv", dtype=str
).rename(columns={"SIC_lookup": "display_text"})
sayt_df["it3"]["code"] = (
    sayt_df["it3"]["SIC07"]
    .apply(lambda x: x if len(x) == SIC_EXPECTED_CODE_LENGTH else f"0{x}")
    .map(get_one_clean_n_digit_code)
)

sayt_df["it2"] = pd.read_csv(
    f"gs://{bucket_name}/evaluation-pipeline/SAYT/Lookup_IT2.csv",
    dtype=str,
    encoding="windows-1252",
).rename(columns={"SIC_lookup": "display_text"})
sayt_df["it2"]["code"] = sayt_df["it2"]["SIC07"].map(get_one_clean_n_digit_code)

sayt_df["it4"] = pd.read_excel(
    f"gs://{bucket_name}/evaluation-pipeline/SAYT/Lookup_IT4_Build.xlsx", dtype=str
).rename(columns={"QEmploy.sic2007_employed_main_job_lookup_text": "display_text"})
sayt_df["it4"]["code"] = sayt_df["it4"][
    "Qemploy_sic2007_employed_main_job_lookup_code "
].apply(lambda x: get_one_clean_n_digit_code(x[:SIC_EXPECTED_CODE_LENGTH]))

# %%
sayt_df_aggr = {}
for lab, one_df in sayt_df.items():
    df = one_df.copy()
    df[f"display_text_{lab}"] = df["display_text"].str.strip()

    # explode rows by full codes
    df["full_code"] = df["code"].map(
        lambda x: get_one_clean_n_digit_code(x, expand=True)
    )
    df = df.explode("full_code").reset_index(drop=True)

    # drop high level code where lower code is present for the same full_code
    df["code_len"] = df["code"].str.len()
    code_lens = df.groupby("full_code")["code_len"].transform("max")
    df = df[df["code_len"] == code_lens]

    df = df[["code", "full_code", f"display_text_{lab}"]].rename(
        columns={"code": f"code_{lab}"}
    )
    df = df.groupby([f"code_{lab}", "full_code"]).agg(list).reset_index()
    sayt_df_aggr[lab] = df

# %%
rephrased_df = pd.read_csv(
    f"gs://{bucket_name}/sic_knowledgebase/sic_rephrased.csv", dtype=str
).rename(
    columns={"rephrased_description": "sa_rephrased_text", "sic_code": "full_code"}
)

merge_with_duplicates = rephrased_df.merge(
    sayt_df_aggr["it2"].merge(
        sayt_df_aggr["it3"].merge(sayt_df_aggr["it4"], how="outer"), how="outer"
    ),
    how="left",
).reset_index(drop=True)

# %% report missing or collapsed titles to SAYT team
msk = merge_with_duplicates["code_it4"].apply(
    lambda x: pd.isna(x) or len(x) < SIC_EXPECTED_CODE_LENGTH
)
out = merge_with_duplicates[msk]

out.head(10)
out.to_csv(
    f"gs://{bucket_name}/evaluation-pipeline/SAYT/wip/collapsed_or_missing_code_groups_it4.csv",
    index=False,
)
out.to_csv(f"{OUTPUT_DIR}/collapsed_or_missing_code_groups_it4.csv", index=False)

# %%
# Collate display texts from it4, when missing fallback to i3 then it2 and lastly rephrased

df4 = sayt_df["it4"][["code", "display_text"]]
display_text_all = df4[df4["code"].isin(rephrased_df["full_code"])]
df3 = sayt_df["it3"][["code", "display_text"]]
msk3 = df3["code"].isin(rephrased_df["full_code"]) & ~df3["code"].isin(
    display_text_all["code"]
)
display_text_all = pd.concat([display_text_all, df3[msk3]], ignore_index=True)
df2 = sayt_df["it2"][["code", "display_text"]]
msk2 = df2["code"].isin(rephrased_df["full_code"]) & ~df2["code"].isin(
    display_text_all["code"]
)
display_text_all = pd.concat([display_text_all, df2[msk2]], ignore_index=True)
df1 = rephrased_df.rename(
    columns={"sa_rephrased_text": "display_text", "full_code": "code"}
)
msk1 = ~df1["code"].isin(display_text_all["code"])
display_text_all = pd.concat([display_text_all, df1[msk1]], ignore_index=True)

logger.info(
    f"Total number of display_texts: {len(display_text_all)}",
    most_common_codes=display_text_all.groupby("code")
    .size()
    .sort_values(ascending=False)
    .head(10)
    .to_dict(),
)


# %% create the link with search texts
DISPLAY_DUPLICATE_THRESHOLD = 0.85
SEARCH_DUPLICATE_THRESHOLD = 0.95
LOG_BORDERLINE_MARGIN = 0.01
LOG_LOW_MATCH_THRESHOLD = 0.10


def cosine_similarity_matrix(
    left_embeddings: np.ndarray,
    right_embeddings: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return pairwise cosine similarities between row-wise embedding matrices."""
    left = np.asarray(left_embeddings)
    if left.ndim == 1:
        left = left.reshape(1, -1)

    if right_embeddings is None:
        right = left
    else:
        right = np.asarray(right_embeddings)
        if right.ndim == 1:
            right = right.reshape(1, -1)

    left_norms = np.linalg.norm(left, axis=1, keepdims=True)
    right_norms = np.linalg.norm(right, axis=1, keepdims=True)
    denom = np.clip(left_norms @ right_norms.T, eps, None)
    return (left @ right.T) / denom


# %%
sic_kb_for_classifai = pd.read_csv(
    f"gs://{bucket_name}/sic_knowledgebase/sic_kb_for_classifai.csv", dtype=str
).rename(columns={"text": "search_text", "label": "code"})


# %% drop duplicates
alpha_numeric = (
    display_text_all["display_text"]
    .fillna("")
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "", regex=True)
)
display_text_all = display_text_all.loc[~alpha_numeric.duplicated()].reset_index(
    drop=True
)

vectoriser = build_vectoriser("nomic-ai/nomic-embed-text-v1.5")  # "all-MiniLM-L6-v2")
display_text_embeddings = vectoriser.transform(
    display_text_all["display_text"].to_list()
)


# %%
# drop display_texts that are too similar to other display_texts for the same code
# keep the sayt approved ones (thats why the start of the ind range is sum(~higher_codes)
display_text_all["max_similarity_score"] = 0.0
display_text_all["most_similar_entry"] = None
report_similar = pd.DataFrame(
    columns=["code", "display_text", "max_similarity_score", "most_similar_entry"]
)

for ind in range(len(display_text_all)):
    code = display_text_all.loc[ind, "code"]
    display_text = display_text_all.loc[ind, "display_text"]
    one_embedding = display_text_embeddings[ind]
    # find any previous display texts with the same code
    same_code_inds = [i for i in range(ind) if display_text_all.loc[i, "code"] == code]
    if not same_code_inds:
        continue
    same_code_embeddings = display_text_embeddings[same_code_inds]
    similarities = cosine_similarity_matrix(same_code_embeddings, one_embedding)[:, 0]
    display_text_all.loc[ind, "max_similarity_score"] = max(similarities)
    display_text_all.loc[ind, "most_similar_entry"] = display_text_all.loc[
        same_code_inds[similarities.argmax()], "display_text"
    ]

# print borderline cases to sense check the threshold
borderline_msk = display_text_all["max_similarity_score"].gt(
    DISPLAY_DUPLICATE_THRESHOLD - LOG_BORDERLINE_MARGIN
)  # & display_text_all["max_similarity_score"].le(
# DISPLAY_DUPLICATE_THRESHOLD + LOG_BORDERLINE_MARGIN
# )
borderline_display_texts = display_text_all.loc[borderline_msk].sort_values(
    "max_similarity_score"
)
logger.info(
    f"Borderline cases ( {DISPLAY_DUPLICATE_THRESHOLD - LOG_BORDERLINE_MARGIN} < similarity <= "
    f"{DISPLAY_DUPLICATE_THRESHOLD + LOG_BORDERLINE_MARGIN}) for the same code. "
    "Please sense check if the threshold is appropriate.",
    borderline_display_texts=borderline_display_texts.to_dict(orient="records"),
)

to_drop_msk = display_text_all["max_similarity_score"].gt(DISPLAY_DUPLICATE_THRESHOLD)
logger.info(
    f"Dropping {sum(to_drop_msk)} display_texts that are too similar "
    "to other display_texts for the same code."
)
display_text_filtered = (
    display_text_all.loc[
        ~to_drop_msk,
        ["code", "display_text"],
    ]
    .sort_values(by=["code", "display_text"])
    .reset_index(drop=True)
)
logger.info(
    f"Total number of display_texts after filtering: {len(display_text_filtered)}",
    most_common_codes=display_text_filtered.groupby("code")
    .size()
    .sort_values(ascending=False)
    .head(10)
    .to_dict(),
)
borderline_display_texts.to_csv(f"{OUTPUT_DIR}/very_similar_display_texts.csv")

#  discuss the filtering with TLFS SAYT team, for now keep all
display_text_filtered = display_text_all.copy()


# %%
search_text_from_display = pd.concat([df4, df3, df2, df1]).rename(
    columns={"display_text": "search_text"}
)
search_text_from_display = search_text_from_display[
    search_text_from_display["code"].map(len) == SIC_EXPECTED_CODE_LENGTH
]

search_text_all = pd.concat(
    [
        search_text_from_display[["code", "search_text"]],
        sic_kb_for_classifai[["code", "search_text"]],
    ],
).reset_index(drop=True)
logger.info(
    f"Total number of search_texts: {len(search_text_all)}",
    most_common_codes=search_text_all.groupby("code")
    .size()
    .sort_values(ascending=False)
    .head(10)
    .to_dict(),
)

# %%
# Prepare embeddings for search_text (takes ~ 5 mins)
search_text_embeddings = vectoriser.transform(search_text_all["search_text"].tolist())
# Recompute display text embeddings so indices stay aligned with filtered display_text rows.
display_text_embeddings = vectoriser.transform(
    display_text_filtered["display_text"].tolist()
)
search_inds_by_code = search_text_all.groupby("code", sort=False).indices
display_inds_by_code = display_text_filtered.groupby("code", sort=False).indices

# %%
# Match each search_text to the most similar display_text for the same code.
search_text_all["search_search_similarity"] = 0.0
search_text_all["most_similar_entry"] = None
search_text_all["display_text"] = None
search_text_all["search_display_similarity"] = 0.0

for code, search_inds_arr in search_inds_by_code.items():
    search_inds = list(search_inds_arr)
    search_embeddings = search_text_embeddings[search_inds]
    search_search_similarities = cosine_similarity_matrix(search_embeddings)

    display_inds_arr = display_inds_by_code.get(code)
    if display_inds_arr is None:
        if len(search_inds) > 1:
            logger.warning(
                f"No display_text found for code {code}. Skipping for "
                f"{search_text_all.loc[search_inds, 'search_text'].tolist()}."
            )
        continue
    display_inds = list(display_inds_arr)
    display_embeddings = display_text_embeddings[display_inds]
    search_display_similarities = cosine_similarity_matrix(
        search_embeddings, display_embeddings
    )

    for num_ind, search_ind in enumerate(search_inds):
        # find the most similar search_text for the same code
        if num_ind > 0:
            search_text_all.loc[search_ind, "search_search_similarity"] = max(
                search_search_similarities[num_ind, 0:num_ind]
            )
            search_text_all.loc[search_ind, "most_similar_entry"] = search_text_all.loc[
                search_inds[search_search_similarities[num_ind, 0:num_ind].argmax()],
                "search_text",
            ]
        # find the most similar display_text for the same code
        search_text_all.loc[search_ind, "search_display_similarity"] = (
            search_display_similarities[num_ind].max()
        )
        search_text_all.loc[search_ind, "display_text"] = display_text_filtered.loc[
            display_inds[search_display_similarities[num_ind].argmax()], "display_text"
        ]

search_drop_msk = search_text_all["search_search_similarity"].gt(
    SEARCH_DUPLICATE_THRESHOLD
)
logger.info(
    f"Number of search_text entries to drop: {search_drop_msk.sum()}",
    most_common_codes=search_text_all.loc[search_drop_msk, "code"]
    .value_counts()
    .head(10)
    .to_dict(),
)
pairs_df = (
    search_text_all.loc[
        ~search_drop_msk,
        ["code", "search_text", "display_text", "search_display_similarity"],
    ]
    .sort_values(by=["code", "search_text"])
    .reset_index(drop=True)
)

# %%
# Inspect entries with low similarity scores, these may be worth adding to the display_texts.
low_similarity_pairs = (
    pairs_df[pairs_df["search_display_similarity"] < LOG_LOW_MATCH_THRESHOLD]
    .sort_values(["code", "search_display_similarity"])
    .reset_index(drop=True)
)
logger.warning(
    f"Found {len(low_similarity_pairs)} search_text/display_text pairs with low similarity "
    f"(similarity < {LOG_LOW_MATCH_THRESHOLD}). "
    "Please consider extending the display_texts input to cover these cases.",
    low_similarity_pairs=low_similarity_pairs.to_dict(orient="records"),
)
low_similarity_pairs.to_csv(
    f"{OUTPUT_DIR}/low_similarity_search_display_pairs.csv", index=False
)


# %%
out_df = (
    pairs_df[["code", "search_text", "display_text"]]
    .sort_values(by=["code", "search_text"])
    .reset_index(drop=True)
)
logger.info(f"Output dataframe shape: {out_df.shape}")
out_df.to_csv(f"{OUTPUT_DIR}/sic_kb_for_sayt.csv", index=False)

# %%
