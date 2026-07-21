"""Combine extended SIC knowledge base with reviewed sayt lookup/rephrased data.

Expects following environment variables to be set:
- EVALUATION_BUCKET_NAME: name of GCS bucket where the data is stored
The variables are loaded from the ".env" file.

The goal is created a dataset of two columns search_text and display_text that
can be used to build a SAYT suggester.
- The search_text column contains the text that will be used for searching and is
based on the extended SIC (activities) knowledge base.
- The display_text column contains the text that will be displayed in the suggestions.
There should be pretty labels for each 5-digit SIC code (can be more than one per group).
"""

# ruff: noqa: PLR2004
# pylint: disable=protected-access,redefined-outer-name,C0103

# %%
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# pylint: disable=R0801
from survey_assist_embed_core.adapters.classifai.vectoriser import build_vectoriser
from survey_assist_utils.logging import get_logger

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")

output_dir = "data/sayt"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logger = get_logger(__name__)
logger.info("Location specs", bucket_name=bucket_name, output_dir=output_dir)


# %%
lookup_file_name = f"gs://{bucket_name}/evaluation-pipeline/SAYT/Lookup_IT3_Final.csv"
sayt_df = pd.read_csv(lookup_file_name, dtype=str).rename(
    columns={"SIC_lookup": "search_text"}
)
sayt_df["code"] = sayt_df["SIC07"].apply(lambda x: x if len(x) == 5 else f"0{x}")
sayt_df["display_text"] = sayt_df["search_text"] + ": " + sayt_df["code"]
sayt_df = (
    sayt_df[["code", "search_text", "display_text"]]
    .sort_values(by=["code", "search_text"])
    .reset_index(drop=True)
)

# %%
sic_kb_for_classifai = pd.read_csv(
    f"gs://{bucket_name}/sic_knowledgebase/sic_kb_for_classifai.csv", dtype=str
).rename(columns={"text": "search_text", "label": "code"})
rephrased_df = (
    pd.read_csv(f"gs://{bucket_name}/sic_knowledgebase/sic_rephrased.csv", dtype=str)
    .rename(columns={"rephrased_description": "search_text", "sic_code": "code"})
    .sort_values(by=["code", "search_text"])
    .reset_index(drop=True)
)
rephrased_df["display_text"] = rephrased_df["search_text"] + ": " + rephrased_df["code"]


# %%
# remove higher level codes from SAYT data
higher_codes = ~sayt_df["code"].isin(rephrased_df["code"])
print(
    f"Following sayt records have higher level codes and will be removed:\n{
        sayt_df.loc[higher_codes, ['code', 'search_text']]}"
)

# %%
display_text_all = pd.concat(
    [
        sayt_df.loc[~higher_codes, ["code", "display_text"]],
        rephrased_df[["code", "display_text"]],
    ],
    ignore_index=True,
)
print(f"Total number of display_texts: {len(display_text_all)}")
display_text_all.groupby("code").size().sort_values(ascending=False).head(10)

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

vectoriser = build_vectoriser("sentence-transformers/all-MiniLM-L6-v2")
display_text_embeddings = vectoriser.transform(
    display_text_all["display_text"].fillna("").str.slice(stop=-7).tolist()
)

# %%
to_drop = [False] * len(display_text_all)
for ind in range(sum(~higher_codes), len(display_text_all)):
    code = display_text_all.loc[ind, "code"]
    display_text = display_text_all.loc[ind, "display_text"]
    embedding = display_text_embeddings[ind]
    # find all other display texts with the same code
    same_code_inds = display_text_all.index[display_text_all["code"] == code].tolist()
    same_code_inds.remove(ind)
    if not same_code_inds:
        continue
    same_code_embeddings = display_text_embeddings[same_code_inds]
    # compute cosine similarity
    similarities = (same_code_embeddings @ embedding) / (
        (same_code_embeddings**2).sum(axis=1) ** 0.5 * (embedding**2).sum() ** 0.5
    )
    if any(similarities > 0.85):
        to_drop[ind] = True
        # print borderline cases to sense check the threshold
        if all(similarities < 0.88):
            print(
                f"Dropping display_text '{display_text}' for code {code} "
                f"as it is too similar to other display texts: "
                f"{display_text_all.loc[same_code_inds, 'display_text'].tolist()}."
            )

print(
    f"Dropping {sum(to_drop)} display_texts that are too similar"
    "to other display_texts for the same code."
)
display_text_filtered = (
    display_text_all.loc[~pd.Series(to_drop), ["code", "display_text"]]
    .sort_values(by=["code", "display_text"])
    .reset_index(drop=True)
)
print(f"Total number of display_texts after filtering: {len(display_text_filtered)}")
display_text_filtered.groupby("code").size().sort_values(ascending=False).head(10)

# %%
search_text_all = pd.concat(
    [
        sayt_df[["code", "search_text"]],
        rephrased_df[["code", "search_text"]],
        sic_kb_for_classifai[["code", "search_text"]],
    ],
).reset_index(drop=True)
search_text_all.groupby("code").size().sort_values(ascending=False).head(10)

# %% (takes ~ 5 mins)
search_text_embeddings = vectoriser.transform(search_text_all["search_text"].tolist())
# Recompute embeddings so indices stay aligned with filtered display_text rows.
display_text_embeddings = vectoriser.transform(
    display_text_filtered["display_text"].fillna("").str.slice(stop=-7).tolist()
)

# %%
out_df = pd.DataFrame(columns=["code", "search_text", "display_text"])

for code in search_text_all["code"].unique():
    search_inds = search_text_all.index[search_text_all["code"] == code].tolist()
    search_embeddings = search_text_embeddings[search_inds]
    search_norms = np.linalg.norm(search_embeddings, axis=1, keepdims=True)
    search_denom = np.clip(search_norms @ search_norms.T, 1e-12, None)
    search_similarities = (search_embeddings @ search_embeddings.T) / search_denom

    display_rows = display_text_filtered[display_text_filtered["code"] == code]
    if display_rows.empty:
        if len(search_inds) > 1:
            logger.warning(
                f"No display_text found for code {code}. Skipping for "
                f"{search_text_all.loc[search_inds, 'search_text'].tolist()}."
            )
        continue
    display_inds = display_rows.index.tolist()
    display_embeddings = display_text_embeddings[display_inds]
    display_norms = np.linalg.norm(display_embeddings, axis=1, keepdims=True).T
    display_denom = np.clip(search_norms * display_norms, 1e-12, None)
    display_similarities = (search_embeddings @ display_embeddings.T) / display_denom

    # remove search_texts that are too similar to each other (cosine similarity > 0.95)
    for num_ind, search_ind in enumerate(search_inds):
        if any(search_similarities[num_ind, 0:num_ind] > 0.95):
            continue

        # find the display_text with the highest similarity to the search_text
        search_text = search_text_all.loc[search_ind, "search_text"]
        display_ind = display_inds[display_similarities[num_ind].argmax()]
        display_text = display_text_filtered.loc[display_ind, "display_text"]
        out_df = pd.concat(
            [
                out_df,
                pd.DataFrame(
                    {
                        "code": [code],
                        "search_text": [search_text],
                        "display_text": [display_text],
                    }
                ),
            ],
            ignore_index=True,
        )
        max_sim = display_similarities[num_ind].max()
        if max_sim < 0.10:
            logger.warning(
                f"Bad match for search_text '{search_text}' and display_text '{display_text}' "
                f"for code {code} (max similarity {max_sim:.2f})."
            )


# %%
out_df = out_df.sort_values(by=["code", "search_text"]).reset_index(drop=True)
print(out_df.shape)
# %%
out_df.to_csv(f"{output_dir}/sic_kb_for_sayt.csv", index=False)

# %%
