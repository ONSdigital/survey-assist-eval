"""Build `sic_kb_for_sayt.csv` for the SAYT suggester.

The notebook reads three bucket sources:
- `evaluation-pipeline/SAYT/Lookup_IT3_Final.csv` for reviewed SAYT lookup terms.
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

# pylint: disable=invalid-name

# %%
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from survey_assist_embed_core.adapters.classifai.vectoriser import build_vectoriser
from survey_assist_utils.logging import get_logger

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
SIC_CODE_LENGTH = 5
DISPLAY_DUPLICATE_THRESHOLD = 0.85
SEARCH_DUPLICATE_THRESHOLD = 0.95
LOG_BORDERLINE_DUPLICATE_THRESHOLD = DISPLAY_DUPLICATE_THRESHOLD + 0.03
LOG_LOW_MATCH_THRESHOLD = 0.10

# %%
sayt_df = pd.read_csv(
    f"gs://{bucket_name}/evaluation-pipeline/SAYT/Lookup_IT3_Final.csv", dtype=str
).rename(columns={"SIC_lookup": "search_text"})
sayt_df["code"] = sayt_df["SIC07"].apply(
    lambda x: x if len(x) == SIC_CODE_LENGTH else f"0{x}"
)
sayt_df["display_text"] = sayt_df["search_text"]
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
rephrased_df["display_text"] = rephrased_df["search_text"]


# %%
# remove higher level codes from SAYT data
higher_codes = ~sayt_df["code"].isin(rephrased_df["code"])
logger.warning(
    f"Following sayt records have higher level codes and will be removed:\n"
    f"{sayt_df.loc[higher_codes, ['code', 'search_text']]}"
)

# %%
display_text_all = pd.concat(
    [
        sayt_df.loc[~higher_codes, ["code", "display_text"]],
        rephrased_df[["code", "display_text"]],
    ],
    ignore_index=True,
)
logger.info(f"Total number of display_texts: {len(display_text_all)}")
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
    display_text_all["display_text"].to_list()
)

# %%
# drop display_texts that are too similar to other display_texts for the same code
# keep the sayt approved ones (thats why the start of the ind range is sum(~higher_codes)
to_drop = [False] * len(display_text_all)

for ind in range(sum(~higher_codes), len(display_text_all)):
    code = display_text_all.loc[ind, "code"]
    display_text = display_text_all.loc[ind, "display_text"]
    embedding = display_text_embeddings[ind]
    # find any previous display texts with the same code
    same_code_inds = [i for i in range(ind) if display_text_all.loc[i, "code"] == code]
    if not same_code_inds:
        continue
    same_code_embeddings = display_text_embeddings[same_code_inds]
    # compute cosine similarity
    similarities = (same_code_embeddings @ embedding) / (
        (same_code_embeddings**2).sum(axis=1) ** 0.5 * (embedding**2).sum() ** 0.5
    )
    if any(similarities > DISPLAY_DUPLICATE_THRESHOLD):
        to_drop[ind] = True
        # print borderline cases to sense check the threshold
        if all(similarities < LOG_BORDERLINE_DUPLICATE_THRESHOLD):
            logger.info(
                f"Dropping display_text '{display_text}' for code {code} "
                f"as it is too similar to other display texts: "
                f"{display_text_all.loc[same_code_inds, 'display_text'].tolist()}."
            )

logger.info(
    f"Dropping {sum(to_drop)} display_texts that are too similar"
    "to other display_texts for the same code."
)
display_text_filtered = (
    display_text_all.loc[~pd.Series(to_drop), ["code", "display_text"]]
    .sort_values(by=["code", "display_text"])
    .reset_index(drop=True)
)
logger.info(
    f"Total number of display_texts after filtering: {len(display_text_filtered)}"
)
display_text_filtered.groupby("code").size().sort_values(ascending=False).head(10)

# %%
search_text_all = pd.concat(
    [
        sayt_df.loc[~higher_codes, ["code", "search_text"]],
        rephrased_df[["code", "search_text"]],
        sic_kb_for_classifai[["code", "search_text"]],
    ],
).reset_index(drop=True)
search_text_all.groupby("code").size().sort_values(ascending=False).head(10)

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
out_rows = []

for code, search_inds_arr in search_inds_by_code.items():
    search_inds = search_inds_arr.tolist()
    search_embeddings = search_text_embeddings[search_inds]
    search_norms = np.linalg.norm(search_embeddings, axis=1, keepdims=True)
    search_denom = np.clip(search_norms @ search_norms.T, 1e-12, None)
    search_similarities = (search_embeddings @ search_embeddings.T) / search_denom

    display_inds_arr = display_inds_by_code.get(code)
    if display_inds_arr is None:
        if len(search_inds) > 1:
            logger.warning(
                f"No display_text found for code {code}. Skipping for "
                f"{search_text_all.loc[search_inds, 'search_text'].tolist()}."
            )
        continue
    display_inds = display_inds_arr.tolist()
    display_embeddings = display_text_embeddings[display_inds]
    display_norms = np.linalg.norm(display_embeddings, axis=1, keepdims=True).T
    display_denom = np.clip(search_norms * display_norms, 1e-12, None)
    display_similarities = (search_embeddings @ display_embeddings.T) / display_denom

    # remove search_texts that are too similar to each other (cosine similarity > 0.95)
    for num_ind, search_ind in enumerate(search_inds):
        if any(search_similarities[num_ind, 0:num_ind] > SEARCH_DUPLICATE_THRESHOLD):
            continue

        # find the display_text with the highest similarity to the search_text
        search_text = search_text_all.loc[search_ind, "search_text"]
        display_ind = display_inds[display_similarities[num_ind].argmax()]
        display_text = display_text_filtered.loc[display_ind, "display_text"]
        out_rows.append(
            {
                "code": code,
                "search_text": search_text,
                "display_text": display_text,
            }
        )
        max_sim = display_similarities[num_ind].max()

        # Log examples for low similarity matches (< 0.1)
        # these may be worth adding to the display_texts.
        if max_sim < LOG_LOW_MATCH_THRESHOLD:
            logger.warning(
                f"Bad match for search_text '{search_text}' and display_text '{display_text}' "
                f"for code {code} (max similarity {max_sim:.2f})."
            )
out_df = pd.DataFrame(out_rows, columns=["code", "search_text", "display_text"])

# %%
out_df = out_df.sort_values(by=["code", "search_text"]).reset_index(drop=True)
logger.info(f"Output dataframe shape: {out_df.shape}")
# %%
out_df.to_csv(f"{OUTPUT_DIR}/sic_kb_for_sayt.csv", index=False)

# %%
