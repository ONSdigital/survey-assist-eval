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
logger.info(
    f"Total number of display_texts: {len(display_text_all)}",
    most_common_codes=display_text_all.groupby("code")
    .size()
    .sort_values(ascending=False)
    .head(10)
    .to_dict(),
)

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
display_text_all["max_similarity_score"] = 0.0
display_text_all["most_similar_entry"] = None

for ind in range(sum(~higher_codes), len(display_text_all)):
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
) & display_text_all["max_similarity_score"].le(
    DISPLAY_DUPLICATE_THRESHOLD + LOG_BORDERLINE_MARGIN
)
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
    f"Dropping {sum(to_drop_msk)} display_texts that are too similar"
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
pairs_rows = []

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

    # remove search_texts that are too similar to each other
    for num_ind, search_ind in enumerate(search_inds):
        if any(
            search_search_similarities[num_ind, 0:num_ind] > SEARCH_DUPLICATE_THRESHOLD
        ):
            continue

        # find the display_text with the highest similarity to the search_text
        search_text = search_text_all.loc[search_ind, "search_text"]
        display_ind = display_inds[search_display_similarities[num_ind].argmax()]
        display_text = display_text_filtered.loc[display_ind, "display_text"]
        similarity_score = search_display_similarities[num_ind].max()
        pairs_rows.append(
            {
                "code": code,
                "search_text": search_text,
                "display_text": display_text,
                "similarity_score": similarity_score,
            }
        )

pairs_df = pd.DataFrame(
    pairs_rows, columns=["code", "search_text", "display_text", "similarity_score"]
)

# %%
# Inspect entries with low similarity scores, these may be worth adding to the display_texts.
low_similarity_pairs = (
    pairs_df[pairs_df["similarity_score"] < LOG_LOW_MATCH_THRESHOLD]
    .sort_values(["code", "similarity_score"])
    .reset_index(drop=True)
)
logger.warning(
    f"Found {len(low_similarity_pairs)} search_text/display_text pairs with low similarity "
    f"(similarity < {LOG_LOW_MATCH_THRESHOLD}). "
    "Please consider extending the display_texts input to cover these cases.",
    low_similarity_pairs=low_similarity_pairs.to_dict(orient="records"),
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
