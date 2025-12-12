# %%
"""Notebook to analyse feedback responses.

Note: ### = commented out to pass linting
"""

# pylint: disable=C0301,C0103,R0801
# %%
import dotenv
import matplotlib.pyplot as plt
import pandas as pd

# %matplotlib inline

data_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET") or ""

# %%
folder = data_bucket + "analysis-interim-results"

out_folder = (
    data_bucket + "analysis-interim-results/feedback-analysis"
)  # set to None to skip saving

# %%
# read data exported from firesore
eval_df = pd.read_parquet(folder + "/evaluation_df_with_sa_clean_codes.parquet")

# %%
### eval_df.columns

# %%
feedback_cols = [
    "feedback_survey_ease",
    "feedback_survey_relevance",
    "feedback_survey_comfort",
]

feedback_score_cols = ["ease_score", "relevance_score", "comfort_score"]

feedback_given_df = eval_df[eval_df["feedback_survey_ease"].apply(len) > 0]

ease_map = {
    "very easy": 5,
    "easy": 4,
    "neither easy or difficult": 3,
    "difficult": 2,
    "very difficult": 1,
}

relevance_map = {
    "very relevant": 5,
    "relevant": 4,
    "neither relevant or irrelevant": 3,
    "irrelevant": 2,
    "very irrelevant": 1,
}

comfort_map = {
    "very comfortable": 5,
    "comfortable": 4,
    "neither comfortable or uncomfortable": 3,
    "uncomfortable": 2,
    "very uncomfortable": 1,
}

feedback_given_df["ease_score"] = feedback_given_df["feedback_survey_ease"].apply(
    lambda r: ease_map[r]
)
feedback_given_df["relevance_score"] = feedback_given_df[
    "feedback_survey_relevance"
].apply(lambda r: relevance_map[r])
feedback_given_df["comfort_score"] = feedback_given_df["feedback_survey_comfort"].apply(
    lambda r: comfort_map[r]
)

# %%
print(
    f"{len(feedback_given_df)} respondents submitted feedback ({100*len(feedback_given_df)/len(eval_df):.3f}%)"
)

for fsc in feedback_score_cols:
    print(f"\n{fsc}\n", feedback_given_df[fsc].describe())

# %%
plt.figure(figsize=(5, 3))
plt.hist(feedback_given_df["ease_score"], density=False)
plt.title("feedback ease response")
plt.ylabel("number of responses")
plt.xticks(
    range(1, len(feedback_given_df["ease_score"].unique()) + 1),
    labels=[k.replace(" ", "\n") for k in reversed(ease_map.keys())],
    rotation=20,
)

plt.figure(figsize=(5, 3))
plt.hist(feedback_given_df["relevance_score"], density=False)
plt.title("feedback relevance response")
plt.ylabel("number of responses")
plt.xticks(
    range(1, len(feedback_given_df["relevance_score"].unique()) + 1),
    labels=[k.replace(" ", "\n") for k in reversed(relevance_map.keys())],
    rotation=20,
)

plt.figure(figsize=(5, 3))
plt.hist(feedback_given_df["comfort_score"], density=False)
plt.title("feedback comfort response")
plt.ylabel("number of responses")
plt.xticks(
    range(1, len(feedback_given_df["comfort_score"].unique()) + 1),
    labels=[k.replace(" ", "\n") for k in reversed(comfort_map.keys())],
    rotation=20,
)

# %%
key_variables = [
    "direct_lookup_classified",
    "survey_assist_classified",
]
key_variables.extend(feedback_score_cols)

fig, ax = plt.subplots(figsize=(5, 5))
corr_mat = ax.imshow(
    feedback_given_df[key_variables].corr(method="pearson"),
    vmax=1,
    vmin=-1,
    cmap="PRGn",
)
ax.set_xticks(
    range(len(key_variables)),
    labels=[k.replace("_", "\n") for k in key_variables],
    rotation=20,
)
ax.set_yticks(
    range(len(key_variables)),
    labels=[k.replace("_", "\n") for k in key_variables],
    rotation=0,
)
cbar = fig.colorbar(corr_mat, ax=ax, shrink=0.8)
cbar.set_label("Pearson correlation coefficient")
plt.show()

# %%
feedback_given_df["most_likely_sic_section"].value_counts()

# %%
feedback_given_df_SIC_dummies = pd.get_dummies(
    feedback_given_df, columns=["most_likely_sic_section"], prefix="", prefix_sep=""
)
key_variables_SIC = key_variables.copy()
key_variables_SIC.extend(
    [f"{sec}" for sec in feedback_given_df["most_likely_sic_section"].unique()]
)

# %%
# Visualisation still WIP

fig, ax = plt.subplots(figsize=(10, 10))
corr_mat = ax.imshow(
    feedback_given_df_SIC_dummies[key_variables_SIC].corr(method="pearson"),
    vmax=1,
    vmin=-1,
    cmap="PRGn",
)
ax.set_xticks(
    range(len(key_variables_SIC)),
    labels=[k.replace("_", "\n") for k in key_variables_SIC],
    rotation=20,
)
ax.set_yticks(
    range(len(key_variables_SIC)),
    labels=[k.replace("_", "\n") for k in key_variables_SIC],
    rotation=0,
)
cbar = fig.colorbar(corr_mat, ax=ax, shrink=0.8)
cbar.set_label("Pearson correlation coefficient")
### plt.show()

# %%
