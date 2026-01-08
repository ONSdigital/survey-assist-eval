# %%
"""Notebook to analyse feedback responses.

Note: ### = commented out to pass linting
"""

import json

# pylint: disable=C0301,C0103,R0801
# %%
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import t as stats_t  # type: ignore[attr-defined]

# %matplotlib inline

data_bucket = dotenv.get_key(".env", "PREPROD_DATA_BUCKET") or ""

# %%
folder = data_bucket + "analysis-interim-results"

out_folder = (
    data_bucket + "analysis-interim-results/feedback-analysis"
)  # set to None to skip saving
out_folder = None  # type: ignore[assignment]

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
valid_responses_df = eval_df[eval_df["response_valid"]]
feedback_given_df = valid_responses_df[
    valid_responses_df["feedback_survey_ease"].apply(len) > 0
]


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
feedback_given_df[feedback_score_cols].describe()


# %%
def mark_questions_asked(row):
    """Helper function to mark cases responses where dynamic questions were asked."""
    if row["direct_lookup_classified"] or row["survey_assist_classified"]:
        return False
    if row["survey_assist_open_question"] is not None:
        return True
    print(
        'row not directly classified or SA classified, but no open question recorded. Marking as "None".'
    )
    return None


feedback_given_df["additional_questions_asked"] = feedback_given_df.apply(
    mark_questions_asked, axis=1
)

path_cols = [
    "direct_lookup_classified",
    "survey_assist_classified",
    "additional_questions_asked",
]

feedback_given_df["survey_assist_classified"] = feedback_given_df[
    "survey_assist_classified"
].fillna(False)

feedback_given_df[path_cols].describe()

# %%
ease_count_dict = feedback_given_df["feedback_survey_ease"].value_counts().to_dict()
relevance_count_dict = (
    feedback_given_df["feedback_survey_relevance"].value_counts().to_dict()
)
comfort_count_dict = (
    feedback_given_df["feedback_survey_comfort"].value_counts().to_dict()
)

fig, (ax1, ax2, ax3) = plt.subplots(
    ncols=3, nrows=1, sharey=True, figsize=(10, 4), constrained_layout=True
)
bar1 = ax1.bar(
    ease_count_dict.keys(), ease_count_dict.values(), color="#12436D", width=0.5
)
bar2 = ax2.bar(
    relevance_count_dict.keys(),
    relevance_count_dict.values(),
    color="#28A197",
    width=0.5,
)
bar3 = ax3.bar(
    comfort_count_dict.keys(), comfort_count_dict.values(), color="#801650", width=0.5
)

for barplot, ax in zip([bar1, bar2, bar3], [ax1, ax2, ax3]):
    for bar in barplot:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height()}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

ax1.set_xticklabels(
    labels=[k.replace(" or", "\nor") for k in ease_count_dict], rotation=60
)
ax2.set_xticklabels(
    labels=[k.replace(" or", "\nor") for k in relevance_count_dict], rotation=60
)
ax3.set_xticklabels(
    labels=[k.replace(" or", "\nor") for k in comfort_count_dict], rotation=60
)

ax1.set_title("Survey Ease")
ax2.set_title("Survey Relevance")
ax3.set_title("Survey Comfort")

for ax in (ax1, ax2, ax3):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

ax1.set_ylabel("Number of responses")
### plt.savefig("quant_feedback_distributions.png", dpi=275)

# %%
key_variables = path_cols + feedback_score_cols

correlations = feedback_given_df[key_variables].corr(method="pearson")
uncertainties = (1 - correlations) / np.sqrt(len(feedback_given_df))

dof_overall = len(feedback_given_df) - 2
t_values = correlations / uncertainties
p_values = 2 * stats_t.sf(np.abs(t_values), dof_overall)

small_nonzero_number = 1e-256
significance_threshold = 0.1
annot_corrs = np.where(
    (p_values > small_nonzero_number) * (p_values < significance_threshold),
    np.round(correlations, 2).astype(str),
    "",
)
annot_pvalues = np.where(  # type: ignore[call-overload]
    (p_values > small_nonzero_number) * (p_values < significance_threshold),
    p_values,
    None,
)
annot_uncertainties = np.where(
    (p_values > small_nonzero_number) * (p_values < significance_threshold),
    np.round(2 * uncertainties, 2).astype(str),
    "",
)
fig, ax = plt.subplots(figsize=(7, 7))
corr_mat = ax.imshow(correlations, vmax=1, vmin=-1, cmap="PRGn")
colour_change_limit = 0.75

for i in range(len(correlations)):
    for j in range(len(correlations)):
        annotation = ""
        colour = (
            "w" if abs(correlations.to_numpy()[i, j]) > colour_change_limit else "k"
        )
        if annot_pvalues[i, j] is not None and max(i, j) > 2:  # noqa: PLR2004
            ax.text(
                j,
                i,
                r"$r=$"
                + annot_corrs[i, j]
                + "\n"
                + r"$\pm$"
                + annot_uncertainties[i, j]
                + "\n"
                + r"$p=$"
                + f"{annot_pvalues[i, j]:.2e}",
                ha="center",
                va="center",
                color=colour,
                fontsize=7,
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
cbar = fig.colorbar(corr_mat, ax=ax, shrink=0.725, pad=0.02)
cbar.set_label(
    r"Pearson correlation coefficient $(r)$"
    + "\n"
    + r"(p-values stated if significant at $p<0.1$)"
)
plt.tight_layout()
### plt.savefig("corr_mat_feedback.png", dpi=275, transparent=True)

# %%
small_nonzero_number = 1e-256
significance_threshold = 0.1

key_variables = ["additional_questions_asked", *feedback_score_cols]

correlations = feedback_given_df[key_variables].corr(method="pearson")
uncertainties = (1 - correlations) / np.sqrt(len(feedback_given_df))

dof_overall = len(feedback_given_df) - 2
t_values = correlations / uncertainties
p_values = 2 * stats_t.sf(np.abs(t_values), dof_overall)

annot_corrs = np.where(
    (p_values > small_nonzero_number) * (p_values < significance_threshold),
    np.round(correlations, 2).astype(str),
    "",
)
annot_uncertainties = np.where(
    (p_values > small_nonzero_number) * (p_values < significance_threshold),
    np.round(2 * uncertainties, 2).astype(str),
    "",
)
annot_pvalues = np.where(  # type: ignore[call-overload]
    (p_values > small_nonzero_number) * (p_values < significance_threshold),
    p_values,
    None,
)
fig, ax = plt.subplots(figsize=(7, 7))
corr_mat = ax.imshow(correlations, vmax=1, vmin=-1, cmap="PRGn")
colour_change_limit = 0.75

for i in range(len(correlations)):
    for j in range(len(correlations)):
        annotation = ""
        colour = (
            "w" if abs(correlations.to_numpy()[i, j]) > colour_change_limit else "k"
        )
        if annot_pvalues[i, j] is not None:
            ax.text(
                j,
                i,
                r"$r=$"
                + annot_corrs[i, j]
                + r"$\pm$"
                + annot_uncertainties[i, j]
                + "\n"
                + r"$p=$"
                + f"{annot_pvalues[i, j]:.2e}",
                ha="center",
                va="center",
                color=colour,
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
cbar.set_label(
    "Pearson correlation coefficient\n"
    + r"($r \pm 2\sigma_r\text{ and }p$ stated if significant at $p<0.1$)"
)
plt.savefig("corr_mat_feedback_concise.png", dpi=275, transparent=True)
plt.show()

# %%
section_counts = feedback_given_df["most_likely_sic_section"].value_counts()
### section_counts

# %%
feedback_given_df_SIC_dummies = pd.get_dummies(
    feedback_given_df, columns=["most_likely_sic_section"], prefix="", prefix_sep=""
)
key_variables_SIC = key_variables.copy()
key_variables_SIC.extend(
    [f"{sec}" for sec in feedback_given_df["most_likely_sic_section"].unique()]
)

# %%
small_nonzero_number = 1e-256
significance_threshold = 0.25

observation_count_section = np.array(
    [section_counts[c] for c in feedback_given_df["most_likely_sic_section"].unique()]
)
dof_section_matrix = (
    np.array(
        [
            observation_count_section,
            observation_count_section,
            observation_count_section,
            observation_count_section,
        ]
    )
    - 2
)

correlations = (
    feedback_given_df_SIC_dummies[key_variables_SIC].corr(method="pearson").to_numpy()
)
correlations = correlations[:4, 4:]
uncertainties = (1 - correlations) / np.sqrt(dof_section_matrix + 2)  # +2 to map dof->n

t_values = correlations / uncertainties
p_values = stats_t.sf(np.abs(t_values), dof_section_matrix)

annot_corrs = np.where(
    (p_values > small_nonzero_number) * (p_values < significance_threshold),
    np.round(correlations, 2).astype(str),
    "",
)
annot_uncertainties = np.where(
    (p_values > small_nonzero_number) * (p_values < significance_threshold),
    np.round(2 * uncertainties, 2).astype(str),
    "",
)
annot_pvalues = np.where(  # type: ignore[call-overload]
    (p_values > small_nonzero_number) * (p_values < significance_threshold),
    p_values,
    None,
)

boundary = np.max(np.abs(correlations))
fig = plt.figure(figsize=(10, 4))
gs = GridSpec(10, 10, figure=fig, wspace=0.01, hspace=0.005)

ax = fig.add_subplot(gs[3:, :])
ax2 = fig.add_subplot(gs[1:6, :])

corr_mat = ax.imshow(
    correlations,
    vmax=boundary,
    vmin=-boundary,
    cmap="PRGn",
)
ax.set_xticks(
    range(len(key_variables_SIC[4:])),
    labels=[k.replace("_", " ") for k in key_variables_SIC[4:]],
    rotation=0,
)
ax.set_yticks(
    range(len(key_variables_SIC[:4])),
    labels=[k.replace("_", " ") for k in key_variables_SIC[:4]],
    rotation=0,
)

for i in range(correlations.shape[0]):
    for j in range(correlations.shape[1]):
        annotation = ""
        colour = (
            "w" if abs(correlations[i, j]) > colour_change_limit * boundary else "k"
        )
        if annot_pvalues[i, j] is not None:
            ax.text(
                j,
                i,
                r"$p=$" + f"\n{annot_pvalues[i, j]:.2f}",
                ha="center",
                va="center",
                color=colour,
                fontsize=6,
            )

cbar = fig.colorbar(corr_mat, ax=ax, shrink=0.6, aspect=6, pad=0.01)
cbar.set_label("Pearson\ncorrelation\ncoefficient")
ax.set_xlabel("Most-likely SIC Section")

observations_heatmap = ax2.imshow(
    np.array([section_counts[s] for s in key_variables_SIC[4:]]).reshape(1, -1),
    cmap="Greys",
)
section_cbar = fig.colorbar(
    observations_heatmap, ax=ax2, shrink=0.2125, aspect=1.5, pad=0.01
)

ax2.set_xticks(range(len(section_counts)), labels=[""] * len(section_counts))
ax2.set_yticks([])
section_cbar.set_label("Obs.")
section_cbar.set_ticks([50, 100, 150], labels=["50", "100", "150"], fontsize=7)
### plt.savefig("section_level_feedback_correlations.png", dpi=275, transparent=True)

# %%
### Parsing LLM wait times:
with open("./time-in-dynamic-questions-08-Dec-1530.json", encoding="utf8") as f:
    llm_time_data = json.load(f)["users"]

llm_wait_time_df = pd.DataFrame(llm_time_data)
llm_wait_time_df.sample(n=10)

# %%
print(
    llm_wait_time_df["person_id"].value_counts().max()
)  # check for multiple llm interaction times per user

# %%
desired_columns = [
    *feedback_given_df.columns.to_list(),
    *llm_wait_time_df.columns.to_list()[1:],
]

feedback_given_llm_waittime_df = pd.merge(
    feedback_given_df,
    llm_wait_time_df,
    how="inner",
    left_on="user",
    right_on="person_id",
)[desired_columns]

# Handle the 3 cases where people were shown dynamic Qs, then restarted the survey
feedback_given_llm_waittime_df = feedback_given_llm_waittime_df[
    feedback_given_llm_waittime_df["additional_questions_asked"]
]


# %%
small_nonzero_number = 1e-256
significance_threshold = 0.1

key_variables = ["time_to_show_dynamic_question", *feedback_score_cols]

correlations = feedback_given_llm_waittime_df[key_variables].corr(method="pearson")
uncertainties = (1 - correlations) / np.sqrt(len(feedback_given_df))

dof_overall = len(feedback_given_llm_waittime_df) - 2
t_values = correlations / uncertainties
p_values = 2 * stats_t.sf(np.abs(t_values), dof_overall)

annot_corrs = np.where(
    (p_values > small_nonzero_number) * (p_values < significance_threshold),
    np.round(correlations, 2).astype(str),
    "",
)
annot_uncertainties = np.where(
    (p_values > small_nonzero_number) * (p_values < significance_threshold),
    np.round(2 * uncertainties, 2).astype(str),
    "",
)
annot_pvalues = np.where(  # type: ignore[call-overload]
    (p_values > small_nonzero_number) * (p_values < significance_threshold),
    p_values,
    None,
)

fig, ax = plt.subplots(figsize=(7, 7))
corr_mat = ax.imshow(correlations, vmax=1, vmin=-1, cmap="PRGn")

for i in range(len(correlations)):
    for j in range(len(correlations)):
        annotation = ""
        colour = (
            "w" if abs(correlations.to_numpy()[i, j]) > colour_change_limit else "k"
        )
        if annot_pvalues[i, j] is not None:
            ax.text(
                j,
                i,
                r"$r=$"
                + annot_corrs[i, j]
                + r"$\pm$"
                + annot_uncertainties[i, j]
                + "\n"
                + r"$p=$"
                + f"{annot_pvalues[i, j]:.2e}",
                ha="center",
                va="center",
                color=colour,
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
cbar.set_label(
    "Pearson correlation coefficient\n"
    + r"($r \pm 2\sigma_r\text{ and }p$ stated if significant at $p<0.1$)"
)
### plt.savefig('corr_mat_feedback_LLM_waittime.png', dpi=275, transparent=True)

# %%
### Age-related analysis:
small_nonzero_number = 1e-256
significance_threshold = 0.25

feedback_given_df_age_dummies = pd.get_dummies(
    feedback_given_df[feedback_given_df["feedback_age_range"].notnull()],
    columns=["feedback_age_range"],
    prefix="",
    prefix_sep="",
)
key_variables_age = feedback_score_cols.copy()
key_variables_age = [
    f"{age}"
    for age in feedback_given_df[feedback_given_df["feedback_age_range"].notnull()][
        "feedback_age_range"
    ].unique()
]
key_variables_age = [*feedback_score_cols, *key_variables_age]

# %%
age_counts = feedback_given_df["feedback_age_range"].value_counts()

observation_count_age = np.array(
    [age_counts[c] for c in feedback_given_df["feedback_age_range"].unique()]
)
dof_age_matrix = (
    np.array([observation_count_age, observation_count_age, observation_count_age]) - 2
)

correlations = (
    feedback_given_df_age_dummies[key_variables_age].corr(method="pearson").to_numpy()
)
correlations = correlations[:3, 3:]
uncertainties = (1 - correlations) / np.sqrt(dof_age_matrix + 2)

t_values = correlations / uncertainties
p_values = stats_t.sf(np.abs(t_values), dof_age_matrix)

annot_corrs = np.where(
    (p_values > small_nonzero_number) * (p_values < significance_threshold),
    np.round(correlations, 2).astype(str),
    "",
)
annot_uncertainties = np.where(
    (p_values > small_nonzero_number) * (p_values < significance_threshold),
    np.round(2 * uncertainties, 2).astype(str),
    "",
)
annot_pvalues = np.where(  # type: ignore[call-overload]
    (p_values > small_nonzero_number) * (p_values < significance_threshold),
    p_values,
    None,
)

boundary = np.max(np.abs(correlations))

fig = plt.figure(figsize=(8, 6))
gs = GridSpec(10, 10, figure=fig, wspace=0.01, hspace=0.005)

ax = fig.add_subplot(gs[4:, :])
ax2 = fig.add_subplot(gs[1:4, 1:])

corr_mat = ax.imshow(
    correlations,
    vmax=boundary,
    vmin=-boundary,
    cmap="PRGn",
)
ax.set_xticks(
    range(len(key_variables_age[3:])),
    labels=[k.replace("_", " ") for k in key_variables_age[3:]],
    rotation=0,
)
ax.set_yticks(
    range(len(key_variables_age[:3])),
    labels=[k.replace("_", " ") for k in key_variables_age[:3]],
    rotation=0,
)

for i in range(correlations.shape[0]):
    for j in range(correlations.shape[1]):
        annotation = ""
        colour = (
            "w" if abs(correlations[i, j]) > colour_change_limit * boundary else "k"
        )
        if annot_pvalues[i, j] is not None:
            ax.text(
                j,
                i,
                r"$p=$" + f"{annot_pvalues[i, j]:.2f}",
                ha="center",
                va="center",
                color=colour,
                fontsize=10,
            )

cbar = fig.colorbar(corr_mat, ax=ax, shrink=1, aspect=15, pad=0.01)
cbar.set_label("Pearson\ncorrelation\ncoefficient")
ax.set_xlabel("Age Range")

observations_heatmap = ax2.imshow(
    np.array([age_counts[s] for s in key_variables_age[3:]]).reshape(1, -1),
    cmap="Greys",
)
for age_idx, ac in enumerate([age_counts[s] for s in key_variables_age[3:]]):
    colour = "w" if abs(ac) > 0.5 * np.max(age_counts) else "k"
    ax2.text(
        age_idx,
        0,
        r"$n=$" + f"{ac}",
        ha="center",
        va="center",
        color=colour,
        fontsize=10,
    )
age_cbar = fig.colorbar(observations_heatmap, ax=ax2, shrink=0.666, aspect=6, pad=0.01)

ax2.set_xticks(range(len(age_counts)), labels=[""] * len(age_counts))
ax2.set_yticks([])
age_cbar.set_label("Obs.\n\n")
plt.tight_layout()
### plt.savefig("age_feedback_correlations.png", dpi=275, transparent=True)
