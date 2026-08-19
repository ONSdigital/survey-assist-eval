# %%
"""Run small tests for industry descriptions SAYT.

Expects following environment variables to be set:
- EVALUATION_BUCKET_NAME: name of GCS bucket where the data is stored
The variables are loaded from the ".env" file.
"""

# pylint: disable=invalid-name, duplicate-code

# %%
import os

import pandas as pd
from dotenv import load_dotenv
from survey_assist_embed_core.sayt import (
    NgramRetrieverSpec,
    PrefixRetrieverSpec,
    SemanticRetrieverSpec,
)
from survey_assist_utils.logging import get_logger

from notebooks.sayt.sayt_utils import (
    build_lookup_suggester,
    create_figure,
    get_suggestions_by_chars,
    melt_results_for_analysis,
)
from survey_assist_eval.evaluation.sayt.performance_metrics_functions import (
    build_sayt_metrics_comparison_table,
    compute_performance_metrics_from_suggestions,
)

# %%
SIC_CODE_LENGTH = 5
MAX_SUGGESTIONS = 9

NUM_CHARACTERS_LIST = range(4, MAX_SUGGESTIONS)

# %%
load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")

OUTPUT_DIR = "data/figures/sayt/min_characters"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

logger = get_logger(__name__)
logger.info("Location specs", bucket_name=bucket_name, output_dir=OUTPUT_DIR)

# %%
test_df = pd.read_excel(
    f"gs://{bucket_name}/evaluation-pipeline/SAYT/SAYT matching.xlsx",
    dtype=str,
    nrows=100,  # Excel formatting causes 10s of thousands of blank input rows after the real 100
    header=1,  # first row is header
)
rename_columns = {
    "Correct SIC code": "correct_sic_code",
    "Full entry looking for": "full_entry",
    "Position of correct SIC ": "rank_5chars_Blaise (as reported from SAYT team)",
    "Position of correct SIC .1": "_rank_5chars_sa_shared",
}

test_df = test_df.rename(columns=rename_columns)
test_df = test_df[rename_columns.values()]

# clean the rank values reported by the SAYT team
for col in [
    "rank_5chars_Blaise (as reported from SAYT team)",
    "_rank_5chars_sa_shared",
]:
    test_df[col] = pd.to_numeric(
        test_df[col].replace({"5 or 12": "5"}), errors="coerce"
    )

# %%
LOOKUP_FILE_NAME = f"gs://{bucket_name}/evaluation-pipeline/SAYT/Lookup_IT3_Final.csv"
sayt_df = pd.read_csv(LOOKUP_FILE_NAME, dtype=str)
sayt_df["code"] = sayt_df["SIC07"].apply(
    lambda x: x if len(x) == SIC_CODE_LENGTH else f"0{x}"
)
sayt_df["display_text_with_code"] = sayt_df["SIC_lookup"] + ": " + sayt_df["code"]

sayt_corpus = list(
    zip(sayt_df["SIC_lookup"], sayt_df["display_text_with_code"], strict=False)
)

# %%
# Commented out due to workstation limitations.
# sic_kb_for_classifai = pd.read_csv(
#     f"gs://{bucket_name}/sic_knowledgebase/sic_kb_for_sayt.csv", dtype=str
# )
# sic_kb_for_classifai["display_text_with_code"] = (
#     sic_kb_for_classifai["display_text"] + ": " + sic_kb_for_classifai["code"]
# )


# sayt2_corpus = list(
#     zip(
#         sic_kb_for_classifai["search_text"],
#         sic_kb_for_classifai["display_text_with_code"],
#         strict=False,
#     )
# )

# %%
# create suggesters using only one of the retrievers:
# PrefixRetrieverSpec, NgramRetrieverSpec, or SemanticRetrieverSpec.

suggesters_simple = {
    "Blaise proxy method (prefix only)": build_lookup_suggester(
        sayt_corpus, retrievers=[PrefixRetrieverSpec()]
    ),
    "Blaise proxy method (ngram only)": build_lookup_suggester(
        sayt_corpus, retrievers=[NgramRetrieverSpec()]
    ),
    "Semantic retriever only": build_lookup_suggester(
        sayt_corpus, retrievers=[SemanticRetrieverSpec()]
    ),
}

# %%
# Test for interactions between suggesters

suggesters_pairs = {
    "Blaise proxy method (prefix and ngram)": build_lookup_suggester(
        sayt_corpus, retrievers=[PrefixRetrieverSpec(), NgramRetrieverSpec()]
    ),
    "Hybrid approach (prefix and semantic)": build_lookup_suggester(
        sayt_corpus, retrievers=[PrefixRetrieverSpec(), SemanticRetrieverSpec()]
    ),
    "Hybrid approach (ngram and semantic)": build_lookup_suggester(
        sayt_corpus, retrievers=[NgramRetrieverSpec(), SemanticRetrieverSpec()]
    ),
}

# %%
suggesters_three = {
    "Hybrid approach (ngram, prefix and semantic)": build_lookup_suggester(
        sayt_corpus,
        retrievers=[
            NgramRetrieverSpec(),
            PrefixRetrieverSpec(),
            SemanticRetrieverSpec(),
        ],
    ),
}

# # %%
# # create suggesters using only one of the retrievers:
# # PrefixRetrieverSpec, NgramRetrieverSpec, or SemanticRetrieverSpec.

# suggesters_simple = {
#     "Blaise proxy method (prefix only)": build_lookup_suggester(
#         sayt2_corpus, retrievers=[PrefixRetrieverSpec()]
#     ),
#     "Blaise proxy method (ngram only)": build_lookup_suggester(
#         sayt2_corpus, retrievers=[NgramRetrieverSpec()]
#     ),
#     "Semantic retriever only": build_lookup_suggester(
#         sayt_corpus, retrievers=[SemanticRetrieverSpec()]
#     ),
# }

# # %%
# # Test for interactions between suggesters

# suggesters_pairs = {
#     "Blaise proxy method (prefix and ngram)": build_lookup_suggester(
#         sayt2_corpus, retrievers=[PrefixRetrieverSpec(), NgramRetrieverSpec()]
#     ),
#     "Hybrid approach (prefix and semantic)": build_lookup_suggester(
#         sayt2_corpus, retrievers=[PrefixRetrieverSpec(), SemanticRetrieverSpec()]
#     ),
#     "Hybrid approach (ngram and semantic)": build_lookup_suggester(
#         sayt2_corpus, retrievers=[NgramRetrieverSpec(), SemanticRetrieverSpec()]
#     ),
# }

# # %%
# suggesters_three = {
#     "Hybrid approach (ngram, prefix and semantic)": build_lookup_suggester(
#         sayt2_corpus,
#         retrievers=[
#             NgramRetrieverSpec(),
#             PrefixRetrieverSpec(),
#             SemanticRetrieverSpec(),
#         ],
#     ),
# }


# %%
def run_eval_for_suggesters(
    df: pd.DataFrame,
    suggesters_dict: dict,
    num_chars: list[int],
    suggestions_limit: int = 9,
    output_dir: str = OUTPUT_DIR,
):
    """Use functions necessary to create a dataframe that allows for grouping by
        rank and suggester type. Create plots.

    Args:
        df (pd.DataFrame): dataframe to be tested.
        suggesters_dict (dict): a dictionary with suggester models.
        num_chars (list): number of characters to be tested.
        suggestions_limit: the maximum rank of suggestions considered as valid.
        output_dir (str): path to file location to be saved.

    Return:
        pd.DataFrame: dataframe with results from suggesters, split by the type of suggester
            and number of characters.

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_copy = df.copy()

    suggestions_df, avg_ms_dict = get_suggestions_by_chars(
        df_copy,
        suggesters_dict=suggesters_dict,
        num_chars=num_chars,
        suggestions_limit=suggestions_limit,
    )

    melt = melt_results_for_analysis(df=suggestions_df)

    fig = create_figure(melt, output_dir=output_dir)

    return melt, suggestions_df, avg_ms_dict, fig


# %%
def suggester_analysis_table(
    df: pd.DataFrame,
    suggester_name: str | None = None,
    num_chars: int | None = None,
):
    """Allows suggesters analysis in a table format.

    Args:
        df (pd.DataFrame): dataframe with suggestions from suggester.
            Requires columns: "suggester", "num_chars", "rank".
        suggester_name (str): suggester name to be analysed.
        num_chars (int): number of characters to be checked.

    Return:
        filtered dataframe with specified suggester name and/or number of characters.
    """
    df_copy = df.copy()

    results = df_copy.groupby(["suggester", "num_chars", "rank"]).count()

    # Helper function
    def print_for_suggester_and_chars(
        df: pd.DataFrame,
        suggester_name: str | None = None,
        num_chars: int | None = None,
    ):
        """Allows printing tables for specified parameters. Handles if any or either are
            not specified.

        Args:
            df (pd.DataFrame): dataframe containing grouped suggestions.
            suggester_name (str): suggester name to be analysed.
            num_chars (int): number of characters to be checked.

        Return:
            filtered dataframe with specified suggester name and/or number of characters.
        """
        if num_chars is None and suggester_name is None:
            return df
        if num_chars is None:
            return df.loc[(suggester_name)]
        if suggester_name is None:
            return df.loc[pd.IndexSlice[:, num_chars], :]
        return df.loc[(suggester_name, num_chars)]

    return print_for_suggester_and_chars(
        df=results, suggester_name=suggester_name, num_chars=num_chars
    )


# %%
melt_df_simple, suggestions_df_simple, avg_ms_dict_simple, fig_simple = (
    run_eval_for_suggesters(
        test_df, suggesters_simple, NUM_CHARACTERS_LIST, output_dir=OUTPUT_DIR
    )
)

# %%
suggester_analysis_table(melt_df_simple)

# %%
suggester_analysis_table(melt_df_simple, num_chars=6)

# %%
melt_df_pairs, suggestions_df_pairs, avg_ms_dict_pairs, fig_pairs = (
    run_eval_for_suggesters(
        test_df,
        suggesters_pairs,
        NUM_CHARACTERS_LIST,
        output_dir=f"{OUTPUT_DIR}_pairs",
    )
)

# %%
suggester_analysis_table(melt_df_pairs)

# %%
suggester_analysis_table(melt_df_pairs, num_chars=6)

# %%
melt_df_three, suggestions_df_three, avg_ms_dict_three, fig_three = (
    run_eval_for_suggesters(
        test_df,
        suggesters_three,
        NUM_CHARACTERS_LIST,
        output_dir=f"{OUTPUT_DIR}_three",
    )
)

# %%
melt_df_all = (
    pd.concat([melt_df_simple, melt_df_pairs, melt_df_three], ignore_index=True)
    .drop_duplicates()
    .copy()
)

# %%
suggestions_df_all = pd.concat(
    [suggestions_df_simple, suggestions_df_pairs, suggestions_df_three],
    ignore_index=True,
).copy()

suggestions_columns = suggestions_df_all.columns[
    suggestions_df_all.columns.str.startswith("suggestions_")
].tolist()

for col in suggestions_columns:
    suggestions_df_all[col] = suggestions_df_all[col].apply(
        lambda x: x if isinstance(x, list) else []
    )

# %%
avg_ms_dict_all = avg_ms_dict_simple | avg_ms_dict_pairs | avg_ms_dict_three

# %%
fig_all = create_figure(melt_df_all, output_dir=OUTPUT_DIR)

# %%
# Specify datframe to check

df_to_check = suggestions_df_pairs
time_to_check = avg_ms_dict_pairs


# Note: performance metrics doesn't allow NaN values (result of concatenating dataframes
# from three different approaches). This results in metrics being unreliable and reults
# should be assessed from using one of:
# - suggestions_df_simple
# - suggestions_df_pairs
# - suggestions_df_three
# df_to_check = suggestions_df_all
# time_to_check = avg_ms_dict_all

# %%
# prepare column names
correct_code_col = "correct_sic_code"
suggestions_cols_to_compare = df_to_check.columns[
    df_to_check.columns.str.startswith("suggestions_")
].tolist()

# Performance metrics for one suggester and prefix length
for _, suggester in enumerate(suggestions_cols_to_compare):
    metrics = compute_performance_metrics_from_suggestions(
        df_to_check,
        correct_code_col=correct_code_col,
        suggestions_col=suggester,
        code_length=SIC_CODE_LENGTH,
        k_values=NUM_CHARACTERS_LIST,
        ave_time_per_query=time_to_check.get(suggester.removeprefix("suggestions_"), 0),
    )
    # print(metrics.report_metrics())

# %%
# Comparison of performance metrics for the different suggesters and prefix lengths
ave_elapsed_per_row_list = [
    time_to_check.get(col.removeprefix("suggestions_"), 0)
    for col in suggestions_cols_to_compare
]

compare_performance_metrics = build_sayt_metrics_comparison_table(
    df_to_check,
    suggestions_cols_to_compare=suggestions_cols_to_compare,
    correct_code_col=correct_code_col,
    k_values=NUM_CHARACTERS_LIST,
    ave_time_per_query_list=ave_elapsed_per_row_list,
)

compare_performance_metrics.head()
