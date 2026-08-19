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
CORRECT_CODE_COL = "correct_sic_code"

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

suggesters_one = {
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

suggesters_two = {
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

# suggesters_one = {
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

# suggesters_two = {
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
def run_eval_for_suggesters(  # noqa: PLR0913 pylint: disable=R0913, R0914
    *,
    df: pd.DataFrame,
    correct_code_col: str,
    suggesters_dict: dict,
    num_chars: list[int],
    suggestions_limit: int = 9,
    code_length: int = 5,
    output_dir: str = OUTPUT_DIR,
):
    """Use functions necessary to create a dataframe that allows for grouping by
        rank and suggester type. Create plots.

    Args:
        df (pd.DataFrame): dataframe to be tested.
        correct_code_col (str): name of the column containing correct codes.
        suggesters_dict (dict): a dictionary with suggester models.
        num_chars (list): number of characters to be tested.
        suggestions_limit: the maximum rank of suggestions considered as valid.
        code_length (int): expected SIC/SOC code length.
        output_dir (str): path to file location to be saved.

    Return:
        pd.DataFrame: dataframe with results from suggesters, split by the type of suggester
            and number of characters.
        pd.DataFrame: dataframe containing suggestions.
        dict: dictionary containing average running time for each suggester.
        dict: dictionary containing performance metrics for each suggester.
        figure: plot showing distribution of ranks by suggestions and number of characters.
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

    suggestions_cols_to_compare = suggestions_df.columns[
        suggestions_df.columns.str.startswith("suggestions_")
    ].tolist()

    metrics_dict = {}

    # Performance metrics for one suggester and prefix length
    for _, suggester in enumerate(suggestions_cols_to_compare):
        metrics = compute_performance_metrics_from_suggestions(
            suggestions_df,
            correct_code_col=correct_code_col,
            suggestions_col=suggester,
            code_length=code_length,
            k_values=num_chars,
            ave_time_per_query=avg_ms_dict.get(
                suggester.removeprefix("suggestions_"), 0
            ),
        )
        metrics_dict[suggester] = metrics.report_metrics()

    return suggestions_df, avg_ms_dict, metrics_dict, fig


# %%
suggestions_df_one, avg_ms_dict_one, metrics_one, fig_one = run_eval_for_suggesters(
    df=test_df,
    suggesters_dict=suggesters_one,
    num_chars=NUM_CHARACTERS_LIST,
    suggestions_limit=MAX_SUGGESTIONS,
    code_length=SIC_CODE_LENGTH,
    correct_code_col=CORRECT_CODE_COL,
    output_dir=OUTPUT_DIR,
)

# %%
suggestions_df_two, avg_ms_dict_two, metrics_two, fig_two = run_eval_for_suggesters(
    df=test_df,
    suggesters_dict=suggesters_two,
    num_chars=NUM_CHARACTERS_LIST,
    suggestions_limit=MAX_SUGGESTIONS,
    code_length=SIC_CODE_LENGTH,
    correct_code_col=CORRECT_CODE_COL,
    output_dir=f"{OUTPUT_DIR}_two",
)

# %%
suggestions_df_three, avg_ms_dict_three, metrics_three, fig_three = (
    run_eval_for_suggesters(
        df=test_df,
        suggesters_dict=suggesters_three,
        num_chars=NUM_CHARACTERS_LIST,
        suggestions_limit=MAX_SUGGESTIONS,
        code_length=SIC_CODE_LENGTH,
        correct_code_col=CORRECT_CODE_COL,
        output_dir=f"{OUTPUT_DIR}_three",
    )
)

# %%
suggesters_all = {**suggesters_one, **suggesters_two, **suggesters_three}

# %%
suggestions_df_all, avg_ms_dict_all, metrics_all, fig_all = run_eval_for_suggesters(
    df=test_df,
    suggesters_dict=suggesters_all,
    num_chars=NUM_CHARACTERS_LIST,
    suggestions_limit=MAX_SUGGESTIONS,
    code_length=SIC_CODE_LENGTH,
    correct_code_col=CORRECT_CODE_COL,
    output_dir=f"{OUTPUT_DIR}_all",
)


# %%
def get_performance_metrics_table(
    df: pd.DataFrame,
    avg_time_dict: dict,
    correct_code_col: str,
    num_chars: list[int],
):
    """Comparison of performance metrics for the different suggesters and prefix lengths.

    Args:
        df (pd.DataFrame): dataframe containing suggestions.
        avg_time_dict (dict): dictionary containing average time for each suggester.
        correct_code_col (str): name of the column containing correct codes.
        num_chars (list): number of characters to be tested.

    Return:
        pd.DataFrame: dataframe with performance metrics.
    """
    suggestions_cols_to_compare = df.columns[
        df.columns.str.startswith("suggestions_")
    ].tolist()

    ave_elapsed_per_row_list = [
        avg_time_dict.get(col.removeprefix("suggestions_"), 0)
        for col in suggestions_cols_to_compare
    ]

    compare_performance_metrics = build_sayt_metrics_comparison_table(
        df,
        suggestions_cols_to_compare=suggestions_cols_to_compare,
        correct_code_col=correct_code_col,
        k_values=num_chars,
        ave_time_per_query_list=ave_elapsed_per_row_list,
    )

    return compare_performance_metrics


# %%
metrics_table_one = get_performance_metrics_table(
    suggestions_df_one, avg_ms_dict_one, CORRECT_CODE_COL, NUM_CHARACTERS_LIST
).head()

# %%
metrics_table_two = get_performance_metrics_table(
    suggestions_df_two, avg_ms_dict_two, CORRECT_CODE_COL, NUM_CHARACTERS_LIST
).head()

# %%
metrics_table_three = get_performance_metrics_table(
    suggestions_df_three, avg_ms_dict_three, CORRECT_CODE_COL, NUM_CHARACTERS_LIST
).head()


# %%
metrics_table_all = get_performance_metrics_table(
    suggestions_df_all, avg_ms_dict_all, CORRECT_CODE_COL, NUM_CHARACTERS_LIST
)

# %%
metrics_table_one.head()

# %%
metrics_table_all.head()
