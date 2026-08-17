# %%
"""Run small tests for industry descriptions SAYT.

Expects following environment variables to be set:
- EVALUATION_BUCKET_NAME: name of GCS bucket where the data is stored
The variables are loaded from the ".env" file.
"""

# pylint: disable=invalid-name, duplicate-code

# %%
import os

import numpy as np
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

# %%
SIC_CODE_LENGTH = 5

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

    suggestions_df = get_suggestions_by_chars(
        df_copy,
        suggesters_dict=suggesters_dict,
        num_chars=num_chars,
        suggestions_limit=suggestions_limit,
    )[0]

    melt = melt_results_for_analysis(df=suggestions_df)

    fig = create_figure(melt, output_dir=output_dir)

    return melt, fig


# %%
def suggester_analysis_table(
    df: pd.DataFrame,
    suggester_name: str | None = None,
    num_chars: int | None = None,
    max_suggestions: int = 9,
):
    """Allows suggesters analysis in a table format.

    Args:
        df (pd.DataFrame): dataframe with suggestions from suggester.
            Requires columns: "suggester", "num_chars", "rank".
        suggester_name (str): suggester name to be analysed.
        num_chars (int): number of characters to be checked.
        max_suggestions: the maximum rank of suggestions considered as valid.

    Return:
        filtered dataframe with specified suggester name and/or number of characters.
    """
    # replace ranks not shown with "NA"
    df_copy = df.copy()
    df_copy.loc[df_copy["rank"] > max_suggestions, "rank"] = np.nan

    results = df.groupby(["suggester", "num_chars", "rank"]).count()

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
melt_df_simple, fig_simple = run_eval_for_suggesters(
    test_df, suggesters_simple, range(4, 10), output_dir=OUTPUT_DIR
)

# %%
suggester_analysis_table(melt_df_simple)

# %%
suggester_analysis_table(melt_df_simple, num_chars=6)

# %%
melt_df_pairs, fig_pairs = run_eval_for_suggesters(
    test_df,
    suggesters_pairs,
    (x for x in range(4, 10)),
    output_dir=f"{OUTPUT_DIR}_pairs",
)

# %%
suggester_analysis_table(melt_df_pairs)

# %%
suggester_analysis_table(melt_df_pairs, num_chars=6)

# %%
melt_df_three, fig_three = run_eval_for_suggesters(
    test_df,
    suggesters_three,
    (x for x in range(4, 10)),
    output_dir=f"{OUTPUT_DIR}_three",
)

# %%
melt_df_all = (
    pd.concat([melt_df_simple, melt_df_pairs, melt_df_three], ignore_index=True)
    .drop_duplicates()
    .copy()
)

# %%
fig_all = create_figure(melt_df_all, output_dir=OUTPUT_DIR)
