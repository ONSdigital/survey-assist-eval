"""Compare automated open question evaluation metrics with the manual evaluation
conducted by Social Surveys in November 2026.
"""

# pylint: disable=C0103
# pylint: disable=duplicate-code
# %%

import os

import pandas as pd

from survey_assist_eval.evaluation.open_questions.open_questions_evaluation import (
    evaluate_open_questions,
)

# %%
MAX_WORD_COUNT_THRESHOLD = 15
MAX_NUM_SENTENCE_THRESHOLD = 1
MAX_WORD_COUNT_PER_SENTENCE_THRESHOLD = 12
MIN_WORD_COUNT_THRESHOLD = 3
MAX_SYLLABLES_THRESHOLD = 4

SOURCE_PATH = os.getenv(
    "MANUAL_EVAL_XLSX_PATH",
    "data/open_question_eval/Nov_Test_2025/Survey_Assist_Qu_Eval_V2.xlsx",
)

MANUAL_TEXT_COLUMN = "survey_assist_open_question"

EVAL_COLUMNS = [
    "Simple Language",
    "Easily Reworded",
    "Relevant",
    "Long Sentences",
    "Missing Key Context",
    "Excessive punctuation",
    "Complex Task",
    "Leading Questions",
    "Double Barrelled Qs",
    "Overall RAG Status",
]

SECTION_TO_MANUAL_COLUMNS = {
    "text_statistics": ["Long Sentences", "Excessive punctuation"],
    "simple_language": ["Simple Language", "Easily Reworded"],
    "question_structure": ["Double Barrelled Qs"],
    "other": [
        "Relevant",
        "Missing Key Context",
        "Complex Task",
        "Overall RAG Status",
        "Leading Questions",
    ],
}

# %%
df = pd.read_excel(
    SOURCE_PATH,
    sheet_name="Sheet1",
)

# %%
text_statistics_config = {
    "word_threshold": MAX_WORD_COUNT_THRESHOLD,
    "sentence_threshold": MAX_NUM_SENTENCE_THRESHOLD,
    "long_sentence_word_threshold": MAX_WORD_COUNT_PER_SENTENCE_THRESHOLD,
    "short_text_word_threshold": MIN_WORD_COUNT_THRESHOLD,
}

simple_language_config = {
    "syllables_threshold": MAX_SYLLABLES_THRESHOLD,
}
# %%
open_question_metrics = evaluate_open_questions(
    df,
    text_column=MANUAL_TEXT_COLUMN,
    text_statistics_config=text_statistics_config,
    simple_language_config=simple_language_config,
)


# %%
summary = (
    pd.concat(
        {col: df[col].value_counts(dropna=False) for col in EVAL_COLUMNS},
        names=["metric", "value"],
    )
    .rename("count")
    .reset_index()
)
summary["pct"] = (summary["count"] / len(df) * 100).round(1)
summary = summary.sort_values(["metric", "count"], ascending=[True, False])


# %%
section_reports = open_question_metrics.report_metrics_by_section()
for section, manual_columns in SECTION_TO_MANUAL_COLUMNS.items():
    if section in section_reports:
        print(section_reports[section])

    section_summary = summary[summary["metric"].isin(manual_columns)]
    for metric, metric_group in section_summary.groupby("metric", sort=False):
        print(f"\n{metric}")
        print(metric_group[["value", "count", "pct"]].to_string(index=False))
# %%
