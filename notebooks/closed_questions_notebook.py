# %%
# pylint: disable=C0103, C0116, C0301, C0114

import re

# %%
import dotenv
import pandas as pd

# %%
evaluation_bucket = dotenv.get_key(".env", "EVALUATION_BUCKET")
analysis_bucket = dotenv.get_key(".env", "ANALYSIS_BUCKET")
if not evaluation_bucket:
    raise ValueError("EVALUATION_BUCKET not found in .env file. Please set it.")
if not analysis_bucket:
    raise ValueError("ANALYSIS_BUCKET not found in .env file. Please set it.")


# %%
data = pd.read_parquet(f"{analysis_bucket}evaluation_df_with_sa_clean_codes.parquet")

# %%
column = "survey_assist_closed_question_option_"
selected_response = []

# iterate through all rows of the data
for row in range(len(data)):
    # save the response selected by the user
    response = data["survey_assist_closed_question_response"][row]

    # check only when closed question was asked
    if response is not None and response != "none of the above":

        # change options from closed questions to lower case (matching the selected response)
        for i in range(1, 7):
            column_name = column + str(i)
            if data[column_name][row] is not None:
                data.loc[row, column_name] = data[column_name][row].lower()

        # find the order of the selected response
        j = 1
        while j < 7:  # noqa: PLR2004
            column_name = column + str(j)
            if response == data[column_name][row]:
                selected_response.append(j)
                j = 7
            j += 1

# %%
# percentage of codes found using closed quesiton
print(round(100 * len(selected_response) / len(data), 2))

# %%
selected_response_order = {}
for i in range(1, 7):
    picked_response = "option_" + str(i)
    selected_response_order[picked_response] = selected_response.count(i)

# %%
# order, which answer was seleceted
print(selected_response_order)
# order of the question selected
print(
    "1st option [%]:",
    round(100 * selected_response_order["option_1"] / len(selected_response), 2),
)
print(
    "2nd option [%]:",
    round(100 * selected_response_order["option_2"] / len(selected_response), 2),
)
print(
    "3rd option [%]:",
    round(100 * selected_response_order["option_3"] / len(selected_response), 2),
)
print(
    "4th option [%]:",
    round(100 * selected_response_order["option_4"] / len(selected_response), 2),
)
print(
    "5th option [%]:",
    round(100 * selected_response_order["option_5"] / len(selected_response), 2),
)
print(
    "6th option [%]:",
    round(100 * selected_response_order["option_6"] / len(selected_response), 2),
)

# %% [markdown]
# Note, "none of the above" is always presented at the bottom of the list, so 6th option never brings a code. Options 1 and 2 will always have possible codes.

# %%
sic_rephrased = pd.read_csv(
    f"{evaluation_bucket}sic_rephrased_descriptions_2025_02_03.csv"
)


# %%
def convert_to_dict(dict_string):
    clean_string = dict_string.strip().strip("'")

    code = re.search(r"{Code: \s*(.*?),\s*Title: ", clean_string, re.DOTALL)
    code_value = code.group(1).strip() if code else ""

    title = re.search(
        r"Title: \s*(.*?),\s*Example activities: ", clean_string, re.DOTALL
    )
    title_value = title.group(1).strip() if title else ""
    title_value = title_value.lower()

    activities = re.search(r"Example activities: \s*(.*?)\s*}", clean_string, re.DOTALL)
    activities_value = activities.group(1).strip() if activities else ""

    result_dictionary = {
        "Code": code_value,
        "Title": title_value,
        "Example activities": activities_value,
    }
    return result_dictionary


# %%
sic_rephrased.sample()

# %%
sic_rephrased["reviewed_description"] = sic_rephrased[
    "reviewed_description"
].str.lower()
sic_rephrased["llm_rephrased_description"] = sic_rephrased[
    "llm_rephrased_description"
].str.lower()


# %%
def get_code_by_title(dictionary, title):
    for item in dictionary:
        if item.get("Title") == title:
            return item.get("Code")
    return "XXXXX"


# %%
sic_dictionary = sic_rephrased["input_description"].apply(convert_to_dict)
response_codes = []
none_of_the_above = 0

for row in range(len(data)):
    # save the response selected by the user
    if data["survey_assist_closed_question_response"][row] is not None:
        response = data["survey_assist_closed_question_response"][row].lower()

        if response == "none of the above":
            none_of_the_above += 1
            response_codes.append("None")

        elif response in list(sic_rephrased["reviewed_description"]):
            sic_code = sic_rephrased[sic_rephrased["reviewed_description"] == response][
                "input_code"
            ].item()
            response_codes.append(str(sic_code))

        else:
            sic_code = get_code_by_title(sic_dictionary, response)
            response_codes.append(str(sic_code))

    else:
        response_codes.append("None")

# %%
# we get "None" when the closed question was not asked or the answer was "none of the above"
response_codes.count("None")

# %%
response_codes.count("XXXXX")

# %%
len(response_codes)

# %%
# asked closed question, but didn't get a code
round(100 * none_of_the_above / len(selected_response), 2)

# %%
data["survey_assist_closed_question_response_code"] = response_codes

# %%
closed_q_data = data[
    [
        "unique_id",
        "user",
        "job_title",
        "job_description",
        "org_description",
        "survey_assist_alt_candidate_code_1",
        "survey_assist_alt_candidate_code_2",
        "survey_assist_alt_candidate_code_3",
        "survey_assist_alt_candidate_code_4",
        "survey_assist_alt_candidate_code_5",
        "survey_assist_closed_question_response",
        "survey_assist_closed_question_option_1",
        "survey_assist_closed_question_option_2",
        "survey_assist_closed_question_option_3",
        "survey_assist_closed_question_option_4",
        "survey_assist_closed_question_option_5",
        "survey_assist_closed_question_option_6",
        "survey_assist_closed_question_response_code",
    ]
]

# %%
sic_dictionary = sic_rephrased["input_description"].apply(convert_to_dict)

options: dict[str, list[str | None]] = {"1": [], "2": [], "3": [], "4": [], "5": []}

for row in range(len(closed_q_data)):
    # if the closed question response is None, it means the question was not asked, and there's no codes.
    if closed_q_data["survey_assist_closed_question_response"][row] is not None:

        i = 1
        while i < 6:  # noqa: PLR2004
            current_row = closed_q_data[f"survey_assist_closed_question_option_{i}"][
                row
            ].lower()
            if current_row == "none of the above":
                while i < 6:  # noqa: PLR2004
                    options[f"{i}"].append(None)
                    i += 1

            else:
                if current_row in list(sic_rephrased["reviewed_description"]):
                    sic_code = sic_rephrased[
                        sic_rephrased["reviewed_description"] == current_row
                    ]["input_code"].item()
                    options[f"{i}"].append(str(sic_code))

                else:
                    sic_code = get_code_by_title(sic_dictionary, current_row)
                    options[f"{i}"].append(str(sic_code))

                i += 1
    else:
        options["1"].append(None)
        options["2"].append(None)
        options["3"].append(None)
        options["4"].append(None)
        options["5"].append(None)


# %%
for k in range(1, 6):
    closed_q_data[f"survey_assist_closed_question_option_{k}_code"] = options[f"{k}"]

# %%
closed_q_data.to_parquet(
    f"{analysis_bucket}closed_questions/closed_questions_codes.parquet", index=False
)
