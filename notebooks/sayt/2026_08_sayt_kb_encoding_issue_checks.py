"""A script to check for encoding issues in the SAYT knowledge base."""

# pylint: disable=invalid-name
# %%
import os
import unicodedata

import chardet
import pandas as pd
from dotenv import load_dotenv
from ftfy import fix_encoding
from google.cloud import storage
from survey_assist_utils.logging import get_logger

load_dotenv()
bucket_name = os.getenv("EVALUATION_BUCKET_NAME")
if not bucket_name:
    raise ValueError("EVALUATION_BUCKET_NAME environment variable not set")


logger = get_logger(__name__)


# %%
def detect_encoding_bucket(gcp_bucket, file_path):
    """Detect the encoding of a file stored in Google Cloud Storage.

    Args:
        gcp_bucket (str): The name of the GCS bucket.
        file_path (str): The path to the file within the bucket.

    Returns:
        tuple: A tuple of (encoding, confidence) where encoding is a lowercase
               string representing the detected encoding and confidence is a
               float between 0 and 1 indicating the confidence of the detection.
    """
    client = storage.Client()
    bucket = client.bucket(gcp_bucket)
    blob = bucket.blob(file_path)
    data = blob.download_as_bytes()
    result = chardet.detect(data)
    return result["encoding"].lower(), result["confidence"]


def detect_encoding_local(file):
    """Detect the encoding of a local file.

    Args:
        file (str): The path to the local file.

    Returns:
        tuple: A tuple of (encoding, confidence) where encoding is a lowercase
               string representing the detected encoding and confidence is a
               float between 0 and 1 indicating the confidence of the detection.
    """
    with open(file, "rb") as f:
        data = f.read()
        result = chardet.detect(data)
    return result["encoding"].lower(), result["confidence"]


# %%
def has_accent(text):
    """Check if a string contains any accented characters.

    Detects accented characters by normalizing to NFD form and checking for
    combining marks (unicode category Mn).

    Args:
        text (str): The string to check for accented characters.

    Returns:
        bool: True if the string contains any accented characters, False otherwise.
              Also returns False if the input is not a string.
    """
    if not isinstance(text, str):
        return False
    for char in text:
        # Normalize to NFD form to separate base letters and accents
        decomposed = unicodedata.normalize("NFD", char)
        # Check if any combining mark (Mn) exists
        if any(unicodedata.category(c) == "Mn" for c in decomposed):
            return True
    return False


# %%
sic_kb_for_sayt = pd.read_csv(
    f"gs://{bucket_name}/sic_knowledgebase/sic_kb_for_sayt.csv", dtype=str
)

encoding, confidence = detect_encoding_bucket(
    bucket_name, "sic_knowledgebase/sic_kb_for_sayt.csv"
)
print(f"\nEncoding: {encoding}, Confidence: {confidence}")

sic_kb_for_sayt["display_has_accent"] = sic_kb_for_sayt["display_text"].apply(
    has_accent
)
print(sic_kb_for_sayt[sic_kb_for_sayt["display_has_accent"]].sum())

sic_kb_for_sayt["search_has_accent"] = sic_kb_for_sayt["search_text"].apply(has_accent)
print(sic_kb_for_sayt[sic_kb_for_sayt["search_has_accent"]].sum())

print(
    sic_kb_for_sayt[sic_kb_for_sayt["display_has_accent"]][
        ["code", "display_text"]
    ].drop_duplicates()
)
print(
    sic_kb_for_sayt[sic_kb_for_sayt["search_has_accent"]][
        ["code", "search_text"]
    ].drop_duplicates()
)
# %%
sic_kb_for_classifai = pd.read_csv(
    f"gs://{bucket_name}/sic_knowledgebase/sic_kb_for_classifai.csv", dtype=str
).rename(columns={"text": "search_text", "label": "code"})


encoding, confidence = detect_encoding_bucket(
    bucket_name, "sic_knowledgebase/sic_kb_for_classifai.csv"
)
print(f"\nEncoding: {encoding}, Confidence: {confidence}")

sic_kb_for_classifai["search_has_accent"] = sic_kb_for_classifai["search_text"].apply(
    has_accent
)
print(sic_kb_for_classifai[sic_kb_for_classifai["search_has_accent"]].sum())
print(sic_kb_for_classifai[sic_kb_for_classifai["search_has_accent"]])
print(
    sic_kb_for_classifai[sic_kb_for_classifai["search_has_accent"]][
        ["code", "search_text"]
    ].drop_duplicates()
)

check = sic_kb_for_classifai[sic_kb_for_classifai["search_has_accent"]][
    ["code", "search_text"]
].drop_duplicates()

check["encoding_fix"] = check["search_text"].apply(fix_encoding)


def _remove_accents(text):
    # Normalize the string to decompose characters with accents
    normalized = unicodedata.normalize("NFD", text)
    # Filter out combining marks (category 'Mn')
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


# %%
check["encoding_fix"] = check["search_text"].apply(fix_encoding)
check["encoding_fix_and_no_accents"] = check["encoding_fix"].apply(_remove_accents)
# %%


def _fix_bom(text):
    if isinstance(text, str):
        text = text.encode("utf-8")
    return text.decode("utf-8-sig")


check["bom_fix"] = check["encoding_fix_and_no_accents"].apply(_fix_bom)
check["bom_fix_only"] = check["search_text"].apply(_fix_bom)


# Output: 'first line'
# %%


def _fix_encoding2(text):
    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


check["encoding_fix2"] = check["search_text"].apply(_fix_encoding2)
check["no_accents2"] = check["encoding_fix2"].apply(_remove_accents)
# %%
