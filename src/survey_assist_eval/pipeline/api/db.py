"""API evaluation database utilities."""

# turning off black for this file: set max line length to match PEP8 (79 chars)
# for improved readability
# fmt: off

from typing import cast

from firebase_admin import firestore, initialize_app
from google.api_core.exceptions import ServiceUnavailable
from google.api_core.retry import Retry
from google.cloud.firestore import CollectionReference, DocumentSnapshot
from survey_assist_utils.logging import get_logger

# firestore client retry config to handle transient errors
_retry = Retry(
    predicate=lambda exc: isinstance(exc, ServiceUnavailable),
    initial=0.5,
    maximum=10.0,
    multiplier=1.5,
    deadline=30.0,
)


def initialise_firestore(
    gcp_project_id: str,
    firestore_db_id: str,
    firestore_collection_id: str,
    job_id: str,
    log_level: str,
) -> CollectionReference:
    """Initialise firestore db and return collection reference.

    Also checks to ensure the results doc will be unique for a given job_id
    to avoid overwriting existing results.
    """
    logger = get_logger(__name__, level=log_level)
    app_options = {"projectId": gcp_project_id}
    logger.debug("Initialising firestore client...")
    app = initialize_app(options=app_options)
    db = firestore.client(app, database_id=firestore_db_id)
    logger.debug("Firestore client initialised.")
    col_ref = db.collection(firestore_collection_id)

    # check if document with job_id already exists, if so raise error to avoid
    # overwriting existing results. Casting to DocumentSnapshot to satify type
    # checker (getting confused by sync call i.e. not async)
    job_doc = col_ref.document(job_id).get(retry=_retry)
    job_exists = cast(DocumentSnapshot, job_doc).exists
    if job_exists:
        logger.error(f"Document with job_id {job_id} already exists.")
        raise ValueError(f"Document with job_id {job_id} already exists.")
    logger.debug(
        f"Document with job_id {job_id} does not exist, safe to proceed."
    )

    return col_ref
