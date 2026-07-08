"""API evaluation database utilities."""

# turning off black for this file: set max line length to match PEP8 (79 chars)
# for improved readability
# fmt: off

from typing import cast

from firebase_admin import firestore, initialize_app
from google.api_core.exceptions import ServiceUnavailable
from google.api_core.retry import Retry
from google.cloud.firestore import DocumentSnapshot
from google.cloud.firestore_v1.base_document import BaseDocumentReference
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
) -> BaseDocumentReference:
    """Initialise firestore db and return collection reference.

    Also checks to ensure the results doc will be unique for a given job_id
    to avoid overwriting existing results.

    Args:
        gcp_project_id: GCP project ID for firestore client initialisation.
        firestore_db_id: Firestore database ID for client initialisation.
        firestore_collection_id: Firestore collection ID for storing evaluation
            results.
        job_id: Unique identifier for the evaluation job, used to ensure
            results
        log_level: Log level for logging within this function.

    Returns:
        BaseDocumentReference: Reference to the firestore document where
            evaluation results will be stored.
    """
    logger = get_logger(__name__, level=log_level)
    app_options = {"projectId": gcp_project_id}
    logger.debug("Initialising firestore client...")
    app = initialize_app(options=app_options)
    db = firestore.client(app, database_id=firestore_db_id)
    logger.debug("Firestore client initialised.")
    job_doc = db.collection(firestore_collection_id).document(job_id)

    # check if document with job_id already exists, if so raise error to avoid
    # overwriting existing results. Casting to DocumentSnapshot to satify type
    # checker (getting confused by sync call i.e. not async)
    job_exists = cast(DocumentSnapshot, job_doc.get(retry=_retry)).exists
    if job_exists:
        logger.error(f"Document with job_id {job_id} already exists.")
        raise ValueError(f"Document with job_id {job_id} already exists.")
    logger.debug(
        f"Document with job_id {job_id} does not exist, safe to proceed."
    )

    return job_doc


def eval_results_to_firestore(
    job_doc: BaseDocumentReference,
    job_id: str,
    results: dict,
    log_level: str,
) -> None:
    """Write evaluation results to firestore.

    Args:
        job_doc: Reference to the firestore document where evaluation results
            will be stored.
        job_id: Unique identifier for the evaluation job, used for logging
            only.
        results: Evaluation results to be written to firestore.
        log_level: Log level for logging within this function.
    """
    logger = get_logger(__name__, level=log_level)
    logger.debug(
        f"Writing evaluation results to firestore for job_id {job_id}..."
    )
    job_doc.set(results, retry=_retry)
    logger.debug(
        f"Evaluation results for job_id {job_id} written to firestore."
    )
