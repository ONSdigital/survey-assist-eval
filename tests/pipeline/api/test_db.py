"""Unit tests for ApiEvaluator Firestore DB utils."""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from survey_assist_eval.pipeline.api import db as db_module

# turning off black for this file: set max line length to match PEP8 (79 chars)
# for improved readability
# fmt: off


@contextmanager
def firestore_init_mocks(exists: bool = True):
    """Initialise and handle mocks for firestore setup."""
    # use exit stack to ensure cleanup post test
    with ExitStack() as stack:
        mock_logger = MagicMock()
        mock_app = object()
        mock_db = MagicMock()
        mock_job_doc = MagicMock()

        mock_get_logger = stack.enter_context(
            patch(
                "survey_assist_eval.pipeline.api.db.get_logger",
                return_value=mock_logger,
            )
        )
        mock_init_app = stack.enter_context(
            patch(
                "survey_assist_eval.pipeline.api.db.initialize_app",
                return_value=mock_app,
            )
        )
        mock_client = stack.enter_context(
            patch(
                "survey_assist_eval.pipeline.api.db.firestore.client",
                return_value=mock_db,
            )
        )

        mock_db.collection.return_value.document.return_value = mock_job_doc
        mock_job_doc.get.return_value = SimpleNamespace(exists=exists)

        yield {
            "logger": mock_logger,
            "app": mock_app,
            "db": mock_db,
            "job_doc": mock_job_doc,
            "get_logger": mock_get_logger,
            "init_app": mock_init_app,
            "client": mock_client,
        }


@contextmanager
def firestore_eval_results_mocks():
    """Initialise and handle mocks for firestore evaluation results."""
    with ExitStack() as stack:
        mock_logger = MagicMock()
        mock_job_doc = MagicMock()
        mock_job_doc.get.return_value = SimpleNamespace(exists=True)
        mock_job_doc.set.return_value = None

        mock_get_logger = stack.enter_context(
            patch(
                "survey_assist_eval.pipeline.api.db.get_logger",
                return_value=mock_logger,
            )
        )

        yield {
            "logger": mock_logger,
            "job_doc": mock_job_doc,
            "get_logger": mock_get_logger,
        }


class TestInitialiseFirestore:
    """Unit tests for the initialise_firestore function."""
    def test_initialise_firestore_success(self):
        """Ensure job doc returned when no existing doc is found."""
        with firestore_init_mocks(exists=False) as mock:
            job_doc = db_module.initialise_firestore(
                gcp_project_id="test_project",
                firestore_db_id="test_db",
                firestore_collection_id="test_collection",
                job_id="test_job",
                log_level="DEBUG",
            )

        assert job_doc is mock["job_doc"], (
            "Expected job_doc to be the mock job_doc returned from firestore "
            "client."
        )
        mock["get_logger"].assert_called_once_with(
            db_module.__name__, level="DEBUG"
        )
        mock["init_app"].assert_called_once_with(
            options={"projectId": "test_project"}
        )
        mock["client"].assert_called_once_with(
            mock["app"], database_id="test_db"
        )
        mock["db"].collection.assert_called_once_with("test_collection")
        mock["db"].collection.return_value.document.assert_called_once_with(
            "test_job"
        )

    def test_initialise_firestore_existing_doc(self):
        """Ensure job doc returned when existing doc is found."""
        with firestore_init_mocks(exists=True) as mock, pytest.raises(
            ValueError,
            match="Document with job_id test_job already exists.",
        ):
            db_module.initialise_firestore(
                gcp_project_id="test_project",
                firestore_db_id="test_db",
                firestore_collection_id="test_collection",
                job_id="test_job",
                log_level="DEBUG",
            )

        mock["get_logger"].assert_called_once_with(
            db_module.__name__, level="DEBUG"
        )
        mock["init_app"].assert_called_once_with(
            options={"projectId": "test_project"}
        )
        mock["client"].assert_called_once_with(
            mock["app"], database_id="test_db"
        )
        mock["db"].collection.assert_called_once_with("test_collection")
        mock["db"].collection.return_value.document.assert_called_once_with(
            "test_job"
        )
        mock["logger"].error.assert_called_once()


# keeping a single test within class to prioritise consistent naming/structure
# pylint: disable=too-few-public-methods
class TestEvalResultsToFirestore:
    """Unit tests for the eval_results_to_firestore function."""
    def test_eval_results_to_firestore(self):
        """Ensure results are written to firestore successfully."""
        with firestore_eval_results_mocks() as mock:
            db_module.eval_results_to_firestore(
                job_doc=mock["job_doc"],
                job_id="test_job",
                results={"test": "value"},
                log_level="DEBUG",
            )

        mock["get_logger"].assert_called_once_with(
            db_module.__name__, level="DEBUG"
        )
        # protected member required here for test - ok for this usecase only
        mock["job_doc"].set.assert_called_once_with(
            {"test": "value"}, retry=db_module._retry  # pylint: disable=W0212
        )
