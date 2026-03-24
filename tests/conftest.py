"""This module contains pytest configuration, hooks and shared fixtures.

It sets up a global logger and defines hooks for pytest to log events
such as the start and finish of a test session.

Functions:
    pytest_configure(config): Applies global test configuration.
    pytest_sessionstart(session): Logs the start of a test session.
    pytest_sessionfinish(session, exitstatus): Logs the end of a test session.
This module contains pytest configuration

Fixtures defined in this file are automatically discovered by pytest and can be
used in any test file within this directory without needing to be imported.

Fixtures:
    raw_data_and_config: Provides a consistent, raw DataFrame and ColumnConfig
                         object for use across multiple test suites.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

# Add src directory to Python path
SRC_PATH = str(Path(__file__).parent.parent / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Configure a global logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Adjust level as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)
)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):  # pylint: disable=unused-argument
    """Pytest hook implementation that is executed at the start of a test session.

    This function logs a message indicating that the test session has started.

    Args:
        session: The pytest session object (not used in this implementation).
    """
    logger.info("=== Test Session Started ===")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):  # pylint: disable=unused-argument
    """Hook function called after the test session ends.

    This function is executed after all tests have been run and the test session
    is about to finish. It can be used to perform cleanup or logging tasks.

    Args:
        session (Session): The pytest session object containing information
            about the test session.
        exitstatus (int): The exit status code of the test session. This indicates
            whether the tests passed, failed, or were interrupted.

    Note:
        The `pylint: disable=unused-argument` directive is used to suppress
        warnings for unused arguments in this function.
    """
    logger.info("=== Test Session Finished ===")
