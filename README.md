# Survey Assist Evaluation

Evaluation utilities used as part of Survey Assist

## Overview

Survey Assist evaluation functions. This repository contains utilities for evaluating the performance of Survey Assist's Large Language Models (LLMs) in classifying Standard Industrial Classification (SIC) codes. The evaluation framework includes tools for batch processing of datasets, as well as a comprehensive suite of metrics to analyze and compare LLM performance against human coders.

## Features

* **Batch Processing:** Send large datasets to the API for SIC classification.
* **Performance Evaluation:** A comprehensive suite of metrics to analyze and compare LLM performance against human coders.

## Local Development & Setup

The Makefile defines a set of commonly used commands and workflows.  Where possible use the files defined in the Makefile.

### Prerequisites

Ensure you have the following installed on your local machine:

* Python 3.12 (Recommended: use `pyenv` to manage versions)
* `poetry` (for dependency management)
* Google Cloud SDK (`gcloud`) with appropriate permissions
* Colima (if running locally with containers)
* Terraform (for infrastructure management)

### Setup Instructions

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/ONSdigital/survey-assist-eval.git](https://github.com/ONSdigital/survey-assist-eval.git)
    cd survey-assist-eval
    ```

2. **Create and activate a virtual environment**

    Using `pyenv` and `pyenv-virtualenv`:

    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    poetry install
    ```

4. **Generate an API Token**

    The API uses Application Default Credentials to generate and authenticate tokens.

    Ensure GOOGLE_APPLICATION_CREDENTIALS are not set in your environment.

    ```bash
    unset GOODLE_APPLICATION_CREDENTIALS
    ```

    Login to gcloud application default:

    ```bash
    gcloud auth application-default login
    ```

    Set to the correct GCP project:

    ```bash
    gcloud auth application-default set-quota-project GCP-PROJECT-NAME
    ```

    Check the project setting:

    ```bash
    cat ~/.config/gcloud/application_default_credentials.json
    ```

    Set the required environment variables:

    ```bash
    export SA_EMAIL="SERVICE-ACCOUNT-FOR-API-ACCESS"
    export API_GATEWAY="API GATEWAY URL NOT INC https://"
    ```

    Then, run the make command to use default expiry (1h):

    ```bash
    make generate-api-token
    ```

    You can run from cli and pass in a chosen expiry time:

    ```bash
    poetry run generate-api-token -e 7200
    ```

## Code Quality & Testing

### Code Quality

Code quality and static analysis are enforced using `isort`, `black`, `ruff`, `mypy`, `pylint`, and `bandit`.

* **To check for errors without auto-fixing:**
    ```bash
    make check-python-nofix
    ```
* **To check and automatically fix errors:**
    ```bash
    make check-python
    ```

### Testing

Pytest is used for testing.

* **To run unit tests:**
    ```bash
    make unit-tests
    ```
* **To run all tests:**
    ```bash
    make all-tests
    ```

### Pre-commit Hooks

Pre-commit hooks are set up to run code quality checks before each commit. They will call `make check-python` under the hood as well.
To install the hooks, run:

```bash
pre-commit install
```
