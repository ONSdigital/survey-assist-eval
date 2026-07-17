# API Evaluation

This directory contains source code for an automated Survey Assist API
evaluation pipeline. It's objective is to exercise the API using a known
dataset to support development, verify performance, and test for model drift.

This pipeline performs the following tasks:
- Evaluation of SIC and SOC Survey Assist API workflows.
- Reads and preprocesses SIC/SOC test input datasets.
- Collects and records the API configuration (log state of system under test).
- Passes test data via direct lookup and (if requrired) classify endpoints.
- Evaluates API responses against clearically coded test input data.
- Stores evaluation run in a firestore database for future analysis.

## Local Usage

1. Ensure a Docker daemon is running (or equivalent).
2. Set up environment variables in a `.env` file in accordance the
[environment variables](#environment-variables) section.
3. Ensure you have:
    - verified you GCP application default creditals for the
      project setup.
    - the necessary permissions for:
        - reading the input test dataset from the configured bukcet
        - storing evaluation results in the configured firestore database
        - authorised usage with the configured API gateway (JWT signing)
4. Build the Docker image:
```bash
bash scripts/api_evaluation/01-local-build.sh
```
5. Run the Docker container, selecting either a SIC or SOC evaluation:
```bash
bash scripts/api_evaluation/02-local-run.sh sic|soc
```

## Environment Variables

| Environment Variable | Description |
| --- | --- |
| `PROJECT_ID` | The GCP project ID |
| `EVALUATION_BUCKET_NAME` | The name of the bucket in which the input test data is stored |
| `API_GATEWAY` | The API Gateway, excluding the `https://` prefix |
| `SA_EMAIL` | The service account email address to use when signing API Gateway JWTs |
| `API_EVAL_FIRESTORE_DB_ID` | Name of the firestore database to use when storing the evaluation results |
| `API_EVAL_FIRESTORE_COLLECTION_ID` | Name of the firestore collection (within the database) to use when storing the evaluation results |
| `API_EVAL_ENVIRONMENT` |  Name of the evaluation environment e.g. `sandbox`, `dev` etc. |
| `CLOUD_RUN_EXECUTION` | A unique ID for a given evaluation pipeline run. Note: when setting this locally, any value is valid (would be set automatically when executed in GCP cloud run). |
| `LOG_LEVEL` | The level of logging required during the run. Must be one of `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`|
| `API_EVAL_LOOKUP_SEMAPHORE_LIMIT` | The number of parallel requests made to the API's lookup endpoint concurrently, by default 5. |
| `API_EVAL_CLASSIFY_SEMAPHORE_LIMIT` | The number of parallel requests made to the API's classify endpoint concurrently, by default 2. |
| `API_EVAL_KEEP_API_ERRORS` | Keep any records relating to API errors (e.g. 5XXs) in the evaluation metrics (an effective penalisation). By default `True`, and set to any other value to exclude them. |
| `API_EVAL_RANDOM_SAMPLE_SIZE` | Select a random sample of a predefined size, useful when debugging. When unset, the full dataset is used. |

## Notes

- The Docker manifest for the API evaluation pipeline resides at
`./containers/api_evaluation/Dockerfile`.
- The Docker manifest is a multi-stage image build, separating out the build
and runtime dependencies.
- Within the build stage, `poetry` is installed for consistency as
this repo's tool for managing python dependencies. A pre-verified SHA256 is
used to verify the open-source `poetry` installation executable script to
protect against source changes/tampering and prevent supply chain attacks.
This SHA is set as the Docker build argument `POETRY_INSTALLER_SHA256` in the
build stage. It is pre-calculated by running the following commands locally:
```bash
curl -fL -o install-poetry.py https://install.python-poetry.org
shasum -a 256 install-poetry.py
rm -f install-poetry.py
```
and then setting the displayed SHA as that build argument.

> [!WARNING]
> When `poetry` updates the `install-poetry.py` installation script, the
abov reference pre-calculation of the SHA256 step will need to be repeated
otherwise the build will fail at the installation script verification stage.
