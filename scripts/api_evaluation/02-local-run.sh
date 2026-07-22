#!/bin/bash
#
# Run a local version of the api_evaluation image
# Notes:
# - container must be built first using 01-local-build.sh,
# - setup and configure the environment variables in .env
# - active GCP application default credentials (to read-only volume mount at
# container runtime for verification purposes).
# - cli args are passed straight to the container entrypoint script
docker run --rm \
    -v "$HOME/.config/gcloud:/gcp/config:ro" \
    --env CLOUDSDK_CONFIG=/gcp/config \
    --env GOOGLE_APPLICATION_CREDENTIALS=/gcp/config/application_default_credentials.json \
    --env-file .env \
    sa_api_evaluation:0.1.0 "$@"
