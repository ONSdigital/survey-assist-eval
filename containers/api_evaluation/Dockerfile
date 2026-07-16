#checkov:skip=CKV_DOCKER_2: healthchecks will be managed by container orchestrator

#########
# Build #
#########
# using sha to ensure reproducible builds and prevent supply chain attacks
FROM python:3.12-slim@sha256:c3d81d25b3154142b0b42eb1e61300024426268edeb5b5a26dd7ddf64d9daf28 AS builder

ARG POETRY_VERSION=2.4.1
# pre-verified sha checksum for the poetry installer script to prevent supply chain attacks
ARG POETRY_INSTALLER_SHA256=75745ca71373a7b22fa150953543f03d826a52f8e4bc4350328a33bddd668026

# poetry build defaults (non-interactive, local .venv, cache directory).
# Add Poetry and project venv executables to PATH.
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=${POETRY_VERSION} \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_CACHE_DIR="/var/cache/pypoetry" \
    PATH=/app/.venv/bin:/opt/poetry/bin:$PATH

# install poetry, verifying sha checksum
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates build-essential && \
    curl -fL -o /tmp/install-poetry.py https://install.python-poetry.org && \
    echo "${POETRY_INSTALLER_SHA256}  /tmp/install-poetry.py" | sha256sum -c - && \
    python3 /tmp/install-poetry.py --version "${POETRY_VERSION}" && \
    poetry --version && \
    rm -f /tmp/install-poetry.py && \
    apt-get purge -y --auto-remove curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy dev dependencies and install them (separate docker layers to improve caching)
# install only required dependencies only and clear cache to reduce image size
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --only main --sync --no-ansi && \
    rm -rf "$POETRY_CACHE_DIR" /root/.cache/pip

# copy source code and install the package (separate docker layers to improve caching)
COPY src/ ./src
COPY README.md ./
RUN poetry install --only-root --no-ansi && \
    rm -rf "$POETRY_CACHE_DIR" /root/.cache/pip

# bring source script into the image
COPY scripts/api_evaluation/run.py ./run.py

###########
# Runtime #
###########
# using sha to ensure reproducible builds and prevent supply chain attacks
FROM python:3.12-slim@sha256:c3d81d25b3154142b0b42eb1e61300024426268edeb5b5a26dd7ddf64d9daf28 AS runtime

# set runtime Python behavior (no bytecode, unbuffered output, optimised mode)
# and expose the project virtualenv on PATH.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1 \
    PATH=/app/.venv/bin:$PATH

# create a non-root user to run the application
RUN useradd -m appuser

WORKDIR /app

# copy venv and source code from builder, setting ownership to the non-root user
COPY --from=builder --chown=appuser:appuser /app /app

# switch to the non-root user before entrypoint call
USER appuser

ENTRYPOINT ["python", "run.py"]
