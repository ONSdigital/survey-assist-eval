#!/bin/bash
#
# Build a local version of the api_evaluation image
docker build \
    -t sa_api_evaluation:0.1.0 \
    -f containers/api_evaluation/Dockerfile \
    .
