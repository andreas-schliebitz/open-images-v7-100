#!/usr/bin/env bash

set -e

if [[ -z "${VIRTUAL_ENV}" ]]; then
    >&2 echo "No active virtual environment found."
    exit 1
fi

pip install --upgrade pip
pip install --upgrade -e .
pip install black==25.11.0 pytest==9.0.1
