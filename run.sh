#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: ./run.sh INPUT OUT" >&2
  exit 2
fi

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
llm-dataset-generator --input "$1" --out "$2"
