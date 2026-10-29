#!/usr/bin/env bash
# Example indexing run against the Hugging Face inference router.
# Adjust the root, the catalog path and the worker count, then run it.
#
# Measure before scaling: add --max-docs 100 and read the doc/s the progress
# line reports (see docs/SCALING.md).
set -euo pipefail

doccat index \
    --root "${DOCS_ROOT:-/data/documents}" \
    --catalog "${CATALOG_PATH:-catalog.jsonl}" \
    --base-url "${LLM_BASE_URL:-https://router.huggingface.co/v1}" \
    --model "${LLM_MODEL:-Qwen/Qwen3.6-35B-A3B}" \
    --api-key "${HF_TOKEN:?export HF_TOKEN first}" \
    --bill-to "${HF_BILL_TO:-BSC-LT}" \
    --llm-workers "${DOCCAT_LLM_WORKERS:-16}" \
    "$@"
