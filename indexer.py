#!/usr/bin/env python3
"""Backwards-compatible front end for `doccat index`.

The indexer now lives in the `doccat` package, which adds domain cataloguing
profiles and a parallel pipeline. This module keeps the original command line
working unchanged::

    python indexer.py --root /path/to/documents --catalog ./catalog.jsonl \
        --base-url http://localhost:8080/v1 --model my-local-model --mode auto

Every new option is available here too (`--profile`, `--llm-workers`,
`--extract-workers`, `--shard`, `--dry-run`, ...); run `python indexer.py -h`.
New work should call `doccat index` directly:

    doccat index --root /path/to/documents --profile legal -j 16

The helper names this module used to export (`LLMConfig`, `call_llm`,
`build_excerpt`, `parse_metadata_json`, `load_catalog`, `save_catalog`,
`METADATA_PROMPT`, `EXCERPT_SUFFIX`) are still importable; they now delegate to
`doccat` and are deprecated.
"""

from __future__ import annotations

import sys
from pathlib import Path

from doccat.catalog import load_catalog, save_catalog          # noqa: F401  (legacy re-export)
from doccat.cli import main as _doccat_main
from doccat.jsonparse import parse_metadata, parse_metadata_json  # noqa: F401  (legacy re-export)
from doccat.llm import EmptyLLMReply, LLMClient, LLMConfig      # noqa: F401  (legacy re-export)
from doccat.pipeline import MAX_UPLOAD_BYTES, OCR_MIN_CHARS_PER_PAGE  # noqa: F401
from doccat.profiles import load_profiles

__all__ = [
    "LLMConfig", "LLMClient", "EmptyLLMReply", "call_llm", "build_excerpt",
    "parse_metadata_json", "parse_metadata", "load_catalog", "save_catalog",
    "METADATA_PROMPT", "EXCERPT_SUFFIX", "main",
]

# Kept for callers that used the old constants. They are derived from the
# `generic` profile so they cannot drift from the prompt actually used.
_LEGACY_EXCERPT_SUFFIX = """
Document content:
---
{excerpt}
---"""


def _legacy_metadata_prompt() -> str:
    """The `generic` profile's prompt, as a `str.format` template on {filename}."""
    prompt = load_profiles().get("generic").build_prompt("\x00FILENAME\x00")
    return prompt.replace("{", "{{").replace("}", "}}").replace("\x00FILENAME\x00", "{filename}")


def __getattr__(name: str):  # PEP 562 — computed lazily, no import-time cost
    if name == "METADATA_PROMPT":
        return _legacy_metadata_prompt()
    if name == "EXCERPT_SUFFIX":
        return _LEGACY_EXCERPT_SUFFIX
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_excerpt(text: str) -> str:
    """Deprecated: use `Profile.excerpt()`. Head+tail excerpt with generic defaults."""
    return load_profiles().get("generic").excerpt(text)


def call_llm(cfg: LLMConfig, prompt: str, file_path: Path | None = None) -> str:
    """Deprecated: use `doccat.llm.LLMClient`. One-shot completion helper."""
    from extractors import MIME_TYPES

    with LLMClient(cfg) as client:
        return client.complete(prompt, file_path=file_path, mime_types=MIME_TYPES)


def main() -> None:
    raise SystemExit(_doccat_main(["index", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
