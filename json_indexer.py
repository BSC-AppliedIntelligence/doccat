#!/usr/bin/env python3
"""Backwards-compatible front end for `doccat json`.

Catalogues a JSON array dump (one record per element, one field holding the
text) with the same metadata schema, profiles and parallel pipeline as the
file indexer. The original command line still works::

    python json_indexer.py --input Plenos.json --field texto \
        --catalog ./plenos_catalog.jsonl --base-url http://localhost:8080/v1 \
        --model my-local-model

New options are available here too — in particular `--profile parliamentary`
and `--llm-workers`. New work should call `doccat json` directly.
"""

from __future__ import annotations

import sys

from doccat.catalog import load_catalog, save_catalog          # noqa: F401  (legacy re-export)
from doccat.cli import main as _doccat_main
from doccat.jsonparse import (find_brace_blocks, parse_metadata,  # noqa: F401
                              parse_metadata_json)
from doccat.sources import KEY_FIELDS, LABEL_FIELDS, JsonRecordSource  # noqa: F401

__all__ = ["parse_llm_metadata", "load_catalog", "save_catalog",
           "JsonRecordSource", "main"]


def parse_llm_metadata(raw: str) -> dict:
    """Deprecated alias of `doccat.jsonparse.parse_metadata`."""
    return parse_metadata(raw)


def main() -> None:
    raise SystemExit(_doccat_main(["json", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
