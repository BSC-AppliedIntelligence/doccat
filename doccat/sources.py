"""Work sources: where documents to catalogue come from.

Two sources ship with doccat and both feed the same parallel pipeline:

* `FileSource` — a folder tree of PDF/Office/text files (what `indexer.py` did);
* `JsonRecordSource` — a JSON array dump where each element carries a text
  field plus scalar metadata (what `json_indexer.py` did).

A source turns its input into `WorkItem`s (cheap, no text extraction yet) and
knows how to materialise the text of one item. Everything else — profile
routing, LLM calls, parallelism, checkpointing — is source-agnostic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from extractors import (LegacyFormatError, SUPPORTED_EXTENSIONS, doc_id_for,
                        extract_text, file_sha256)

log = logging.getLogger("doccat.sources")

__all__ = ["WorkItem", "Extraction", "FileSource", "JsonRecordSource", "extract_job"]


@dataclass
class WorkItem:
    """One unit of cataloguing work, before any text has been read."""

    doc_id: str
    key: str                      # stable human-facing key (relative path / record key)
    label: str                    # what the prompt calls the document
    source_kind: str              # "files" | "json"
    change_key: str               # content hash, or mtime:size when hashing is off
    path: Path | None = None
    filetype: str = ""
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)   # routing + provenance
    inline_text: str | None = None                           # JSON sources carry text already
    base_record: dict[str, Any] = field(default_factory=dict)


@dataclass
class Extraction:
    """Result of materialising an item's text."""

    text: str = ""
    units: int = 0                # pages / paragraphs / slides / sheets / lines
    error: str | None = None
    legacy: bool = False


# ---------------------------------------------------------------------------
# Process-pool entry point (must be a module-level function to be picklable)
# ---------------------------------------------------------------------------

def extract_job(path_str: str) -> tuple[str, int, str | None, bool]:
    """Extract text from one file. Runs in a worker process; never raises."""
    path = Path(path_str)
    try:
        text, units = extract_text(path)
        return text, units, None, False
    except LegacyFormatError as e:
        return "", 0, str(e), True
    except ModuleNotFoundError as e:
        hint = {"pptx": "python-pptx", "docx": "python-docx"}.get(
            e.name or "", e.name or "the missing library")
        return "", 0, f"missing library for {path.suffix}: {e.name} — run: pip install {hint}", False
    except Exception as e:  # noqa: BLE001 - a bad file must not kill the pool
        return "", 0, f"{type(e).__name__}: {e}", False


# ---------------------------------------------------------------------------
# Filesystem source
# ---------------------------------------------------------------------------

class FileSource:
    """A folder tree of documents."""

    kind = "files"

    def __init__(self, root: Path, *, include: Iterable[str] = (),
                 exclude: Iterable[str] = (), extensions: Iterable[str] | None = None,
                 follow_symlinks: bool = False) -> None:
        self.root = Path(root).resolve()
        self.include = list(include)
        self.exclude = list(exclude)
        self.extensions = {e if e.startswith(".") else f".{e}"
                           for e in (extensions or SUPPORTED_EXTENSIONS)}
        self.follow_symlinks = follow_symlinks

    def _iter_paths(self) -> list[Path]:
        import fnmatch

        paths: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=self.follow_symlinks):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                p = Path(dirpath) / name
                if p.suffix.lower() not in self.extensions:
                    continue
                rel = str(p.relative_to(self.root)).replace("\\", "/")
                if self.include and not any(fnmatch.fnmatch(rel, g) for g in self.include):
                    continue
                if self.exclude and any(fnmatch.fnmatch(rel, g) for g in self.exclude):
                    continue
                paths.append(p)
        paths.sort()
        return paths

    def scan(self, *, hash_workers: int = 8, change_detection: str = "hash") -> list[WorkItem]:
        """Enumerate documents and compute their change keys (hashing in parallel)."""
        paths = self._iter_paths()
        log.info("Scanning %s: %d supported document(s)", self.root, len(paths))

        def build(path: Path) -> WorkItem | None:
            try:
                stat = path.stat()
            except OSError as e:
                log.warning("Cannot stat %s: %s — skipping", path, e)
                return None
            if change_detection == "mtime":
                change_key = f"mtime:{int(stat.st_mtime)}:{stat.st_size}"
                content_hash = None
            else:
                try:
                    content_hash = file_sha256(path)
                except OSError as e:
                    log.warning("Cannot read %s: %s — skipping", path, e)
                    return None
                change_key = content_hash
            rel = str(path.relative_to(self.root))
            record = {
                "doc_id": doc_id_for(path, self.root),
                "path": rel,
                "filename": path.name,
                "filetype": path.suffix.lower().lstrip("."),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "change_key": change_key,
            }
            if content_hash:
                record["content_hash"] = content_hash
            return WorkItem(
                doc_id=record["doc_id"], key=rel, label=path.name, source_kind=self.kind,
                change_key=change_key, path=path, filetype=record["filetype"],
                size_bytes=stat.st_size, base_record=record,
            )

        workers = max(1, min(hash_workers, 32))
        if workers == 1 or len(paths) < 32:
            items = [build(p) for p in paths]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                items = list(pool.map(build, paths))
        return [i for i in items if i is not None]

    def load_text(self, item: WorkItem) -> Extraction:
        text, units, error, legacy = extract_job(str(item.path))
        return Extraction(text=text, units=units, error=error, legacy=legacy)

    def save_text_cache(self, cache_dir: Path, item: WorkItem, text: str) -> str:
        """Persist the extraction as Markdown + YAML front matter, mirroring the tree."""
        rel = Path(item.key)
        out = cache_dir / rel.parent / (rel.name + ".md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_front_matter({
            "doc_id": item.doc_id,
            "source_path": item.key,
            "source_type": item.filetype,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "char_count": len(text),
        }) + text, encoding="utf-8")
        return str(out.relative_to(cache_dir))


# ---------------------------------------------------------------------------
# JSON record source
# ---------------------------------------------------------------------------

#: Fields tried, in order, as a stable per-record key.
KEY_FIELDS = ("id", "cve", "numdoc", "uuid", "doc_id")
#: Fields used to build a short human-readable label for the prompt.
LABEL_FIELDS = ("titulo", "title", "orga", "fecha", "date")


class JsonRecordSource:
    """A JSON array (or `{data: [...]}`) where each element holds a text field."""

    kind = "json"

    def __init__(self, input_path: Path, field_name: str = "texto", *,
                 key_fields: Iterable[str] = KEY_FIELDS,
                 label_fields: Iterable[str] = LABEL_FIELDS,
                 limit: int | None = None) -> None:
        self.input_path = Path(input_path)
        self.field = field_name
        self.key_fields = tuple(key_fields)
        self.label_fields = tuple(label_fields)
        self.limit = limit
        self.source_name = self.input_path.name

    def _load_records(self) -> list[dict]:
        with open(self.input_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            for key in ("data", "items", "records", "results"):
                if isinstance(data.get(key), list):
                    records = data[key]
                    break
            else:
                raise ValueError(
                    f"{self.input_path}: expected a JSON array of records (or a dict "
                    f"wrapping one under 'data'/'items'/'records'/'results'), got a dict "
                    f"with keys {list(data.keys())}")
        else:
            raise ValueError(f"{self.input_path}: expected a JSON array of records, "
                             f"got {type(data).__name__}")
        return records[: self.limit] if self.limit is not None else records

    def _key(self, rec: dict, index: int) -> str:
        for candidate in self.key_fields:
            value = rec.get(candidate)
            if value not in (None, ""):
                return str(value)
        return str(index)

    def _label(self, rec: dict, key: str) -> str:
        parts = [str(rec[f]) for f in self.label_fields if rec.get(f) not in (None, "")]
        return " — ".join(parts) if parts else f"record {key}"

    def scan(self, *, hash_workers: int = 8, change_detection: str = "hash") -> list[WorkItem]:
        items: list[WorkItem] = []
        for index, rec in enumerate(self._load_records()):
            if not isinstance(rec, dict):
                log.warning("Record %d is not a JSON object — skipping", index)
                continue
            text = rec.get(self.field)
            if not isinstance(text, str) or not text.strip():
                log.warning("Record %d has no usable text in field %r — skipping",
                            index, self.field)
                continue
            key = self._key(rec, index)
            doc_id = hashlib.sha1(f"{self.source_name}:{key}".encode("utf-8")).hexdigest()[:16]
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            other = {k: v for k, v in rec.items() if k != self.field}
            items.append(WorkItem(
                doc_id=doc_id, key=key, label=self._label(rec, key), source_kind=self.kind,
                change_key=content_hash, filetype="json_record", size_bytes=len(text),
                metadata=other, inline_text=text,
                base_record={
                    "doc_id": doc_id,
                    "source_file": self.source_name,
                    "record_key": key,
                    "record_index": index,
                    "field": self.field,
                    "filename": f"{self.source_name}#{key}",
                    "path": f"{self.source_name}#{key}",
                    "filetype": "json_record",
                    "content_hash": content_hash,
                    "change_key": content_hash,
                    "source_metadata": other,
                },
            ))
        log.info("Loaded %d indexable record(s) from %s", len(items), self.input_path)
        return items

    def load_text(self, item: WorkItem) -> Extraction:
        text = item.inline_text or ""
        return Extraction(text=text, units=text.count("\n") + 1)

    def save_text_cache(self, cache_dir: Path, item: WorkItem, text: str) -> str:
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", item.key).strip("_") or "record"
        out = cache_dir / f"{safe_key}_{item.doc_id}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_front_matter({
            "doc_id": item.doc_id,
            "source_file": self.source_name,
            "record_key": item.key,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "char_count": len(text),
        }) + text, encoding="utf-8")
        return str(out.relative_to(cache_dir))


def _front_matter(fields: dict[str, Any]) -> str:
    lines = ["---", *(f"{k}: {v}" for k, v in fields.items()), "---", ""]
    return "\n".join(lines)
