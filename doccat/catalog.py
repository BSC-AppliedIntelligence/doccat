"""Crash-safe, concurrency-safe JSONL catalog store.

The catalog format is unchanged — one JSON object per line, `doc_id` as key —
so existing catalogs and `server.py` keep working. What changes is how it is
written when dozens of workers finish documents at once:

* every finished record is appended to a **journal** (`<catalog>.journal`)
  under a lock, so a crash (or a Ctrl-C, or an OOM kill on document 8,431)
  loses at most the records still in flight;
* the full catalog is rewritten atomically every `checkpoint_every` records —
  a rewrite of thousands of records is far too expensive to do per document;
* on start-up, a leftover journal is replayed into the catalog automatically,
  so an interrupted run resumes instead of restarting.

The store is the single writer: workers only call `add()`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import Counter
from pathlib import Path

log = logging.getLogger("doccat.catalog")

__all__ = ["CatalogStore", "load_catalog", "save_catalog"]


def load_catalog(catalog_path: Path) -> dict[str, dict]:
    """Read a JSONL catalog into a dict keyed by `doc_id` (bad lines are skipped)."""
    catalog: dict[str, dict] = {}
    if not catalog_path.exists():
        return catalog
    with open(catalog_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("%s:%d is not valid JSON (%s) — skipping the line",
                            catalog_path, lineno, e)
                continue
            if "doc_id" in rec:
                catalog[rec["doc_id"]] = rec
    return catalog


def save_catalog(catalog_path: Path, catalog: dict[str, dict]) -> None:
    """Write the whole catalog atomically (temp file + rename, fsynced)."""
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = catalog_path.with_suffix(catalog_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in catalog.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(catalog_path)


class CatalogStore:
    """In-memory catalog with journalled, checkpointed persistence.

    Usage::

        with CatalogStore(Path("catalog.jsonl")) as store:
            store.add(record)          # thread-safe, called by workers
    """

    def __init__(self, catalog_path: Path, checkpoint_every: int = 50,
                 journal: bool = True) -> None:
        self.path = Path(catalog_path)
        self.checkpoint_every = max(int(checkpoint_every), 1)
        self.journal_path = self.path.with_suffix(self.path.suffix + ".journal")
        self._records = load_catalog(self.path)
        self._lock = threading.Lock()
        self._since_checkpoint = 0
        self._dirty = False
        self._journal_file = None
        self._use_journal = journal
        recovered = self._recover()
        if recovered:
            # The recovered records exist only in the journal so far — mark the
            # store dirty so the checkpoint below actually writes them out
            # before the journal is reset.
            self._dirty = True
            log.warning("Recovered %d record(s) from an interrupted run (%s)",
                        recovered, self.journal_path)
            self.checkpoint()
        if self._use_journal:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            self._journal_file = open(self.journal_path, "a", encoding="utf-8")

    # -- recovery ----------------------------------------------------------
    def _recover(self) -> int:
        if not self.journal_path.exists():
            return 0
        recovered = 0
        with open(self.journal_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last line of an interrupted write
                if "doc_id" in rec:
                    self._records[rec["doc_id"]] = rec
                    recovered += 1
        return recovered

    # -- reads -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._records

    def get(self, doc_id: str) -> dict | None:
        return self._records.get(doc_id)

    def ids(self) -> set[str]:
        return set(self._records)

    def records(self) -> list[dict]:
        return list(self._records.values())

    def status_counts(self) -> dict[str, int]:
        return dict(Counter(r.get("status", "unknown") for r in self._records.values()))

    # -- writes ------------------------------------------------------------
    def add(self, record: dict) -> None:
        """Store one finished record. Safe to call from any worker thread."""
        with self._lock:
            self._records[record["doc_id"]] = record
            self._dirty = True
            if self._journal_file is not None:
                self._journal_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._journal_file.flush()
            self._since_checkpoint += 1
            due = self._since_checkpoint >= self.checkpoint_every
        if due:
            self.checkpoint()

    def remove(self, doc_ids) -> int:
        """Drop records (e.g. documents deleted from disk). Returns how many went."""
        removed = 0
        with self._lock:
            for doc_id in list(doc_ids):
                if self._records.pop(doc_id, None) is not None:
                    removed += 1
                    self._dirty = True
        return removed

    def checkpoint(self) -> None:
        """Rewrite the catalog atomically and truncate the journal."""
        with self._lock:
            if not self._dirty:
                self._since_checkpoint = 0
                return
            snapshot = dict(self._records)
            self._since_checkpoint = 0
            self._dirty = False
            save_catalog(self.path, snapshot)
            if self._journal_file is not None:
                self._journal_file.close()
                self._journal_file = open(self.journal_path, "w", encoding="utf-8")

    def close(self) -> None:
        self.checkpoint()
        with self._lock:
            if self._journal_file is not None:
                self._journal_file.close()
                self._journal_file = None
        # The journal is only useful while a run is in flight.
        if self._use_journal and self.journal_path.exists():
            try:
                self.journal_path.unlink()
            except OSError:  # pragma: no cover - best effort
                pass

    def __enter__(self) -> "CatalogStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_path_list(catalog_path: Path, records, filename: str, predicate, label: str) -> None:
    """Write a plain-text list of document paths matching `predicate` next to the catalog."""
    docs = sorted(r.get("path") or r.get("filename", "") for r in records if predicate(r))
    out = catalog_path.with_name(filename)
    if docs:
        out.write_text("\n".join(docs) + "\n", encoding="utf-8")
        log.warning("%d document(s) %s — list written to %s", len(docs), label, out)
    elif out.exists():
        out.unlink()
