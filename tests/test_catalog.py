"""Catalog store: concurrency, crash recovery, atomic checkpoints."""

from __future__ import annotations

import json
import threading

import pytest

from doccat.catalog import CatalogStore, load_catalog, save_catalog


def test_concurrent_adds_are_all_persisted(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    with CatalogStore(catalog, checkpoint_every=7) as store:
        def worker(base: int) -> None:
            for i in range(50):
                store.add({"doc_id": f"{base}-{i}", "status": "ok"})

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert len(load_catalog(catalog)) == 400


def test_journal_recovers_records_written_after_the_last_checkpoint(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    store = CatalogStore(catalog, checkpoint_every=1000)   # never checkpoints
    store.add({"doc_id": "a", "status": "ok"})
    store.add({"doc_id": "b", "status": "ok"})
    # Simulate a crash: the journal is on disk, the catalog is not written yet.
    store._journal_file.close()
    assert not catalog.exists()
    assert store.journal_path.exists()

    with CatalogStore(catalog, checkpoint_every=1000) as recovered:
        assert recovered.ids() == {"a", "b"}
    assert set(load_catalog(catalog)) == {"a", "b"}


def test_corrupt_lines_are_skipped_not_fatal(tmp_path, caplog):
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text('{"doc_id": "ok"}\n{"doc_id": broken\n{"doc_id": "ok2"}\n',
                       encoding="utf-8")
    assert set(load_catalog(catalog)) == {"ok", "ok2"}


def test_checkpoint_is_atomic(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    save_catalog(catalog, {"a": {"doc_id": "a"}})
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads(catalog.read_text())["doc_id"] == "a"


def test_remove_prunes_records(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    with CatalogStore(catalog) as store:
        store.add({"doc_id": "a"})
        store.add({"doc_id": "b"})
        assert store.remove(["a", "missing"]) == 1
    assert set(load_catalog(catalog)) == {"b"}


def test_existing_catalog_is_loaded_on_open(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    save_catalog(catalog, {"a": {"doc_id": "a", "status": "ok"}})
    with CatalogStore(catalog) as store:
        assert len(store) == 1
        assert store.status_counts() == {"ok": 1}
