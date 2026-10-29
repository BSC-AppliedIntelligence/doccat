"""End-to-end pipeline tests against a mock OpenAI-compatible endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doccat.catalog import CatalogStore, load_catalog
from doccat.llm import LLMClient, LLMConfig
from doccat.pipeline import (PipelineOptions, _shard_of, needs_ocr, run_pipeline,
                             select_work)
from doccat.profiles import load_profiles
from doccat.sources import FileSource, JsonRecordSource


def run(source, catalog: Path, mock, **opt_kwargs):
    profiles = load_profiles(extra_dirs=opt_kwargs.pop("profiles_dir", ()) or ())
    opts = PipelineOptions(text_cache=catalog.parent / "cache", error_log=None,
                           **opt_kwargs)
    cfg = LLMConfig(base_url=mock.base_url, model="mock", max_retries=2,
                    max_rate_limit_retries=4, timeout=30)
    with CatalogStore(catalog, checkpoint_every=opts.checkpoint_every) as store:
        with LLMClient(cfg) as client:
            stats = run_pipeline(source, store, profiles, client, opts)
    return stats, load_catalog(catalog)


def test_indexes_a_tree_and_routes_profiles(docs_tree, tmp_path, mock_llm):
    catalog = tmp_path / "catalog.jsonl"
    stats, records = run(FileSource(docs_tree), catalog, mock_llm, llm_workers=4)

    assert stats.indexed == 5 and stats.failed == 0
    by_path = {r["path"]: r for r in records.values()}
    assert by_path["contracts/supply_contract_2024.md"]["profile"] == "legal"
    assert by_path["contracts/nda_acme.txt"]["profile"] == "legal"
    assert by_path["manuals/installation_guide.md"]["profile"] == "technical"
    assert by_path["misc/notes.txt"]["profile"] == "generic"
    # profile-specific field made it into the record
    assert by_path["contracts/nda_acme.txt"]["parties"] == ["ACME", "Contoso"]
    # every record carries the shared core schema
    for rec in records.values():
        assert rec["status"] == "ok"
        assert rec["title"] and rec["summary"] and rec["keywords"]
        assert rec["ingestion"] == "parse"


def test_second_run_skips_unchanged_and_reindexes_modified(docs_tree, tmp_path, mock_llm):
    catalog = tmp_path / "catalog.jsonl"
    run(FileSource(docs_tree), catalog, mock_llm, llm_workers=4)
    calls_after_first = mock_llm.call_count

    stats, _ = run(FileSource(docs_tree), catalog, mock_llm, llm_workers=4)
    assert stats.skipped == 5 and stats.indexed == 0
    assert mock_llm.call_count == calls_after_first    # no LLM traffic at all

    (docs_tree / "misc" / "notes.txt").write_text("Different content now.", encoding="utf-8")
    stats, _ = run(FileSource(docs_tree), catalog, mock_llm, llm_workers=4)
    assert stats.indexed == 1 and stats.skipped == 4


def test_deleted_files_are_pruned(docs_tree, tmp_path, mock_llm):
    catalog = tmp_path / "catalog.jsonl"
    run(FileSource(docs_tree), catalog, mock_llm)
    (docs_tree / "misc" / "notes.txt").unlink()
    stats, records = run(FileSource(docs_tree), catalog, mock_llm)
    assert stats.removed == 1
    assert not any(r["path"] == "misc/notes.txt" for r in records.values())


def test_forcing_a_profile_recatalogues(docs_tree, tmp_path, mock_llm):
    catalog = tmp_path / "catalog.jsonl"
    run(FileSource(docs_tree), catalog, mock_llm)
    stats, records = run(FileSource(docs_tree), catalog, mock_llm, profile_name="intranet")
    assert stats.indexed == 5
    assert {r["profile"] for r in records.values()} == {"intranet"}
    assert "cataloguing an organisation's internal documentation" in mock_llm.prompts()[-1]


def test_parallel_workers_actually_overlap(tmp_path, mock_llm):
    root = tmp_path / "many"
    root.mkdir()
    for i in range(24):
        (root / f"doc_{i:02d}.txt").write_text(f"Document number {i}.", encoding="utf-8")
    stats, records = run(FileSource(root), tmp_path / "catalog.jsonl", mock_llm,
                         llm_workers=8, extract_workers=2)
    assert stats.indexed == 24 and len(records) == 24
    assert mock_llm.max_concurrent > 1, "requests never overlapped — workers are serialised"


def test_rate_limit_pauses_all_workers_and_the_run_still_completes(tmp_path, mock_llm):
    root = tmp_path / "few"
    root.mkdir()
    for i in range(4):
        (root / f"doc_{i}.txt").write_text(f"Document {i}.", encoding="utf-8")
    mock_llm.fail_with_429 = 3
    stats, records = run(FileSource(root), tmp_path / "catalog.jsonl", mock_llm,
                         llm_workers=4)
    assert stats.indexed == 4 and stats.failed == 0
    assert mock_llm.call_count > 4  # the throttled calls were retried


def test_map_reduce_profile_makes_one_call_per_chunk_plus_one(tmp_path, mock_llm):
    root = tmp_path / "long"
    root.mkdir()
    (root / "huge_report.txt").write_text("sentence. " * 30_000, encoding="utf-8")
    stats, records = run(FileSource(root), tmp_path / "catalog.jsonl", mock_llm,
                         profile_name="long_document")
    rec = next(iter(records.values()))
    assert rec["ingestion"] == "map_reduce"
    assert rec["indexed_chunks"] > 1
    assert mock_llm.call_count == rec["indexed_chunks"] + 1
    assert stats.llm_calls == mock_llm.call_count


def test_json_source_with_metadata_routing(tmp_path, mock_llm):
    dump = tmp_path / "Plenos.json"
    dump.write_text(json.dumps([
        {"id": "1", "orga": "Senado", "fecha": "2024-03-01", "texto": "Sesión plenaria..."},
        {"id": "2", "orga": "Congreso", "fecha": "2024-03-02", "texto": "Otra sesión..."},
        {"id": "3", "texto": ""},  # skipped: no text
    ]), encoding="utf-8")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "senado.yaml").write_text(
        "name: senado\ndescription: Senate sittings\nextends: parliamentary\n"
        "priority: 60\nmatch:\n  metadata_equals:\n    orga: Senado\n", encoding="utf-8")

    stats, records = run(JsonRecordSource(dump, "texto"), tmp_path / "catalog.jsonl",
                         mock_llm, profiles_dir=[profiles_dir])
    assert stats.indexed == 2
    by_key = {r["record_key"]: r for r in records.values()}
    assert by_key["1"]["profile"] == "senado"
    assert by_key["2"]["profile"] == "parliamentary"
    assert by_key["1"]["source_metadata"]["fecha"] == "2024-03-01"
    assert by_key["1"]["chamber"] == "Congreso de los Diputados"  # comes from the text, not the folder


def test_text_cache_is_written(docs_tree, tmp_path, mock_llm):
    catalog = tmp_path / "catalog.jsonl"
    _, records = run(FileSource(docs_tree), catalog, mock_llm)
    cache = tmp_path / "cache"
    rec = next(r for r in records.values() if r["path"] == "misc/notes.txt")
    cached = cache / rec["extracted_text_path"]
    assert cached.exists()
    assert cached.read_text(encoding="utf-8").startswith("---\ndoc_id:")


def test_dry_run_calls_no_llm(docs_tree, tmp_path, mock_llm, caplog):
    catalog = tmp_path / "catalog.jsonl"
    stats, records = run(FileSource(docs_tree), catalog, mock_llm, dry_run=True)
    assert mock_llm.call_count == 0
    assert stats.todo == 5
    assert not records


def test_include_exclude_and_max_docs(docs_tree, tmp_path, mock_llm):
    stats, records = run(FileSource(docs_tree, exclude=["misc/*"]),
                         tmp_path / "a.jsonl", mock_llm)
    assert stats.indexed == 3
    stats, records = run(FileSource(docs_tree, include=["contracts/*"]),
                         tmp_path / "b.jsonl", mock_llm, max_docs=1)
    assert stats.indexed == 1


def test_sharding_partitions_the_corpus(tmp_path, mock_llm):
    root = tmp_path / "shardable"
    root.mkdir()
    for i in range(20):
        (root / f"doc_{i:02d}.txt").write_text(f"Doc {i}", encoding="utf-8")

    shards = []
    for i in range(4):
        out = tmp_path / f"shard_{i}.jsonl"
        stats, records = run(FileSource(root), out, mock_llm, shard=(i, 4))
        shards.append(records)
        assert stats.indexed == len(records)

    merged = {}
    for part in shards:
        merged.update(part)
    assert len(merged) == 20                        # complete
    assert sum(len(p) for p in shards) == 20        # and disjoint


def test_shard_assignment_is_stable():
    assert _shard_of("abc123", 4) == _shard_of("abc123", 4)
    assert 0 <= _shard_of("abc123", 4) < 4


def test_needs_ocr_detection():
    assert needs_ocr("pdf", "[page 1]\n\n[page 2]\n", 2)
    assert not needs_ocr("pdf", "x" * 500, 2)
    assert not needs_ocr("docx", "", 3)


def test_unparseable_reply_is_recorded_as_an_error(docs_tree, tmp_path, mock_llm):
    mock_llm.reply = "I am afraid I cannot do that."
    stats, records = run(FileSource(docs_tree), tmp_path / "catalog.jsonl", mock_llm)
    assert stats.indexed == 0 and stats.failed == 5
    assert all(r["status"] == "metadata_parse_error" for r in records.values())

    # ...and is retried on the next run without --force
    mock_llm.reply = None
    stats, records = run(FileSource(docs_tree), tmp_path / "catalog.jsonl", mock_llm)
    assert stats.indexed == 5 and stats.skipped == 0


def test_incomplete_replies_are_flagged(docs_tree, tmp_path, mock_llm):
    mock_llm.reply = {"summary": "", "keywords": []}
    _, records = run(FileSource(docs_tree), tmp_path / "catalog.jsonl", mock_llm)
    rec = next(iter(records.values()))
    assert rec["status"] == "ok"
    assert set(rec["incomplete_fields"]) == {"summary", "keywords"}


def test_select_work_honours_force(docs_tree, tmp_path, mock_llm):
    catalog = tmp_path / "catalog.jsonl"
    run(FileSource(docs_tree), catalog, mock_llm)
    source = FileSource(docs_tree)
    items = source.scan()
    with CatalogStore(catalog) as store:
        todo, skipped = select_work(items, store, PipelineOptions())
        assert not todo and skipped == 5
        todo, skipped = select_work(items, store, PipelineOptions(force=True))
        assert len(todo) == 5 and skipped == 0


# ---------------------------------------------------------------------------
# Graph extraction: inline vs. dedicated-pass relations through the real pipeline
# ---------------------------------------------------------------------------

def test_inline_relation_extraction_costs_no_extra_call(docs_tree, tmp_path, mock_llm):
    """legal.yaml has graph: without dedicated_pass — relations ride the one call."""
    catalog = tmp_path / "catalog.jsonl"
    stats, records = run(FileSource(docs_tree), catalog, mock_llm)
    legal_docs = [r for r in records.values() if r["profile"] == "legal"]
    assert legal_docs and all(r.get("relations") for r in legal_docs)
    assert stats.llm_calls == 5          # one call per document, same as without graph


def test_dedicated_pass_costs_one_extra_call_and_grounds_in_entities(tmp_path, mock_llm):
    corpus = tmp_path / "corpus"
    (corpus / "plenos").mkdir(parents=True)
    (corpus / "plenos" / "ds_2024_01.txt").write_text(
        "Sesion plenaria. " + "Debate largo. " * 40, encoding="utf-8")
    stats, records = run(FileSource(corpus), tmp_path / "catalog.jsonl", mock_llm)

    rec = next(iter(records.values()))
    assert rec["profile"] == "parliamentary"
    assert rec["relations"] == [
        {"subject": "ACME", "relation": "supplies_to", "object": "Contoso",
         "description": "ACME supplies widgets to Contoso.", "strength": 0.9}]
    assert stats.llm_calls == 2          # main call + dedicated relations call

    prompts = mock_llm.prompts()
    relation_prompt = next(p for p in prompts if "already identified these entities" in p)
    assert "ACME" in relation_prompt     # grounded in the main call's own entities output


def test_dedicated_pass_failure_leaves_empty_relations_not_a_failed_document(
        tmp_path, mock_llm):
    corpus = tmp_path / "corpus"
    (corpus / "plenos").mkdir(parents=True)
    (corpus / "plenos" / "ds_2024_01.txt").write_text(
        "Sesion plenaria sobre presupuestos.", encoding="utf-8")
    original_reply = mock_llm._make_reply

    def flaky_reply(prompt):
        if "already identified these entities" in prompt:
            return "not json at all"
        return original_reply(prompt)
    mock_llm._make_reply = flaky_reply

    stats, records = run(FileSource(corpus), tmp_path / "catalog.jsonl", mock_llm)
    rec = next(iter(records.values()))
    assert rec["status"] == "ok"          # the document itself still indexed fine
    assert rec.get("relations") == []     # relations degrade to empty, not an error


def test_min_confidence_filters_weak_relations_end_to_end(tmp_path, mock_llm):
    corpus = tmp_path / "corpus"
    (corpus / "contracts").mkdir(parents=True)
    (corpus / "contracts" / "supply.txt").write_text(
        "Supply contract between ACME and Contoso.", encoding="utf-8")
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "strict_legal.yaml").write_text("""
name: strict_legal
extends: legal
priority: 90
match:
  path_globs: ["**/contracts/**"]
graph:
  min_confidence: 0.5
""", encoding="utf-8")

    stats, records = run(FileSource(corpus), tmp_path / "catalog.jsonl", mock_llm,
                         profiles_dir=[profiles_dir])
    rec = next(iter(records.values()))
    assert rec["profile"] == "strict_legal"
    # the mock's inline reply has one relation at 0.9 and one at 0.1
    assert rec["relations"] == [
        {"subject": "ACME", "relation": "supplies_to", "object": "Contoso",
         "description": "ACME supplies widgets to Contoso.", "strength": 0.9}]
    assert rec["relations_below_confidence"] == 1


def test_graph_disabled_profile_has_no_relations_field_at_all(docs_tree, tmp_path, mock_llm):
    stats, records = run(FileSource(docs_tree), tmp_path / "catalog.jsonl", mock_llm)
    generic_docs = [r for r in records.values() if r["profile"] == "generic"]
    technical_docs = [r for r in records.values() if r["profile"] == "technical"]
    assert generic_docs and all("relations" not in r for r in generic_docs)
    assert technical_docs and all("relations" not in r for r in technical_docs)
