"""CLI surface: argument wiring, profile commands, merge, stats, back-compat."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doccat.cli import build_parser, build_pipeline_options, main


def parse(argv):
    return build_parser().parse_args(argv)


def test_index_maps_flags_onto_pipeline_options(tmp_path):
    args = parse(["index", "--root", str(tmp_path), "--catalog", str(tmp_path / "c.jsonl"),
                  "-j", "16", "--extract-workers", "4", "--profile", "legal",
                  "--shard", "1/4", "--checkpoint-every", "10"])
    opts = build_pipeline_options(args)
    assert opts.llm_workers == 16
    assert opts.extract_workers == 4
    assert opts.profile_name == "legal"
    assert opts.shard == (1, 4)
    assert opts.checkpoint_every == 10
    assert opts.text_cache == tmp_path / "extracted_text"
    assert opts.error_log == tmp_path / "llm_errors"


def test_bad_shard_is_rejected(tmp_path):
    for bad in ["4/4", "9/4", "nonsense", "1/0"]:
        with pytest.raises(SystemExit):
            build_pipeline_options(parse(["index", "--root", str(tmp_path), "--shard", bad]))


def test_env_defaults_are_picked_up(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCCAT_LLM_WORKERS", "32")
    monkeypatch.setenv("DOCCAT_PROFILE", "financial")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    import importlib

    import doccat.cli as cli
    importlib.reload(cli)
    args = cli.build_parser().parse_args(["index", "--root", str(tmp_path)])
    assert args.llm_workers == 32
    assert args.profile == "financial"
    assert args.model == "some-model"
    importlib.reload(cli)


def test_profiles_list_and_show(capsys):
    assert main(["profiles", "list"]) == 0
    out = capsys.readouterr().out
    assert "legal" in out and "parliamentary" in out

    assert main(["profiles", "show", "legal"]) == 0
    out = capsys.readouterr().out
    assert "legal document cataloguer" in out
    assert '"parties"' in out


def test_profiles_init_scaffolds_a_loadable_profile(tmp_path, capsys):
    out_file = tmp_path / "mydomain.yaml"
    assert main(["profiles", "init", "mydomain", "-o", str(out_file)]) == 0
    assert out_file.exists()
    assert main(["profiles", "validate", "--profiles-dir", str(tmp_path)]) == 0
    assert "mydomain" in capsys.readouterr().out


def test_profiles_init_refuses_to_clobber(tmp_path):
    out_file = tmp_path / "x.yaml"
    out_file.write_text("name: x\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["profiles", "init", "x", "-o", str(out_file)])


def test_merge_combines_shards(tmp_path, capsys):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({"doc_id": "1", "status": "ok"}) + "\n", encoding="utf-8")
    b.write_text(json.dumps({"doc_id": "2", "status": "ok"}) + "\n", encoding="utf-8")
    merged = tmp_path / "all.jsonl"
    assert main(["merge", str(a), str(b), "-o", str(merged)]) == 0
    assert len(merged.read_text().strip().splitlines()) == 2


def test_stats_reports_profiles(tmp_path, capsys):
    catalog = tmp_path / "c.jsonl"
    catalog.write_text("\n".join(json.dumps(r) for r in [
        {"doc_id": "1", "status": "ok", "profile": "legal", "filetype": "pdf",
         "doc_type": "contract"},
        {"doc_id": "2", "status": "llm_error", "profile": "generic", "filetype": "pdf"},
    ]) + "\n", encoding="utf-8")
    assert main(["stats", "--catalog", str(catalog)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["documents"] == 2
    assert report["by_profile"] == {"legal": 1, "generic": 1}
    assert report["by_status"] == {"ok": 1, "llm_error": 1}


def test_unknown_profile_is_rejected_early(tmp_path, docs_tree):
    with pytest.raises(SystemExit):
        main(["index", "--root", str(docs_tree), "--catalog", str(tmp_path / "c.jsonl"),
              "--profile", "does-not-exist", "--dry-run"])


def test_dry_run_through_the_cli(docs_tree, tmp_path, caplog):
    import logging
    with caplog.at_level(logging.INFO):
        assert main(["index", "--root", str(docs_tree),
                     "--catalog", str(tmp_path / "c.jsonl"), "--dry-run"]) == 0
    assert "would be catalogued" in caplog.text
    assert not (tmp_path / "c.jsonl").exists() or (tmp_path / "c.jsonl").read_text() == ""


def test_legacy_indexer_entry_points_still_import():
    import indexer
    import json_indexer

    assert "{filename}" in indexer.METADATA_PROMPT
    assert indexer.EXCERPT_SUFFIX.strip().startswith("Document content:")
    assert callable(indexer.call_llm) and callable(indexer.build_excerpt)
    assert json_indexer.parse_llm_metadata('{"title": "x"}') == {"title": "x"}


def test_full_cli_run_against_a_live_endpoint(docs_tree, tmp_path, mock_llm, monkeypatch):
    """The real `doccat index` path end to end: flags in, catalog + side files out."""
    catalog = tmp_path / "catalog.jsonl"
    rc = main(["index", "--root", str(docs_tree), "--catalog", str(catalog),
               "--base-url", mock_llm.base_url, "--model", "mock",
               "-j", "4", "--extract-workers", "2", "--checkpoint-every", "2"])
    assert rc == 0

    records = [json.loads(line) for line in catalog.read_text().splitlines()]
    assert len(records) == 5
    assert {r["profile"] for r in records} == {"legal", "technical", "generic"}
    assert (tmp_path / "extracted_text").is_dir()

    # stats reads back what index wrote
    assert main(["stats", "--catalog", str(catalog)]) == 0

    # a second run is a no-op
    calls = mock_llm.call_count
    assert main(["index", "--root", str(docs_tree), "--catalog", str(catalog),
                 "--base-url", mock_llm.base_url, "--model", "mock"]) == 0
    assert mock_llm.call_count == calls


def test_cli_run_with_a_custom_profile_directory(docs_tree, tmp_path, mock_llm):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "house.yaml").write_text(
        "name: house\ndescription: House style\nextends: generic\npriority: 99\n"
        "match:\n  any_of:\n    - filetypes: [html]\n"
        "prompt:\n  role: You are the house cataloguer.\n"
        "  fields:\n    house_code:\n      type: string\n"
        "      description: internal code\n", encoding="utf-8")

    catalog = tmp_path / "catalog.jsonl"
    assert main(["index", "--root", str(docs_tree), "--catalog", str(catalog),
                 "--profiles-dir", str(profiles_dir),
                 "--base-url", mock_llm.base_url, "--model", "mock"]) == 0

    records = {r["path"]: r for r in
               (json.loads(line) for line in catalog.read_text().splitlines())}
    assert records["misc/readme.html"]["profile"] == "house"
    assert "You are the house cataloguer." in mock_llm.prompts()[-1] or any(
        "house cataloguer" in p for p in mock_llm.prompts())


# ---------------------------------------------------------------------------
# `doccat graph` — consolidation CLI
# ---------------------------------------------------------------------------

def _index_a_small_graph_corpus(tmp_path, mock_llm):
    """Two contracts, inline (legal) relation extraction, cross-document aliasing."""
    root = tmp_path / "docs" / "contracts"
    root.mkdir(parents=True)
    (root / "supply_acme.txt").write_text(
        "Supply agreement. ACME Corp supplies widgets to Contoso.", encoding="utf-8")
    (root / "supply_globex.txt").write_text(
        "Supply agreement. Globex Inc supplies gadgets to Contoso. "
        "Acme Corporation competes with Globex Inc.", encoding="utf-8")

    mock_llm.reply = None
    original_reply = mock_llm._make_reply

    def reply(prompt: str) -> str:
        if '"relations"' not in prompt or "already identified" in prompt:
            return original_reply(prompt)
        if "ACME Corp supplies widgets" in prompt:
            return json.dumps({
                "title": "T", "summary": "S", "description": "D", "doc_type": "contract",
                "language": "en", "keywords": ["k"], "topics": ["t"], "questions_answered": ["q"],
                "entities": {"people": [], "organizations": ["ACME Corp", "Contoso"],
                            "places": [], "dates": []},
                "parties": ["ACME Corp", "Contoso"],
                "relations": [{"subject": "ACME Corp", "relation": "supplies_to",
                              "object": "Contoso", "description": "ACME supplies widgets.",
                              "strength": 0.9}],
            })
        return json.dumps({
            "title": "T", "summary": "S", "description": "D", "doc_type": "contract",
            "language": "en", "keywords": ["k"], "topics": ["t"], "questions_answered": ["q"],
            "entities": {"people": [],
                        "organizations": ["Globex Inc", "Contoso", "Acme Corporation"],
                        "places": [], "dates": []},
            "parties": ["Globex Inc", "Contoso"],
            "relations": [
                {"subject": "Globex Inc", "relation": "supplies_to", "object": "Contoso",
                 "description": "Globex supplies gadgets.", "strength": 0.85},
                {"subject": "Acme Corporation", "relation": "competes_with",
                 "object": "Globex Inc", "description": "Competitors.", "strength": 0.6},
            ],
        })
    mock_llm._make_reply = reply

    catalog = tmp_path / "catalog.jsonl"
    assert main(["index", "--root", str(root.parent.parent), "--catalog", str(catalog),
                 "--base-url", mock_llm.base_url, "--model", "mock"]) == 0
    return catalog


def test_graph_build_resolves_entities_across_documents(tmp_path, mock_llm):
    catalog = _index_a_small_graph_corpus(tmp_path, mock_llm)
    assert main(["graph", "build", "--catalog", str(catalog)]) == 0

    graph_dir = tmp_path / "graph"
    nodes = [json.loads(l) for l in (graph_dir / "nodes.jsonl").read_text().splitlines()]
    edges = [json.loads(l) for l in (graph_dir / "edges.jsonl").read_text().splitlines()]
    acme = next(n for n in nodes if n["name"] in ("ACME Corp", "Acme Corporation"))
    assert set(acme["aliases"]) | {acme["name"]} == {"ACME Corp", "Acme Corporation"}
    assert len(acme["doc_ids"]) == 2         # mentioned in both documents
    assert len(edges) == 3


def test_graph_build_output_dir_follows_config(tmp_path, mock_llm):
    catalog = _index_a_small_graph_corpus(tmp_path, mock_llm)
    custom = tmp_path / "somewhere_else"
    assert main(["graph", "build", "--catalog", str(catalog), "--out", str(custom)]) == 0
    assert (custom / "nodes.jsonl").exists()
    assert not (tmp_path / "graph").exists()


def test_graph_build_without_communities_flag(tmp_path, mock_llm):
    catalog = _index_a_small_graph_corpus(tmp_path, mock_llm)
    assert main(["graph", "build", "--catalog", str(catalog), "--no-communities"]) == 0
    communities = (tmp_path / "graph" / "communities.jsonl").read_text()
    assert communities == ""


def test_graph_build_on_an_empty_catalog_is_a_clear_error(tmp_path):
    catalog = tmp_path / "empty.jsonl"
    with pytest.raises(SystemExit, match="missing or empty"):
        main(["graph", "build", "--catalog", str(catalog)])


def test_graph_build_on_a_catalog_with_no_relations_reports_nothing_to_build(
        tmp_path, mock_llm, capsys):
    """A plain `generic`-profiled document still has `entities` (a core field),
    so nodes alone prove nothing — it's zero *edges* that means no profile in
    this catalog has a `graph:` block doing real relation extraction."""
    root = tmp_path / "misc"
    root.mkdir()
    (root / "note.txt").write_text("A plain note with no particular structure.",
                                   encoding="utf-8")
    catalog = tmp_path / "catalog.jsonl"
    assert main(["index", "--root", str(root), "--catalog", str(catalog),
                 "--base-url", mock_llm.base_url, "--model", "mock"]) == 0
    assert main(["graph", "build", "--catalog", str(catalog)]) == 0
    out = capsys.readouterr().out
    assert "0 edges" in out
    assert "Nothing to build yet" in out


def test_graph_stats_reads_back_what_build_wrote(tmp_path, mock_llm, capsys):
    catalog = _index_a_small_graph_corpus(tmp_path, mock_llm)
    assert main(["graph", "build", "--catalog", str(catalog)]) == 0
    capsys.readouterr()   # discard `build`'s "Wrote ..." line
    assert main(["graph", "stats", "--catalog", str(catalog)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["nodes"] == 3 and report["edges"] == 3


def test_graph_stats_before_build_is_a_clear_error(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(json.dumps({"doc_id": "1", "status": "ok"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no graph at"):
        main(["graph", "stats", "--catalog", str(catalog)])


def test_graph_build_with_summarize_makes_one_call_per_community(tmp_path, mock_llm):
    catalog = _index_a_small_graph_corpus(tmp_path, mock_llm)
    mock_llm.reply = None
    calls_before = mock_llm.call_count
    assert main(["graph", "build", "--catalog", str(catalog),
                 "--base-url", mock_llm.base_url, "--model", "mock", "--summarize"]) == 0
    communities = [json.loads(l) for l in
                  (tmp_path / "graph" / "communities.jsonl").read_text().splitlines()]
    assert communities and all(c["summary"] for c in communities)
    assert mock_llm.call_count == calls_before + len(communities)


def test_graph_config_section_drives_a_flagless_build(tmp_path, mock_llm, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOCCAT_CONFIG", raising=False)
    catalog = _index_a_small_graph_corpus(tmp_path, mock_llm)
    (tmp_path / "doccat.yaml").write_text(f"""
paths:
  catalog: {catalog}
graph:
  out_dir: {tmp_path}/configured_graph
  min_community_size: 10
""", encoding="utf-8")
    assert main(["graph", "build"]) == 0
    assert (tmp_path / "configured_graph" / "nodes.jsonl").exists()
    communities = (tmp_path / "configured_graph" / "communities.jsonl").read_text()
    assert communities == ""   # the 3-member cluster doesn't meet min_community_size=10
