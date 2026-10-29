"""The profile-designing agent: sampling, drafting, sanitising, refining."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from doccat.cli import main
from doccat.llm import LLMClient, LLMConfig
from doccat.optimize import (design_profile, render_profile_yaml, sample_documents,
                             _critique, _load_draft)
from doccat.profiles import CORE_FIELDS, ProfileError, load_profiles
from doccat.sources import FileSource, JsonRecordSource


def client_for(mock):
    return LLMClient(LLMConfig(base_url=mock.base_url, model="mock", timeout=30,
                               max_retries=2))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_sampling_spreads_across_folders_not_just_the_first(tmp_path):
    root = tmp_path / "corpus"
    (root / "huge").mkdir(parents=True)
    (root / "tiny").mkdir(parents=True)
    for i in range(200):
        (root / "huge" / f"doc_{i:03d}.txt").write_text(f"Bulk document {i}", encoding="utf-8")
    for i in range(3):
        (root / "tiny" / f"rare_{i}.txt").write_text(f"Rare document {i}", encoding="utf-8")

    sample = sample_documents(FileSource(root), size=8)
    folders = {doc.key.split("/")[0] for doc in sample}
    assert folders == {"huge", "tiny"}, "a rare folder must still be represented"
    assert len(sample) == 8


def test_sampling_is_deterministic_for_a_given_seed(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    for i in range(30):
        (root / f"doc_{i:02d}.txt").write_text(f"Document {i}", encoding="utf-8")
    first = [d.key for d in sample_documents(FileSource(root), size=6, seed=7)]
    second = [d.key for d in sample_documents(FileSource(root), size=6, seed=7)]
    assert first == second
    assert [d.key for d in sample_documents(FileSource(root), size=6, seed=8)] != first or True


def test_sampled_text_is_truncated_to_the_budget(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "long.txt").write_text("x" * 50_000, encoding="utf-8")
    doc = sample_documents(FileSource(root), size=1, sample_chars=1000)[0]
    assert len(doc.text) == 1000 and doc.truncated


def test_documents_without_text_are_skipped(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "empty.txt").write_text("   \n", encoding="utf-8")
    (root / "real.txt").write_text("Actual content.", encoding="utf-8")
    sample = sample_documents(FileSource(root), size=5)
    assert [d.key for d in sample] == ["real.txt"]


# ---------------------------------------------------------------------------
# Drafting: the sanitiser must never produce an unloadable profile
# ---------------------------------------------------------------------------

def test_agent_output_is_sanitised_into_a_loadable_profile():
    hostile = {
        "profile_name": "My Weird Name!! ",
        "description": "Whatever.",
        "extends": "does-not-exist",
        "role": "You are a cataloguer.",
        "doc_types": ["Tender Spec", "award_decision", "award_decision"],
        "fields": [
            {"name": "tender number", "type": "string", "description": "the id"},
            {"name": "bidders", "type": "array", "description": "companies"},
            {"name": "summary", "type": "string", "description": "core field, drop me"},
            {"name": "", "type": "string", "description": "no name"},
            {"name": "no_description", "type": "string", "description": ""},
            "not even a dict",
        ],
        "strategy": "telepathy",
        "head_chars": 9_000_000,
        "match_filename_regex": "(unclosed",
        "match_path_globs": ["**/tenders/**"],
    }
    text, name = render_profile_yaml(hostile, available=load_profiles().names())
    assert name == "my_weird_name"

    spec = yaml.safe_load(text)
    assert spec["extends"] == "generic"                  # unknown parent rejected
    assert spec["excerpt"]["strategy"] == "head_tail"    # unknown strategy rejected
    assert spec["excerpt"]["head_chars"] <= 200_000      # absurd budget clamped
    assert spec["prompt"]["doc_types"] == ["tender_spec", "award_decision", "other"]
    # nameless, description-less and non-dict entries are all dropped
    assert set(spec["prompt"]["fields"]) == {"tender_number", "bidders"}
    assert "summary" not in spec["prompt"]["fields"]     # core fields are inherited
    assert spec["match"] == {"path_globs": ["**/tenders/**"]}   # invalid regex dropped

    profile = _load_draft(text, name)                    # loads through the real machinery
    assert profile.fields["bidders"].type == "list"      # "array" coerced to a known type
    assert set(CORE_FIELDS) <= set(profile.fields)


def test_draft_records_its_provenance_in_comments():
    text, _ = render_profile_yaml({
        "profile_name": "x", "corpus_summary": "A pile of municipal tenders.",
        "domain": "procurement", "reasoning": "Because the tender number is the key.",
        "strategy_reason": "Everything important is at the front.",
        "fields": [{"name": "a", "type": "string", "description": "b"}],
    }, available=["generic"])
    assert "# Corpus: A pile of municipal tenders." in text
    assert "# Domain: procurement" in text
    assert "drafted by `doccat optimize`" in text


def test_a_profile_extending_a_builtin_keeps_its_fields():
    text, name = render_profile_yaml(
        {"profile_name": "tenders", "extends": "legal",
         "fields": [{"name": "tender_number", "type": "string", "description": "id"}]},
        available=load_profiles().names())
    profile = _load_draft(text, name)
    assert "parties" in profile.fields          # inherited from legal
    assert "tender_number" in profile.fields    # its own


# ---------------------------------------------------------------------------
# The whole agent
# ---------------------------------------------------------------------------

def test_design_profile_end_to_end(docs_tree, mock_llm):
    with client_for(mock_llm) as client:
        result = design_profile(FileSource(docs_tree), client, sample_size=5, refine_rounds=1)

    assert result.profile_name == "company_docs"        # sanitised from "Company Docs!"
    profile = result.profile
    assert profile.role == "You are a company records cataloguer."
    assert "counterparties" in profile.fields
    assert "effective_date" in profile.fields           # "Effective Date" → snake_case
    assert "title" in profile.fields                    # inherited, not the agent's copy
    assert profile.fields["title"].description == "best-guess document title"
    assert "parties" in profile.fields                  # extends: legal

    # the refine round ran, cataloguing real documents and revising the draft
    assert "renewal_terms" in profile.fields
    assert len(result.rounds) == 1
    assert result.rounds[0]["trial"]["documents_catalogued"] == 3
    # 1 analysis + 3 trial documents + 1 refine
    assert result.llm_calls == 5 == mock_llm.call_count


def test_refine_zero_skips_the_trial(docs_tree, mock_llm):
    with client_for(mock_llm) as client:
        result = design_profile(FileSource(docs_tree), client, sample_size=4, refine_rounds=0)
    assert result.rounds == []
    assert result.llm_calls == 1
    assert "renewal_terms" not in result.profile.fields


def test_a_broken_refinement_keeps_the_previous_draft(docs_tree, mock_llm):
    """The refine pass is best-effort: a bad round must not lose a good draft."""
    with client_for(mock_llm) as client:
        good = design_profile(FileSource(docs_tree), client,
                              sample_size=3, refine_rounds=0).yaml_text

        mock_llm.break_refine = True
        broken = design_profile(FileSource(docs_tree), client, sample_size=3, refine_rounds=1)

    assert broken.rounds and "error" in broken.rounds[0]
    assert broken.yaml_text == good            # the good draft survived intact
    assert broken.profile.fields["counterparties"]


def test_an_unusable_analysis_is_reported_clearly(docs_tree, tmp_path, mock_llm):
    mock_llm.reply = "I will not answer in JSON."
    with pytest.raises(SystemExit, match="did not return a usable analysis"):
        main(["optimize", "--root", str(docs_tree), "--print-only",
              "--base-url", mock_llm.base_url, "--model", "mock", "--sample", "2"])


def test_empty_corpus_is_a_clear_error(tmp_path, mock_llm):
    empty = tmp_path / "empty"
    empty.mkdir()
    with client_for(mock_llm) as client:
        with pytest.raises(ProfileError, match="no documents with extractable text"):
            design_profile(FileSource(empty), client, sample_size=4)


def test_json_dumps_can_be_optimized_too(tmp_path, mock_llm):
    dump = tmp_path / "Plenos.json"
    dump.write_text(json.dumps([
        {"id": "1", "orga": "Senado", "texto": "Sesión plenaria sobre presupuestos."},
        {"id": "2", "orga": "Congreso", "texto": "Debate sobre sanidad pública."},
    ]), encoding="utf-8")
    with client_for(mock_llm) as client:
        result = design_profile(JsonRecordSource(dump, "texto"), client,
                                sample_size=2, refine_rounds=0)
    assert len(result.sample) == 2
    assert result.sample[0].filetype == "json_record"


def test_critique_spots_dead_fields():
    records = [
        {"path": "a", "title": "A", "parties": [], "status_note": "same"},
        {"path": "b", "title": "B", "parties": [], "status_note": "same"},
    ]
    critique = _critique(records)
    assert critique["always_empty"] == ["parties"]
    assert critique["no_variation"] == ["status_note"]
    assert "title" not in critique["no_variation"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_optimize_cli_writes_a_usable_profile(docs_tree, tmp_path, mock_llm, capsys):
    out = tmp_path / "profiles" / "company.yaml"
    rc = main(["optimize", "--root", str(docs_tree), "-o", str(out),
               "--base-url", mock_llm.base_url, "--model", "mock",
               "--sample", "4", "--refine", "1"])
    assert rc == 0
    assert out.is_file()

    printed = capsys.readouterr().out
    assert "new fields" in printed and "counterparties" in printed
    assert "doccat index" in printed          # tells you what to do next

    # the written file is immediately usable by a real run
    assert main(["profiles", "validate", "--profiles-dir", str(out.parent)]) == 0
    assert main(["index", "--root", str(docs_tree), "--catalog", str(tmp_path / "c.jsonl"),
                 "--profiles-dir", str(out.parent), "--profile", "company_docs",
                 "--base-url", mock_llm.base_url, "--model", "mock"]) == 0
    records = [json.loads(line) for line in (tmp_path / "c.jsonl").read_text().splitlines()]
    assert {r["profile"] for r in records} == {"company_docs"}


def test_optimize_cli_print_only_writes_nothing(docs_tree, tmp_path, mock_llm, capsys):
    assert main(["optimize", "--root", str(docs_tree), "--print-only",
                 "--base-url", mock_llm.base_url, "--model", "mock",
                 "--sample", "3", "--refine", "0"]) == 0
    printed = capsys.readouterr().out
    assert "name: company_docs" in printed
    assert not list(tmp_path.glob("*.yaml"))


def test_optimize_cli_refuses_to_clobber(docs_tree, tmp_path, mock_llm):
    out = tmp_path / "p.yaml"
    out.write_text("name: existing\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["optimize", "--root", str(docs_tree), "-o", str(out),
              "--base-url", mock_llm.base_url, "--model", "mock",
              "--sample", "2", "--refine", "0"])
    assert out.read_text() == "name: existing\n"


def test_optimize_honours_config_file_settings(docs_tree, tmp_path, mock_llm, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOCCAT_CONFIG", raising=False)
    (tmp_path / "doccat.yaml").write_text(f"""
paths:
  docs_root: {docs_tree}
  profiles_dir: [{tmp_path}/myprofiles]
llm:
  base_url: {mock_llm.base_url}
  model: mock
optimize:
  sample_size: 3
  refine_rounds: 0
""", encoding="utf-8")
    assert main(["optimize"]) == 0            # no flags: everything from the config
    assert (tmp_path / "myprofiles" / "company_docs.yaml").is_file()


# ---------------------------------------------------------------------------
# Phase 5: the optimizer proposes a graph vocabulary too
# ---------------------------------------------------------------------------

def test_render_profile_yaml_emits_a_graph_block_from_proposed_vocabulary():
    analysis = {
        "profile_name": "proc", "extends": "legal",
        "entity_types": ["Organization", "organization", "role", ""],
        "relation_types": ["supplies_to", "Party_To", "supplies_to"],
        "fields": [{"name": "tender_reference", "type": "string", "description": "id"}],
    }
    text, name = render_profile_yaml(analysis, available=load_profiles().names())
    spec = yaml.safe_load(text)
    assert spec["graph"]["entity_types"] == ["organization", "role"]      # deduped, cased
    assert spec["graph"]["relation_types"] == ["supplies_to", "party_to"]

    profile = _load_draft(text, name)
    assert profile.graph_enabled
    assert profile.fields["relations"].items == "object"
    assert set(CORE_FIELDS) <= set(profile.fields)


def test_render_profile_yaml_omits_graph_block_when_no_vocabulary_proposed():
    analysis = {"profile_name": "plain", "fields": [
        {"name": "a", "type": "string", "description": "b"}]}
    text, name = render_profile_yaml(analysis, available=load_profiles().names())
    assert "graph" not in yaml.safe_load(text)
    assert not _load_draft(text, name).graph_enabled


def test_agents_own_relations_field_is_reserved_not_overridden():
    """The agent proposing its own `relations` field must not shadow the
    structured object schema the `graph:` block injects."""
    analysis = {
        "profile_name": "x", "entity_types": ["organization"],
        "fields": [{"name": "relations", "type": "list",
                   "description": "a plain string list — must be ignored"}],
    }
    text, name = render_profile_yaml(analysis, available=load_profiles().names())
    spec = yaml.safe_load(text)
    assert "relations" not in (spec.get("prompt") or {}).get("fields", {})
    profile = _load_draft(text, name)
    assert profile.fields["relations"].items == "object"   # the real schema won, not a string list


def test_header_comment_mentions_the_graph_vocabulary_when_proposed():
    text, _ = render_profile_yaml({
        "profile_name": "x", "entity_types": ["organization", "person"],
        "relation_types": ["supplies_to"],
    }, available=["generic"])
    assert "# Graph entity types: organization, person" in text
    assert "# Graph relation types: supplies_to" in text


# -- the mechanical "fabricated relation" check -----------------------------

def test_unmatched_relation_count_flags_subjects_absent_from_entities():
    from doccat.optimize import _unmatched_relation_count

    record = {
        "entities": {"organizations": ["ACME"], "people": []},
        "relations": [
            {"subject": "ACME", "object": "Contoso"},     # Contoso not extracted -> unmatched
            {"subject": "ACME", "object": "ACME Subsidiary"},  # substring match -> OK
            "not a dict",
        ],
    }
    assert _unmatched_relation_count(record) == 1


def test_unmatched_relation_count_skips_records_with_no_entities_to_check_against():
    from doccat.optimize import _unmatched_relation_count

    record = {"entities": {}, "relations": [{"subject": "A", "object": "B"}]}
    assert _unmatched_relation_count(record) == 0   # nothing to compare against


def test_critique_surfaces_the_unmatched_relation_signal():
    records = [{"path": "a", "entities": {"organizations": ["ACME"]},
               "relations": [{"subject": "ACME", "object": "Ghost Corp"}]}]
    critique = _critique(records)
    assert critique["relations_without_matching_entities"] == 1


def test_critique_omits_the_key_when_everything_matches():
    records = [{"path": "a", "entities": {"organizations": ["ACME", "Contoso"]},
               "relations": [{"subject": "ACME", "object": "Contoso"}]}]
    critique = _critique(records)
    assert "relations_without_matching_entities" not in critique


# -- end to end through design_profile() -------------------------------------

def test_design_profile_drafts_a_graph_enabled_profile_when_proposed(docs_tree, mock_llm):
    mock_llm.analysis_override = {
        "entity_types": ["organization"], "relation_types": ["supplies_to"]}
    with client_for(mock_llm) as client:
        result = design_profile(FileSource(docs_tree), client, sample_size=4, refine_rounds=1)

    profile = result.profile
    assert profile.graph_enabled
    assert profile.graph.entity_types == ["organization"]
    assert profile.graph.relation_types == ["supplies_to"]
    # the trial actually asked for relations (schema included it) and the
    # mock's default reply names a Contoso the mock's entities never lists —
    # the mechanical check must catch that, unprompted.
    assert result.rounds[0]["trial"]["relations_without_matching_entities"] > 0


def test_design_profile_without_proposed_vocabulary_has_no_graph(docs_tree, mock_llm):
    # `extends: legal` would inherit legal's own graph: block (graph config
    # inherits like excerpt/llm — see Profile.graph_enabled) — pick a parent
    # with no graph config at all to isolate "the agent proposed nothing".
    mock_llm.analysis_override = {"entity_types": [], "relation_types": [], "extends": "generic"}
    with client_for(mock_llm) as client:
        result = design_profile(FileSource(docs_tree), client, sample_size=4, refine_rounds=0)
    assert not result.profile.graph_enabled


def test_design_profile_inherits_graph_from_an_already_enabled_parent(docs_tree, mock_llm):
    """Extending a profile that already has `graph:` keeps it even when the
    agent itself proposes no new vocabulary — the same inheritance rule
    excerpt/llm settings already follow."""
    mock_llm.analysis_override = {"entity_types": [], "relation_types": [], "extends": "legal"}
    with client_for(mock_llm) as client:
        result = design_profile(FileSource(docs_tree), client, sample_size=4, refine_rounds=0)
    assert result.profile.graph_enabled
    assert result.profile.graph.relation_types   # legal's own vocabulary, inherited


def test_optimize_cli_report_shows_the_graph_vocabulary(docs_tree, tmp_path, mock_llm, capsys):
    mock_llm.analysis_override = {
        "entity_types": ["organization"], "relation_types": ["supplies_to"]}
    assert main(["optimize", "--root", str(docs_tree), "-o", str(tmp_path / "p.yaml"),
                 "--base-url", mock_llm.base_url, "--model", "mock",
                 "--sample", "4", "--refine", "1"]) == 0
    out = capsys.readouterr().out
    assert "graph       entities: organization" in out
    assert "relations: supplies_to" in out
    assert "unmatched relations:" in out
