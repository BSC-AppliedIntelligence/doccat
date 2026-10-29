"""Corpus graph consolidation: entity resolution, traversal, communities, persistence."""

from __future__ import annotations

import json

import pytest

from doccat.graph import (Community, GraphEdge, GraphNode, build_corpus_graph,
                          load_corpus_graph, normalize_entity_name,
                          save_corpus_graph, summarize_communities)
from doccat.llm import LLMClient, LLMConfig


# ---------------------------------------------------------------------------
# Entity-name normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variants", [
    ["ACME Corp.", "Acme Corporation", "ACME CORP", "  acme corp  "],
    ["Contoso S.L.", "Contoso SL", "contoso s.l."],
    ["Globex Inc", "Globex Inc.", "GLOBEX INC"],
])
def test_normalize_collapses_known_variants(variants):
    keys = {normalize_entity_name(v) for v in variants}
    assert len(keys) == 1, f"expected one key, got {keys} for {variants}"


def test_normalize_keeps_genuinely_different_names_apart():
    assert normalize_entity_name("ACME Corp") != normalize_entity_name("ACME Holdings")
    assert normalize_entity_name("Ana Pérez") != normalize_entity_name("Ana García")


def test_normalize_does_not_collapse_a_name_to_nothing():
    # "Corp" alone must not strip to an empty key just because it matches the suffix.
    assert normalize_entity_name("Corp") == "corp"


# ---------------------------------------------------------------------------
# Building the graph: resolution, edge merging, degree
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_records():
    return [
        {"doc_id": "d1", "entities": {"organizations": ["ACME Corp.", "Contoso"], "people": []},
         "relations": [{"subject": "ACME Corp.", "relation": "supplies_to", "object": "Contoso",
                        "description": "ACME supplies widgets.", "strength": 0.9}]},
        {"doc_id": "d2",
         "entities": {"organizations": ["Acme Corporation", "Globex"], "people": ["Ana"]},
         "relations": [{"subject": "Acme Corporation", "relation": "supplies_to",
                        "object": "Contoso", "description": "Confirmed again.", "strength": 0.8},
                       {"subject": "Globex", "relation": "competes_with",
                        "object": "Acme Corporation", "strength": 0.5}]},
        {"doc_id": "d3", "entities": {"organizations": [], "people": ["Ana"]}, "relations": []},
    ]


def test_entity_mentions_resolve_across_documents(sample_records):
    graph, stats = build_corpus_graph(sample_records)
    assert stats.documents_scanned == 3
    assert stats.nodes == 4   # acme, contoso, globex, ana
    acme = graph.nodes[normalize_entity_name("ACME Corp.")]
    assert set(acme.doc_ids) == {"d1", "d2"}
    assert acme.name in ("ACME Corp.", "Acme Corporation")   # most common spelling
    assert set(acme.aliases) == {"ACME Corp.", "Acme Corporation"} - {acme.name}
    assert acme.type == "organization"


def test_duplicate_relations_merge_into_one_edge_with_combined_support(sample_records):
    graph, _ = build_corpus_graph(sample_records)
    acme_key, contoso_key = normalize_entity_name("ACME Corp."), normalize_entity_name("Contoso")
    matches = [e for e in graph.edges if e.source == acme_key and e.target == contoso_key]
    assert len(matches) == 1
    edge = matches[0]
    assert set(edge.doc_ids) == {"d1", "d2"}
    assert edge.weight == pytest.approx(1.7)
    assert edge.description == "ACME supplies widgets."   # first non-empty description kept


def test_relations_register_entities_missing_from_the_entities_field():
    # Globex appears only in a relation, never in `entities` — still gets a node.
    records = [{"doc_id": "d1", "entities": {}, "relations": [
        {"subject": "Mystery Co", "relation": "owns", "object": "Shell LLC", "strength": 1.0}]}]
    graph, stats = build_corpus_graph(records)
    assert stats.nodes == 2
    assert normalize_entity_name("Mystery Co") in graph.nodes
    assert normalize_entity_name("Shell LLC") in graph.nodes


def test_self_relations_are_dropped_as_resolution_artifacts():
    records = [{"doc_id": "d1", "entities": {}, "relations": [
        {"subject": "ACME Corp.", "relation": "owns", "object": "Acme Corporation",
         "strength": 1.0}]}]
    graph, stats = build_corpus_graph(records)
    assert stats.self_relations_dropped == 1
    assert graph.edges == []


def test_malformed_relations_are_skipped_not_fatal():
    records = [{"doc_id": "d1", "entities": {}, "relations": [
        "not a dict", {"subject": "A"}, {"subject": "", "object": "B"},
        {"subject": "A", "object": "B", "relation": "x", "strength": "not-a-number"},
    ]}]
    graph, stats = build_corpus_graph(records)
    assert stats.nodes == 2            # only the one well-formed relation registers A, B
    assert len(graph.edges) == 1
    assert graph.edges[0].weight == 1.0   # bad strength falls back to the default


def test_degree_counts_edges_touching_a_node(sample_records):
    graph, _ = build_corpus_graph(sample_records)
    acme = graph.nodes[normalize_entity_name("ACME Corp.")]
    ana = graph.nodes[normalize_entity_name("Ana")]
    assert acme.degree == 2     # supplies_to Contoso, competes_with (incoming from Globex)
    assert ana.degree == 0      # never appears in a relation


def test_empty_corpus_yields_an_empty_graph():
    graph, stats = build_corpus_graph([])
    assert len(graph) == 0 and graph.edges == [] and graph.communities == []
    assert stats.documents_scanned == 0


# ---------------------------------------------------------------------------
# Communities
# ---------------------------------------------------------------------------

def test_interconnected_entities_form_a_community(sample_records):
    graph, stats = build_corpus_graph(sample_records)
    assert stats.communities >= 1
    acme_key = normalize_entity_name("ACME Corp.")
    community = graph.community_of(acme_key)
    assert community is not None
    assert normalize_entity_name("Contoso") in community.members
    assert normalize_entity_name("Globex") in community.members


def test_isolated_node_belongs_to_no_community(sample_records):
    graph, _ = build_corpus_graph(sample_records)
    ana_key = normalize_entity_name("Ana")
    assert graph.community_of(ana_key) is None   # degree 0 — never grouped


def test_detect_communities_false_skips_detection_entirely(sample_records):
    graph, stats = build_corpus_graph(sample_records, detect_communities=False)
    assert graph.communities == [] and stats.communities == 0


def test_min_community_size_filters_small_clusters():
    records = [{"doc_id": "d1", "entities": {}, "relations": [
        {"subject": "A", "object": "B", "relation": "x", "strength": 1.0}]}]
    graph, _ = build_corpus_graph(records, min_community_size=3)
    assert graph.communities == []   # the 2-member {A, B} cluster doesn't meet the bar


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------

def test_find_matches_canonical_name_alias_and_substring(sample_records):
    graph, _ = build_corpus_graph(sample_records)
    assert graph.find("acme")[0].id == normalize_entity_name("ACME Corp.")
    assert graph.find("ACME Corp.")[0].id == normalize_entity_name("ACME Corp.")   # via alias
    assert graph.find("ont")[0].name == "Contoso"       # substring
    assert graph.find("no such entity") == []
    assert graph.find("") == []


def test_find_respects_limit(sample_records):
    graph, _ = build_corpus_graph(sample_records)
    assert len(graph.find("a", limit=2)) <= 2


def test_neighbors_is_direction_agnostic_and_hop_bounded(sample_records):
    graph, _ = build_corpus_graph(sample_records)
    acme_key = normalize_entity_name("ACME Corp.")
    result = graph.neighbors(acme_key, hops=1)
    names = {n["name"] for n in result["neighbors"]}
    assert names == {"Contoso", "Globex"}    # Globex->Acme (incoming) still counts
    assert len(result["edges"]) == 2


def test_neighbors_of_unknown_node_is_empty(sample_records):
    graph, _ = build_corpus_graph(sample_records)
    assert graph.neighbors("does-not-exist") == {"node": None, "neighbors": [], "edges": []}


def test_shortest_path_crosses_relation_direction(sample_records):
    graph, _ = build_corpus_graph(sample_records)
    globex_key, contoso_key = normalize_entity_name("Globex"), normalize_entity_name("Contoso")
    result = graph.shortest_path(globex_key, contoso_key)
    assert [n["name"] for n in result["path"]] == ["Globex", "Acme Corporation", "Contoso"]
    assert len(result["edges"]) == 2


def test_shortest_path_returns_none_when_unreachable(sample_records):
    graph, _ = build_corpus_graph(sample_records)
    ana_key, acme_key = normalize_entity_name("Ana"), normalize_entity_name("ACME Corp.")
    assert graph.shortest_path(ana_key, acme_key) is None     # Ana has no edges at all


def test_shortest_path_with_unknown_node_returns_none(sample_records):
    graph, _ = build_corpus_graph(sample_records)
    assert graph.shortest_path("nope", normalize_entity_name("ACME Corp.")) is None


def test_overview_reports_types_and_relation_counts(sample_records):
    graph, _ = build_corpus_graph(sample_records)
    overview = graph.overview()
    assert overview["nodes"] == 4 and overview["edges"] == 2
    assert overview["by_entity_type"]["organization"] == 3
    assert overview["top_relations"]["supplies_to"] == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_save_and_load_round_trips_everything(sample_records, tmp_path):
    graph, _ = build_corpus_graph(sample_records)
    save_corpus_graph(graph, tmp_path)
    assert (tmp_path / "nodes.jsonl").exists()
    assert (tmp_path / "edges.jsonl").exists()
    assert (tmp_path / "communities.jsonl").exists()

    reloaded = load_corpus_graph(tmp_path)
    assert len(reloaded.nodes) == len(graph.nodes)
    assert len(reloaded.edges) == len(graph.edges)
    assert len(reloaded.communities) == len(graph.communities)
    acme_key = normalize_entity_name("ACME Corp.")
    assert reloaded.nodes[acme_key].name == graph.nodes[acme_key].name
    assert reloaded.find("acme")[0].aliases == graph.nodes[acme_key].aliases
    # the reloaded graph is independently traversable (NetworkX view rebuilt from the files)
    assert reloaded.neighbors(acme_key, hops=1)["neighbors"]


def test_load_on_a_missing_directory_is_an_empty_graph_not_an_error(tmp_path):
    graph = load_corpus_graph(tmp_path / "does_not_exist_yet")
    assert len(graph) == 0 and graph.edges == [] and graph.communities == []


def test_load_skips_corrupt_lines_with_a_warning(tmp_path, caplog):
    (tmp_path / "nodes.jsonl").write_text(
        '{"id": "ok", "name": "OK"}\n{"id": broken\n{"id": "ok2", "name": "OK2"}\n',
        encoding="utf-8")
    graph = load_corpus_graph(tmp_path)
    assert set(graph.nodes) == {"ok", "ok2"}


def test_jsonl_files_are_plain_and_greppable(sample_records, tmp_path):
    graph, _ = build_corpus_graph(sample_records)
    save_corpus_graph(graph, tmp_path)
    lines = (tmp_path / "edges.jsonl").read_text(encoding="utf-8").splitlines()
    assert all(json.loads(line) for line in lines)   # every line is standalone valid JSON
    assert any("supplies_to" in line for line in lines)   # human-greppable


# ---------------------------------------------------------------------------
# Community summaries (LLM-backed)
# ---------------------------------------------------------------------------

def test_summarize_communities_fills_in_summary_text(sample_records, mock_llm):
    mock_llm.reply = "ACME, Contoso and Globex form a supplier-competitor cluster."
    graph, _ = build_corpus_graph(sample_records)
    assert graph.communities and graph.communities[0].summary == ""

    with LLMClient(LLMConfig(base_url=mock_llm.base_url, model="mock", timeout=30)) as client:
        done = summarize_communities(graph, client)

    assert done == len(graph.communities)
    assert graph.communities[0].summary == (
        "ACME, Contoso and Globex form a supplier-competitor cluster.")


def test_summarize_communities_degrades_gracefully_on_failure(sample_records, mock_llm):
    mock_llm.reply = "I will not answer in JSON."  # irrelevant content; complete() still succeeds
    graph, _ = build_corpus_graph(sample_records)

    class _BreakingClient:
        def complete(self, *a, **kw):
            raise __import__("doccat.llm", fromlist=["LLMError"]).LLMError("endpoint is down")

    done = summarize_communities(graph, _BreakingClient())
    assert done == 0
    assert all(c.summary == "" for c in graph.communities)


def test_summarize_communities_with_no_communities_is_a_noop(mock_llm):
    graph, _ = build_corpus_graph([])
    with LLMClient(LLMConfig(base_url=mock_llm.base_url, model="mock", timeout=30)) as client:
        assert summarize_communities(graph, client) == 0
