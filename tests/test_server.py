"""MCP server tools, exercised directly against a catalog on disk.

The tool functions are called as plain Python: the MCP transport is the SDK's
job, the retrieval logic is ours. The real FastMCP is used when the SDK is
installed; when it is not, a minimal stub stands in so the retrieval logic is
still covered.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


def _ensure_fastmcp() -> bool:
    """Make `mcp.server.fastmcp` importable. Returns True if it is the real SDK."""
    try:
        import mcp.server.fastmcp  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - absent, or an inconsistent install
        class _FastMCP:
            def __init__(self, *a, **kw):
                self.settings = types.SimpleNamespace(host="0.0.0.0", port=8000)

            def tool(self, *a, **kw):
                return lambda fn: fn

            def custom_route(self, *a, **kw):
                return lambda fn: fn

            def run(self, *a, **kw):
                raise NotImplementedError

        for name in ("mcp", "mcp.server", "mcp.server.fastmcp"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["mcp.server.fastmcp"].FastMCP = _FastMCP
        return False


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    """Import `server`, pointed at a small two-shard catalog."""
    real_sdk = _ensure_fastmcp()
    print(f"\n(server tests using {'the real MCP SDK' if real_sdk else 'a FastMCP stub'})")

    tmp = tmp_path_factory.mktemp("srv")
    catalog_a = tmp / "shard_0.jsonl"
    catalog_b = tmp / "shard_1.jsonl"
    docs = tmp / "docs"
    (docs / "contracts").mkdir(parents=True)
    (docs / "contracts" / "acme.txt").write_text("Supply agreement with ACME.", encoding="utf-8")

    catalog_a.write_text("\n".join(json.dumps(r) for r in [
        {"doc_id": "1", "path": "contracts/acme.txt", "filename": "acme.txt",
         "filetype": "txt", "status": "ok", "profile": "legal", "doc_type": "contract",
         "title": "ACME supply agreement", "summary": "Supply of widgets by ACME.",
         "description": "Governs widget supply.", "keywords": ["widgets", "supply"],
         "topics": ["procurement"], "questions_answered": ["Who supplies widgets?"],
         "entities": {"organizations": ["ACME"]},
         "parties": ["ACME Corp (supplier)", "Contoso (client)"],
         "expiry_date": "2026-12-31", "governing_law": "Spanish law"},
        {"doc_id": "2", "path": "manuals/hr_portal.txt", "filename": "hr_portal.txt",
         "filetype": "txt", "status": "ok", "profile": "technical", "doc_type": "user_manual",
         "title": "HR portal manual", "summary": "How to use the HR portal.",
         "description": "Covers leave requests.", "keywords": ["hr", "leave"],
         "topics": ["hr"], "questions_answered": ["How do I request leave?"],
         "entities": {}, "systems": ["HR Portal v4"], "error_codes": ["E-403"]},
    ]) + "\n", encoding="utf-8")
    catalog_b.write_text(json.dumps(
        {"doc_id": "3", "path": "plenos/ds_2024_01.txt", "filename": "ds_2024_01.txt",
         "filetype": "txt", "status": "ok", "profile": "parliamentary",
         "doc_type": "plenary_transcript", "title": "Sesión plenaria 12",
         "summary": "Debate on the budget.", "description": "Budget debate.",
         "keywords": ["presupuesto"], "topics": ["budget"], "entities": {},
         "questions_answered": ["What was debated?"],
         "chamber": "Congreso de los Diputados", "session_date": "2024-03-01",
         "speakers": ["Ana Pérez (GP Socialista)"]}) + "\n", encoding="utf-8")

    import os
    os.environ["CATALOG_PATH"] = f"{catalog_a},{catalog_b}"
    os.environ["DOCS_ROOT"] = str(docs)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import server
    server.CATALOG_PATHS = [catalog_a, catalog_b]
    server.CATALOG_PATH = catalog_a
    server.DOCS_ROOT = docs.resolve()
    server._catalog_mtimes = {}
    return server


def test_several_catalogs_are_merged(srv):
    overview = json.loads(srv.catalog_overview())
    assert overview["total_documents"] == 3
    assert overview["by_profile"] == {"legal": 1, "technical": 1, "parliamentary": 1}


def test_overview_advertises_profile_specific_fields(srv):
    fields = json.loads(srv.catalog_overview())["profile_fields"]
    assert "parties" in fields["legal"] and "expiry_date" in fields["legal"]
    assert "systems" in fields["technical"]
    assert "chamber" in fields["parliamentary"]


def test_profile_fields_are_keyword_searchable(srv):
    """A field a profile invented must be findable without a server change."""
    hits = json.loads(srv.search_documents(srv.SearchInput(query="Contoso")))
    assert hits["total_matches"] == 1
    assert hits["results"][0]["doc_id"] == "1"

    hits = json.loads(srv.search_documents(srv.SearchInput(query="E-403")))
    assert hits["results"][0]["doc_id"] == "2"


def test_search_can_be_filtered_by_profile(srv):
    hits = json.loads(srv.search_documents(srv.SearchInput(query="budget", profile="legal")))
    assert hits["total_matches"] == 0
    hits = json.loads(srv.search_documents(
        srv.SearchInput(query="budget", profile="parliamentary")))
    assert hits["total_matches"] == 1


def test_search_by_field_exact_and_listing(srv):
    hits = json.loads(srv.search_by_field(srv.FieldQueryInput(field="parties", value="ACME")))
    assert hits["total_matches"] == 1
    assert hits["results"][0]["parties"] == ["ACME Corp (supplier)", "Contoso (client)"]

    # no value: every document that has the field at all
    hits = json.loads(srv.search_by_field(srv.FieldQueryInput(field="chamber")))
    assert hits["total_matches"] == 1
    assert hits["results"][0]["chamber"] == "Congreso de los Diputados"


def test_search_by_field_reports_available_fields_when_empty(srv):
    hits = json.loads(srv.search_by_field(srv.FieldQueryInput(field="nonexistent")))
    assert hits["total_matches"] == 0
    assert "parties" in hits["available_fields"]


def test_list_documents_filters_by_profile(srv):
    out = json.loads(srv.list_documents(srv.ListInput(profile="technical")))
    assert out["total"] == 1 and out["documents"][0]["profile"] == "technical"


def test_summary_view_carries_the_profile(srv):
    out = json.loads(srv.list_documents(srv.ListInput()))
    assert {d["profile"] for d in out["documents"]} == {"legal", "technical", "parliamentary"}


def test_read_document_reads_from_disk(srv):
    out = json.loads(srv.read_document(srv.ReadInput(doc_id="1")))
    assert "ACME" in out["text"] and out["source"] == "live_extraction"


def test_catalog_spec_parsing(srv, tmp_path):
    (tmp_path / "a.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "b.jsonl").write_text("", encoding="utf-8")
    import os
    got = srv._catalog_paths(f"{tmp_path}/a.jsonl,{tmp_path}/b.jsonl")
    assert [p.name for p in got] == ["a.jsonl", "b.jsonl"]
    got = srv._catalog_paths(str(tmp_path / "*.jsonl"))
    assert sorted(p.name for p in got) == ["a.jsonl", "b.jsonl"]


# ---------------------------------------------------------------------------
# Graph tools — traversal over a corpus graph built independently of the
# catalog fixture above (the server loads the two from separate files, same
# as it does in production).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def graph_srv(srv, tmp_path_factory):
    """`srv`, plus a small, HAND-BUILT corpus graph wired into `server.GRAPH_DIR`.

    ACME -supplies_to-> Contoso <-supplies_to- Globex -owns-> Umbrella
    ACME -competes_with-> Globex
    Ana: isolated, no relations at all (no community, no neighbors).

    Built directly from the dataclasses rather than through
    `build_corpus_graph` + real Louvain detection: these are tests of the
    *server tools* (does graph_communities paginate correctly, does
    graph_path report "not connected" correctly, ...), not of clustering —
    that already has its own thorough, algorithm-aware coverage in
    test_graph.py. A real community-detection pass is free to legitimately
    decide this small a graph splits into two tighter clusters rather than
    one big one; pinning a specific Community here keeps this fixture about
    what it's actually testing.
    """
    from doccat.graph import Community, CorpusGraph, GraphEdge, GraphNode, save_corpus_graph

    nodes = {
        "acme": GraphNode(id="acme", name="ACME Corp", aliases=["Acme Corporation"],
                          type="organization", doc_ids=["g1", "g2"], degree=2),
        "contoso": GraphNode(id="contoso", name="Contoso", type="organization",
                             doc_ids=["g1", "g2"], degree=2),
        "globex": GraphNode(id="globex", name="Globex", type="organization",
                            doc_ids=["g2", "g3"], degree=3),
        "umbrella": GraphNode(id="umbrella", name="Umbrella Corp", type="organization",
                              doc_ids=["g3"], degree=1),
        "ana": GraphNode(id="ana", name="Ana", type="person", doc_ids=["g4"], degree=0),
    }
    edges = [
        GraphEdge(source="acme", target="contoso", relation="supplies_to",
                 description="ACME supplies widgets to Contoso.", doc_ids=["g1"], weight=0.9),
        GraphEdge(source="globex", target="contoso", relation="supplies_to",
                 description="Globex supplies gadgets to Contoso.", doc_ids=["g2"], weight=0.8),
        GraphEdge(source="acme", target="globex", relation="competes_with",
                 description="ACME and Globex are competitors.", doc_ids=["g2"], weight=0.5),
        GraphEdge(source="globex", target="umbrella", relation="owns",
                 description="Globex owns Umbrella.", doc_ids=["g3"], weight=0.7),
    ]
    communities = [Community(
        id="c0", members=["acme", "contoso", "globex", "umbrella"], size=4,
        summary="ACME, Contoso, Globex and Umbrella form one supplier and ownership cluster.")]
    graph = CorpusGraph(nodes=nodes, edges=edges, communities=communities)

    graph_dir = tmp_path_factory.mktemp("graph")
    save_corpus_graph(graph, graph_dir)

    srv.GRAPH_DIR = graph_dir
    srv._graph = None
    srv._graph_mtimes = {}
    return srv


def test_graph_overview_reports_counts_and_hint(graph_srv):
    overview = json.loads(graph_srv.graph_overview())
    assert overview["nodes"] == 5 and overview["edges"] == 4 and overview["communities"] == 1
    assert "supplies_to" in overview["top_relations"]
    assert "graph_search" in overview["hint"]


def test_graph_overview_without_a_built_graph_gives_a_clear_hint(srv, tmp_path):
    # `srv` is the `server` module itself — a process-wide singleton other
    # tests share via the module-scoped `graph_srv` fixture — so this must
    # restore every bit of state it touches, not just set it and move on.
    saved = (srv.GRAPH_DIR, srv._graph, srv._graph_mtimes)
    try:
        srv.GRAPH_DIR = tmp_path / "no_graph_here"
        srv._graph = None
        srv._graph_mtimes = {}
        result = srv.graph_overview()
        assert isinstance(result, str) and "doccat graph build" in result
        assert not result.startswith("{")   # not JSON — a plain hint message
    finally:
        srv.GRAPH_DIR, srv._graph, srv._graph_mtimes = saved


def test_catalog_overview_flags_whether_a_graph_exists(graph_srv):
    overview = json.loads(graph_srv.catalog_overview())
    assert overview["has_graph"] is True


def test_graph_search_matches_canonical_name_and_alias(graph_srv):
    hits = json.loads(graph_srv.graph_search(graph_srv.GraphSearchInput(query="acme")))
    assert hits["count"] == 1
    assert hits["results"][0]["name"] in ("ACME Corp", "Acme Corporation")
    assert set(hits["results"][0]["aliases"]) | {hits["results"][0]["name"]} == {
        "ACME Corp", "Acme Corporation"}

    empty = json.loads(graph_srv.graph_search(graph_srv.GraphSearchInput(query="nonexistent")))
    assert empty["count"] == 0 and "graph_overview" in empty["hint"]


def test_graph_neighbors_is_direction_agnostic_and_hop_bounded(graph_srv):
    one_hop = json.loads(graph_srv.graph_neighbors(
        graph_srv.GraphNeighborsInput(entity="ACME", hops=1)))
    names_1 = {n["name"] for n in one_hop["neighbors"]}
    assert names_1 == {"Contoso", "Globex"}            # Umbrella is 2 hops away

    two_hop = json.loads(graph_srv.graph_neighbors(
        graph_srv.GraphNeighborsInput(entity="ACME", hops=2)))
    names_2 = {n["name"] for n in two_hop["neighbors"]}
    assert names_2 == {"Contoso", "Globex", "Umbrella Corp"}
    assert "read_document" in two_hop["hint"]


def test_graph_neighbors_of_isolated_entity_is_empty_not_an_error(graph_srv):
    result = json.loads(graph_srv.graph_neighbors(
        graph_srv.GraphNeighborsInput(entity="Ana", hops=1)))
    assert result["node"]["name"] == "Ana"
    assert result["neighbors"] == [] and result["edges"] == []


def test_graph_neighbors_of_unresolvable_entity_is_an_error(graph_srv):
    result = graph_srv.graph_neighbors(graph_srv.GraphNeighborsInput(entity="Nonexistent Ltd"))
    assert result.startswith("Error:")


def test_graph_path_crosses_relation_direction(graph_srv):
    result = json.loads(graph_srv.graph_path(
        graph_srv.GraphPathInput(source="Umbrella", target="Contoso")))
    assert result["connected"] is True
    names = [n["name"] for n in result["path"]]
    assert names[0] == "Umbrella Corp" and names[-1] == "Contoso"
    assert len(result["edges"]) == len(names) - 1


def test_graph_path_reports_not_connected_explicitly(graph_srv):
    result = json.loads(graph_srv.graph_path(
        graph_srv.GraphPathInput(source="Ana", target="ACME")))
    assert result["connected"] is False
    assert result["source"]["name"] == "Ana"
    assert "not" in result["hint"].lower() or "No chain" in result["hint"]


def test_graph_path_with_unresolvable_entity_names_which_side_failed(graph_srv):
    result = graph_srv.graph_path(graph_srv.GraphPathInput(source="Nope", target="ACME"))
    assert result.startswith("Error (source):")
    result = graph_srv.graph_path(graph_srv.GraphPathInput(source="ACME", target="Nope"))
    assert result.startswith("Error (target):")


def test_graph_communities_lists_clusters_with_summaries(graph_srv):
    result = json.loads(graph_srv.graph_communities(graph_srv.GraphCommunitiesInput()))
    assert result["total"] == 1
    community = result["communities"][0]
    assert community["size"] == 4
    assert "supplier and ownership cluster" in community["summary"]
    assert set(community["members"]) == {"ACME Corp", "Contoso", "Globex", "Umbrella Corp"} or \
           set(community["members"]) == {"Acme Corporation", "Contoso", "Globex", "Umbrella Corp"}


def test_graph_communities_looks_up_one_entity(graph_srv):
    result = json.loads(graph_srv.graph_communities(
        graph_srv.GraphCommunitiesInput(entity="Contoso")))
    assert result["entity"]["name"] == "Contoso"
    assert result["community"]["size"] == 4


def test_graph_communities_reports_isolation_not_an_error(graph_srv):
    result = json.loads(graph_srv.graph_communities(
        graph_srv.GraphCommunitiesInput(entity="Ana")))
    assert result["community"] is None
    assert "isolated" in result["hint"] or "not" in result["hint"].lower()


def test_graph_communities_pagination(graph_srv):
    result = json.loads(graph_srv.graph_communities(
        graph_srv.GraphCommunitiesInput(limit=1, offset=1)))
    assert result["total"] == 1 and result["count"] == 0 and not result["has_more"]


def test_load_graph_hot_reloads_on_the_same_cadence_as_the_catalog(graph_srv):
    """The /health route (exercised at the HTTP layer in the manual MCP smoke
    test, not here) reports exactly these three numbers — this is the
    function it calls to get them."""
    g = graph_srv._load_graph()
    assert len(g) == 5 and len(g.edges) == 4 and len(g.communities) == 1
