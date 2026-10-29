"""Corpus-level graph consolidation: entity resolution, traversal, communities.

A profile's `graph:` block (`doccat/profiles.py`) controls what gets
extracted **per document** — document-local entities and relations, named
exactly as that one document spells them. This module is the separate,
re-runnable step that reads every document's local names and resolves them
into **one graph across the whole corpus**: the same "ACME Corp" mentioned in
fifty documents collapses into a single node with fifty supporting
`doc_id`s, not fifty disconnected mini-graphs. See docs/GRAPH.md, Decisions 3
and 4, for the reasoning behind every choice here.

    from doccat.graph import build_corpus_graph, save_corpus_graph

    records = load_catalog(Path("catalog.jsonl")).values()
    graph = build_corpus_graph(records)
    save_corpus_graph(graph, Path("graph"))

Persistence is plain JSONL — `graph/nodes.jsonl`, `graph/edges.jsonl`,
`graph/communities.jsonl` — greppable and diffable like `catalog.jsonl`
itself, not an opaque store. The `networkx.MultiDiGraph` used for traversal
is rebuilt from those files on load, never persisted directly, so the
on-disk format stays stable even if the in-memory graph library ever changes.

Entity resolution is tier 1 only (normalized exact match) — see
`normalize_entity_name`. A fuzzy, embedding-based tier 2 is a documented
future upgrade in docs/GRAPH.md, not implemented here.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from .llm import LLMClient, LLMError

log = logging.getLogger("doccat.graph")

__all__ = ["GraphNode", "GraphEdge", "Community", "CorpusGraph",
           "build_corpus_graph", "load_corpus_graph", "save_corpus_graph",
           "summarize_communities", "normalize_entity_name", "GraphBuildStats"]

#: `entities` bucket name (as the `generic` profile already produces it) -> a
#: node type label. `dates` is included for completeness, though dates rarely
#: appear as the subject/object of a stated relation in practice.
_ENTITY_BUCKET_TYPES = {
    "people": "person", "organizations": "organization",
    "places": "place", "dates": "date",
}

#: Legal-entity suffixes stripped during normalization, word-bounded so a
#: company genuinely named e.g. "Co" in full is not silently mangled by a
#: substring match. Deliberately narrow — see docs/GRAPH.md Decision 3: tier 1
#: is meant to be cheap and safe, not exhaustive.
_SUFFIX_RE = re.compile(
    r"\b(s\.?a\.?|s\.?l\.?u?\.?|ltd|llc|inc|corp|corporation|gmbh|co|plc)\b\.?",
    re.IGNORECASE)
_PUNCT_RE = re.compile(r"[.,;:()\"'’`]")
_SPACE_RE = re.compile(r"\s+")


def normalize_entity_name(name: str) -> str:
    """Tier-1 entity-resolution key: case-fold, strip punctuation and common
    legal-entity suffixes, collapse whitespace. Deterministic and free —
    "ACME Corp.", "acme corporation" and "ACME CORP" all resolve to the same
    node; genuinely different names stay different.
    """
    text = _PUNCT_RE.sub("", name.strip().lower())
    stripped = _SUFFIX_RE.sub("", text)
    stripped = _SPACE_RE.sub(" ", stripped).strip()
    # Don't let suffix-stripping collapse a name to nothing (e.g. a document
    # that literally names an entity "Corp").
    return stripped or _SPACE_RE.sub(" ", text).strip() or name.strip().lower()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """A resolved, corpus-level entity — the merge of every document-local
    mention that normalized to the same key."""

    id: str
    name: str                                    # canonical display name
    aliases: list[str] = field(default_factory=list)
    type: str = ""                                # person | organization | place | date | ""
    doc_ids: list[str] = field(default_factory=list)
    degree: int = 0                                # filled in after the graph is built

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "aliases": sorted(set(self.aliases)),
                "type": self.type, "doc_ids": sorted(set(self.doc_ids)),
                "degree": self.degree}

    @classmethod
    def from_dict(cls, d: dict) -> "GraphNode":
        return cls(id=d["id"], name=d.get("name", d["id"]),
                   aliases=list(d.get("aliases") or []), type=d.get("type", ""),
                   doc_ids=list(d.get("doc_ids") or []), degree=int(d.get("degree", 0)))


@dataclass
class GraphEdge:
    """A resolved relation, merged across every document that stated it with
    the same (source, target, relation) triple."""

    source: str                                    # GraphNode.id
    target: str                                    # GraphNode.id
    relation: str
    description: str = ""
    doc_ids: list[str] = field(default_factory=list)
    weight: float = 1.0                             # accumulated strength/support

    def to_dict(self) -> dict:
        return {"source": self.source, "target": self.target, "relation": self.relation,
                "description": self.description, "doc_ids": sorted(set(self.doc_ids)),
                "weight": round(self.weight, 4)}

    @classmethod
    def from_dict(cls, d: dict) -> "GraphEdge":
        return cls(source=d["source"], target=d["target"], relation=d.get("relation", ""),
                   description=d.get("description", ""), doc_ids=list(d.get("doc_ids") or []),
                   weight=float(d.get("weight", 1.0)))


@dataclass
class Community:
    """A detected cluster of densely-interrelated nodes, GraphRAG-style."""

    id: str
    members: list[str] = field(default_factory=list)   # GraphNode.id values
    size: int = 0
    summary: str = ""                                    # filled by `summarize_communities`

    def to_dict(self) -> dict:
        return {"id": self.id, "members": self.members, "size": self.size,
                "summary": self.summary}

    @classmethod
    def from_dict(cls, d: dict) -> "Community":
        members = list(d.get("members") or [])
        return cls(id=d["id"], members=members, size=int(d.get("size", len(members))),
                   summary=d.get("summary", ""))


@dataclass
class GraphBuildStats:
    """What happened during `build_corpus_graph`, for `doccat graph build`'s report."""

    documents_scanned: int = 0
    documents_with_entities: int = 0
    documents_with_relations: int = 0
    raw_entity_mentions: int = 0
    raw_relation_mentions: int = 0
    nodes: int = 0
    edges: int = 0
    communities: int = 0
    self_relations_dropped: int = 0


class CorpusGraph:
    """The resolved, corpus-level graph, plus a NetworkX view for traversal.

    `nodes` and `edges` are the plain-data truth — what gets persisted.
    `_nx` (a `networkx.MultiDiGraph`, several edges between the same pair of
    nodes are real and expected: "ACME party_to Contoso" and "ACME
    supplies_to Contoso" are both true at once) is rebuilt from them on
    construction, purely to make traversal queries convenient.
    """

    def __init__(self, nodes: dict[str, GraphNode] | None = None,
                 edges: list[GraphEdge] | None = None,
                 communities: list[Community] | None = None) -> None:
        self.nodes = nodes or {}
        self.edges = edges or []
        self.communities = communities or []
        self._nx = self._build_nx()

    def _build_nx(self) -> "nx.MultiDiGraph":
        g = nx.MultiDiGraph()
        for node in self.nodes.values():
            g.add_node(node.id)
        for edge in self.edges:
            if edge.source in self.nodes and edge.target in self.nodes:
                g.add_edge(edge.source, edge.target,
                          relation=edge.relation, weight=edge.weight)
        return g

    # -- lookups -------------------------------------------------------
    def find(self, query: str, limit: int = 10) -> list[GraphNode]:
        """Fuzzy entity search: the normalized key first (exact), then a
        case-insensitive substring match over every name and alias.
        """
        query = (query or "").strip()
        if not query:
            return []
        key = normalize_entity_name(query)
        exact = self.nodes.get(key)
        results = [exact] if exact else []
        needle = query.lower()
        for node in self.nodes.values():
            if node is exact or len(results) >= limit:
                continue
            haystack = " ".join([node.name, *node.aliases]).lower()
            if needle in haystack:
                results.append(node)
        return results[:limit]

    def neighbors(self, node_id: str, hops: int = 1) -> dict[str, Any]:
        """Everything connected to `node_id` within `hops`, direction-agnostic
        (a relation pointing either way still counts as "connected"), with
        the edges among the whole resulting set — not just edges touching the
        origin — so a 2-hop query also reveals structure between neighbors.
        """
        if node_id not in self.nodes:
            return {"node": None, "neighbors": [], "edges": []}
        seen = {node_id}
        frontier = {node_id}
        for _ in range(max(hops, 1)):
            nxt: set[str] = set()
            for n in frontier:
                nxt |= set(self._nx.successors(n)) | set(self._nx.predecessors(n))
            nxt -= seen
            if not nxt:
                break
            seen |= nxt
            frontier = nxt
        relevant = seen  # includes node_id itself
        edges_out = [e for e in self.edges if e.source in relevant and e.target in relevant]
        return {
            "node": self.nodes[node_id].to_dict(),
            "neighbors": [self.nodes[n].to_dict() for n in seen if n != node_id],
            "edges": [e.to_dict() for e in edges_out],
        }

    def shortest_path(self, source_id: str, target_id: str) -> dict[str, Any] | None:
        """Shortest explanatory path between two entities, or None if they're
        not connected. Direction-agnostic for pathfinding (a relation stated
        in either direction still connects the two entities); the edge
        actually stated is reported alongside each hop.
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        try:
            path_ids = nx.shortest_path(self._nx.to_undirected(as_view=True),
                                        source_id, target_id)
        except nx.NetworkXNoPath:
            return None
        hops = []
        for u, v in zip(path_ids, path_ids[1:]):
            match = next((e for e in self.edges if {e.source, e.target} == {u, v}), None)
            hops.append(match.to_dict() if match else
                       {"source": u, "target": v, "relation": "", "description": "",
                        "doc_ids": [], "weight": 0.0})
        return {"path": [self.nodes[n].to_dict() for n in path_ids], "edges": hops}

    def community_of(self, node_id: str) -> Community | None:
        return next((c for c in self.communities if node_id in c.members), None)

    def overview(self) -> dict[str, Any]:
        type_counts = Counter(n.type or "unknown" for n in self.nodes.values())
        relation_counts = Counter(e.relation for e in self.edges)
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "communities": len(self.communities),
            "by_entity_type": dict(type_counts.most_common()),
            "top_relations": dict(relation_counts.most_common(20)),
        }

    def __len__(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------------------
# Building the graph from a catalog
# ---------------------------------------------------------------------------

def _entity_mentions(record: dict) -> Iterable[tuple[str, str]]:
    """Yield (name, type) for every entity a record's `entities` field names."""
    entities = record.get("entities")
    if isinstance(entities, dict):
        for bucket, names in entities.items():
            etype = _ENTITY_BUCKET_TYPES.get(bucket, "")
            for name in (names or []):
                if isinstance(name, str) and name.strip():
                    yield name.strip(), etype


def _relation_mentions(record: dict) -> Iterable[dict]:
    """Yield each well-formed relation dict a record's `relations` field holds."""
    for rel in (record.get("relations") or []):
        if (isinstance(rel, dict) and isinstance(rel.get("subject"), str)
                and rel["subject"].strip() and isinstance(rel.get("object"), str)
                and rel["object"].strip()):
            yield rel


def build_corpus_graph(records: Iterable[dict], *, detect_communities: bool = True,
                       community_seed: int = 0,
                       min_community_size: int = 2) -> tuple[CorpusGraph, GraphBuildStats]:
    """Resolve every record's document-local entities/relations into one graph.

    Two passes: first collect every entity mention (from `entities`, and from
    any `relations` subject/object not already listed there — models don't
    always keep the two fields in perfect sync) and resolve them into nodes;
    then build edges, merging duplicate (source, target, relation) triples
    across documents into one edge with combined `doc_ids` and weight.
    """
    stats = GraphBuildStats()
    mentions: dict[str, list[tuple[str, str, str]]] = defaultdict(list)  # key -> [(doc_id, type, raw_name)]
    relation_records: list[tuple[str, dict]] = []

    for record in records:
        stats.documents_scanned += 1
        doc_id = record.get("doc_id", "")
        had_entities = had_relations = False

        for name, etype in _entity_mentions(record):
            had_entities = True
            stats.raw_entity_mentions += 1
            mentions[normalize_entity_name(name)].append((doc_id, etype, name))

        for rel in _relation_mentions(record):
            had_relations = True
            stats.raw_relation_mentions += 1
            relation_records.append((doc_id, rel))
            for role in ("subject", "object"):
                name = rel[role].strip()
                mentions[normalize_entity_name(name)].append((doc_id, "", name))

        stats.documents_with_entities += had_entities
        stats.documents_with_relations += had_relations

    nodes: dict[str, GraphNode] = {}
    for key, occurrences in mentions.items():
        name_counts = Counter(raw for _, _, raw in occurrences)
        canonical = name_counts.most_common(1)[0][0]
        type_counts = Counter(t for _, t, _ in occurrences if t)
        node_type = type_counts.most_common(1)[0][0] if type_counts else ""
        doc_ids = sorted({d for d, _, _ in occurrences if d})
        aliases = sorted({raw for _, _, raw in occurrences} - {canonical})
        nodes[key] = GraphNode(id=key, name=canonical, aliases=aliases,
                               type=node_type, doc_ids=doc_ids)

    edge_index: dict[tuple[str, str, str], GraphEdge] = {}
    for doc_id, rel in relation_records:
        source_key = normalize_entity_name(rel["subject"].strip())
        target_key = normalize_entity_name(rel["object"].strip())
        if source_key == target_key:
            # Almost always a resolution artifact (two spellings of the same
            # entity collapsing together) rather than a real self-relation.
            stats.self_relations_dropped += 1
            continue
        label = str(rel.get("relation") or "relates_to").strip() or "relates_to"
        try:
            strength = float(rel["strength"]) if rel.get("strength") is not None else 1.0
        except (TypeError, ValueError):
            strength = 1.0
        description = str(rel.get("description") or "").strip()

        edge_key = (source_key, target_key, label)
        existing = edge_index.get(edge_key)
        if existing is None:
            edge_index[edge_key] = GraphEdge(
                source=source_key, target=target_key, relation=label,
                description=description, doc_ids=[doc_id] if doc_id else [],
                weight=strength)
        else:
            if doc_id and doc_id not in existing.doc_ids:
                existing.doc_ids.append(doc_id)
            existing.weight += strength
            if not existing.description and description:
                existing.description = description

    edges = list(edge_index.values())
    degree: Counter[str] = Counter()
    for edge in edges:
        degree[edge.source] += 1
        degree[edge.target] += 1
    for key, node in nodes.items():
        node.degree = degree.get(key, 0)

    graph = CorpusGraph(nodes=nodes, edges=edges)
    stats.nodes = len(nodes)
    stats.edges = len(edges)

    if detect_communities and edges:
        try:
            raw_communities = nx.algorithms.community.louvain_communities(
                graph._nx.to_undirected(), weight="weight", seed=community_seed)
        except Exception as e:  # noqa: BLE001 - detection is best-effort, never fatal
            log.warning("Community detection failed (%s) — continuing without communities", e)
            raw_communities = []
        graph.communities = [
            Community(id=f"c{i}", members=sorted(members), size=len(members))
            for i, members in enumerate(raw_communities) if len(members) >= min_community_size
        ]
    stats.communities = len(graph.communities)
    return graph, stats


# ---------------------------------------------------------------------------
# Community summaries (GraphRAG-style; optional, one LLM call per community)
# ---------------------------------------------------------------------------

_COMMUNITY_SUMMARY_PROMPT = """You are summarising one cluster of related entities found in a \
document repository. The repository is catalogued without a vector store; an AI agent will \
read your summary to decide whether this cluster is relevant to a question, before opening \
any individual document.

Entities in this cluster: {entities}

Relationships among them, as stated in the source documents:
{relations}

Write a 2-4 sentence summary of what connects these entities and why they form one cluster. \
Base it strictly on the relationships shown — never invent a connection that isn't listed. \
Return ONLY the summary text: no preamble, no markdown, no quotation marks around it.
"""


def summarize_communities(graph: CorpusGraph, client: LLMClient, *,
                          max_members: int = 15, max_relations: int = 15) -> int:
    """One LLM call per community (2+ members) to generate its `summary`.

    Cheap in aggregate — there are far fewer communities than documents — and
    this is the one place `doccat graph build` needs an LLM at all; skip it
    (pass no client / don't call this) to build a purely structural graph.
    """
    done = 0
    for community in graph.communities:
        member_ids = community.members[:max_members]
        names = [graph.nodes[m].name for m in member_ids if m in graph.nodes]
        if not names:
            continue
        member_set = set(community.members)
        relevant_edges = [e for e in graph.edges
                         if e.source in member_set and e.target in member_set]
        rel_lines = []
        for edge in relevant_edges[:max_relations]:
            source = graph.nodes.get(edge.source)
            target = graph.nodes.get(edge.target)
            if source and target:
                line = f"- {source.name} {edge.relation} {target.name}"
                if edge.description:
                    line += f": {edge.description}"
                rel_lines.append(line)
        prompt = _COMMUNITY_SUMMARY_PROMPT.format(
            entities=", ".join(names),
            relations="\n".join(rel_lines) or "(no relations recorded between them)")
        try:
            community.summary = client.complete(prompt).strip()
            done += 1
        except LLMError as e:
            log.warning("Could not summarise community %s: %s", community.id, e)
            community.summary = ""
    return done


# ---------------------------------------------------------------------------
# Persistence — plain JSONL, atomic writes, tolerant reads
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("%s:%d is not valid JSON (%s) — skipping", path, lineno, e)
    return rows


def save_corpus_graph(graph: CorpusGraph, out_dir: Path) -> None:
    """Persist as `nodes.jsonl` / `edges.jsonl` / `communities.jsonl` under `out_dir`."""
    _write_jsonl(out_dir / "nodes.jsonl", (n.to_dict() for n in graph.nodes.values()))
    _write_jsonl(out_dir / "edges.jsonl", (e.to_dict() for e in graph.edges))
    _write_jsonl(out_dir / "communities.jsonl", (c.to_dict() for c in graph.communities))


def load_corpus_graph(out_dir: Path) -> CorpusGraph:
    """Load a graph persisted by `save_corpus_graph`. Missing files mean an empty graph,
    not an error — the server's graph tools rely on this to degrade gracefully."""
    nodes = {d["id"]: GraphNode.from_dict(d) for d in _read_jsonl(out_dir / "nodes.jsonl")}
    edges = [GraphEdge.from_dict(d) for d in _read_jsonl(out_dir / "edges.jsonl")]
    communities = [Community.from_dict(d) for d in _read_jsonl(out_dir / "communities.jsonl")]
    return CorpusGraph(nodes=nodes, edges=edges, communities=communities)
