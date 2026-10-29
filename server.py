#!/usr/bin/env python3
"""doc_catalog_mcp — MCP server exposing a document metadata catalog.

Serves the JSONL catalog produced by `indexer.py` over streamable HTTP so an
agent can (1) understand what the document collection contains, (2) find the
documents relevant to a user query via metadata search, and (3) read the
actual content of chosen documents on demand — no vector store required.

Configuration: environment variables (DOCS_ROOT, CATALOG_PATH, TEXT_CACHE,
MCP_HOST, MCP_PORT, MCP_STATELESS) or equivalent CLI flags (CLI wins):

    python server.py --docs-root /data/documents --catalog ./catalog.jsonl \
                     --port 8010        # listens on http://0.0.0.0:8010/mcp

CATALOG_PATH accepts several catalogs, separated by commas or os.pathsep, and
shell globs — so the shards produced by `doccat index --shard i/n` can be served
directly without merging them first:

    CATALOG_PATH='/data/catalogs/shard_*.jsonl' python server.py

Catalogs built with cataloguing profiles carry domain-specific fields (a
contract's `parties` and `expiry_date`, a session transcript's `speakers` and
`agenda_items`, ...). Those fields are searchable like any other metadata, and
`search_by_field` queries them directly.

A catalog built with a `graph:`-enabled profile (legal, parliamentary, ...)
also carries `relations` between entities. `doccat graph build` consolidates
those into one corpus-level graph (GRAPH_DIR, default: `graph/` next to the
catalog); this server reloads it the same way it reloads the catalog, and the
five `graph_*` tools below traverse it: who connects to whom, how, and through
which documents. A catalog with no graph built yet is served exactly as
before — the graph tools report that cleanly rather than erroring.

A plain-HTTP health check is served at GET /health (no MCP client needed).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("doc_catalog_mcp")

def _catalog_paths(spec: str) -> list[Path]:
    """Expand a catalog spec into paths: comma/os.pathsep separated, globs allowed."""
    paths: list[Path] = []
    for part in re.split(r"[,%s]" % re.escape(os.pathsep), spec):
        part = part.strip()
        if not part:
            continue
        if any(c in part for c in "*?["):
            matches = sorted(Path(m) for m in glob.glob(os.path.expanduser(part)))
            paths.extend(matches)            # a glob that matches nothing contributes nothing
        else:
            paths.append(Path(os.path.expanduser(part)))
    return paths or [Path("catalog.jsonl")]


DOCS_ROOT = Path(os.getenv("DOCS_ROOT", ".")).resolve()
CATALOG_PATHS = _catalog_paths(os.getenv("CATALOG_PATH", "catalog.jsonl"))
CATALOG_PATH = CATALOG_PATHS[0]   # the primary one, for messages and defaults
TEXT_CACHE = Path(os.getenv("TEXT_CACHE", str(CATALOG_PATH.resolve().parent / "extracted_text")))
GRAPH_DIR = Path(os.getenv("GRAPH_DIR", str(CATALOG_PATH.resolve().parent / "graph")))
HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("MCP_PORT", "8000"))
# Stateless mode: every request is self-contained (no session pinning).
# More robust behind proxies (mcpo, nginx) and across server restarts.
STATELESS = os.getenv("MCP_STATELESS", "true").strip().lower() in ("1", "true", "yes")

try:
    mcp = FastMCP("doc_catalog_mcp", host=HOST, port=PORT, stateless_http=STATELESS)
except TypeError:  # older mcp SDK without the stateless_http parameter
    mcp = FastMCP("doc_catalog_mcp", host=HOST, port=PORT)

# ---------------------------------------------------------------------------
# Catalog loading (reloaded automatically when the JSONL file changes on disk)
# ---------------------------------------------------------------------------

_catalog: dict[str, dict] = {}
_search_blobs: dict[str, str] = {}   # doc_id -> precomputed lowercase search text
_catalog_mtimes: dict[str, float] = {}

#: Bookkeeping fields: useful in a record, useless as search text.
_NON_SEARCHABLE = frozenset({
    "doc_id", "content_hash", "change_key", "indexed_at", "modified_at", "size_bytes",
    "char_count", "pages_or_paragraphs", "status", "extracted_text_path",
    "raw_reply_path", "error_detail", "extraction_error", "ingestion", "record_index",
    "incomplete_fields", "needs_ocr", "indexed_chunks", "filetype", "profile",
})


def _load_catalog() -> dict[str, dict]:
    """Load (and hot-reload) every configured catalog, merged by doc_id."""
    global _catalog, _search_blobs, _catalog_mtimes
    mtimes = {str(p): p.stat().st_mtime for p in CATALOG_PATHS if p.exists()}
    if mtimes == _catalog_mtimes:
        return _catalog

    catalog: dict[str, dict] = {}
    for path in CATALOG_PATHS:
        if not path.exists():
            continue
        loaded = 0
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning("%s:%d is not valid JSON (%s) — skipping", path, lineno, e)
                    continue
                if "doc_id" in rec:
                    catalog[rec["doc_id"]] = rec
                    loaded += 1
        log.info("Catalog loaded: %d documents from %s", loaded, path)
    _catalog, _catalog_mtimes = catalog, mtimes
    _search_blobs = {doc_id: _searchable_text(rec) for doc_id, rec in catalog.items()}
    if len(CATALOG_PATHS) > 1:
        log.info("Serving %d documents from %d catalog(s)", len(catalog), len(CATALOG_PATHS))
    return _catalog


# ---------------------------------------------------------------------------
# Corpus graph loading (reloaded automatically when `doccat graph build`
# rewrites it on disk — same hot-reload posture as the catalog)
# ---------------------------------------------------------------------------

_graph = None            # a doccat.graph.CorpusGraph, or None until first loaded/built
_graph_mtimes: dict[str, float] = {}
_GRAPH_FILES = ("nodes.jsonl", "edges.jsonl", "communities.jsonl")


def _load_graph():
    """Load (and hot-reload) the corpus graph built by `doccat graph build`.

    Returns an empty graph — never None, never raises — when nothing has been
    built yet, so every graph_* tool can check `len(graph) == 0` rather than
    handling a missing-graph case separately from an empty one.
    """
    global _graph, _graph_mtimes
    from doccat.graph import load_corpus_graph

    files = [GRAPH_DIR / name for name in _GRAPH_FILES]
    mtimes = {str(p): p.stat().st_mtime for p in files if p.exists()}
    if _graph is not None and mtimes == _graph_mtimes:
        return _graph

    graph = load_corpus_graph(GRAPH_DIR)
    _graph, _graph_mtimes = graph, mtimes
    if len(graph):
        log.info("Corpus graph loaded: %d nodes, %d edges, %d communities from %s",
                 len(graph), len(graph.edges), len(graph.communities), GRAPH_DIR)
    return _graph


def _flatten(value) -> str:
    """Flatten a metadata value (string, list, or nested dict) into search text."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_flatten(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    return ""


def _searchable_text(rec: dict) -> str:
    """Flatten every meaningful metadata field, including profile-specific ones.

    Profiles add domain fields (`parties`, `speakers`, `error_codes`, ...) that
    differ per document family, so the blob is built generically rather than
    from a fixed list — a new profile's fields become searchable with no server
    change.
    """
    parts = [rec.get("filetype", "")]
    for key, value in rec.items():
        if key in _NON_SEARCHABLE:
            continue
        parts.append(_flatten(value))
    return " ".join(p for p in parts if p).lower()


def _score(rec: dict, terms: list[str]) -> float:
    """Simple weighted term-frequency score across metadata fields."""
    score = 0.0
    title = (rec.get("title") or "").lower()
    keywords = " ".join(rec.get("keywords") or []).lower()
    body = _search_blobs.get(rec["doc_id"]) or _searchable_text(rec)
    for term in terms:
        if term in title:
            score += 3.0
        if term in keywords:
            score += 2.0
        score += body.count(term) * 0.5
    return score


def _summary_view(rec: dict) -> dict:
    """Compact record for search/list results (keeps agent context small)."""
    return {
        "doc_id": rec["doc_id"],
        "path": rec.get("path"),
        "title": rec.get("title") or rec.get("filename"),
        "doc_type": rec.get("doc_type"),
        "summary": rec.get("summary"),
        "topics": rec.get("topics"),
        "profile": rec.get("profile"),
        "pages_or_paragraphs": rec.get("pages_or_paragraphs"),
    }


def _profile_fields(catalog: dict[str, dict]) -> dict[str, list[str]]:
    """Which extra metadata fields each profile actually produced in this catalog.

    Lets an agent discover, at runtime, that contracts carry `parties` and
    `expiry_date` while session transcripts carry `speakers` and `agenda_items`
    — without the server knowing anything about the profiles themselves.
    """
    common = {"doc_id", "path", "filename", "filetype", "size_bytes", "modified_at",
              "content_hash", "change_key", "pages_or_paragraphs", "char_count",
              "indexed_at", "status", "profile", "ingestion", "extracted_text_path",
              "title", "summary", "description", "doc_type", "language", "keywords",
              "topics", "entities", "questions_answered"}
    found: dict[str, set[str]] = {}
    for rec in catalog.values():
        name = rec.get("profile") or "(none)"
        extra = {k for k, v in rec.items() if k not in common and v not in (None, "", [], {})}
        found.setdefault(name, set()).update(extra)
    return {name: sorted(fields) for name, fields in sorted(found.items()) if fields}


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class SearchInput(BaseModel):
    """Input for catalog search."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(..., min_length=1, max_length=500,
                       description="Keywords describing what you need, e.g. 'supplier contracts 2024 penalties'")
    doc_type: Optional[str] = Field(default=None, description="Filter by document type, e.g. 'contract', 'report'")
    topic: Optional[str] = Field(default=None, description="Filter by topic (substring match)")
    profile: Optional[str] = Field(default=None,
                                   description="Filter by cataloguing profile, e.g. 'legal', "
                                               "'parliamentary' (see catalog_overview)")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class ListInput(BaseModel):
    """Input for listing catalog entries."""
    model_config = ConfigDict(extra="forbid")

    doc_type: Optional[str] = Field(default=None, description="Filter by document type")
    path_prefix: Optional[str] = Field(default=None, description="Filter by relative path prefix, e.g. 'finance/2024'")
    profile: Optional[str] = Field(default=None, description="Filter by cataloguing profile")
    limit: int = Field(default=25, ge=1, le=100, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class FieldQueryInput(BaseModel):
    """Input for a direct query on one metadata field."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    field: str = Field(..., min_length=1, max_length=64,
                       description="Metadata field to query, e.g. 'parties', 'expiry_date', "
                                   "'speakers', 'systems', 'authors'. Call catalog_overview "
                                   "to see which fields the catalog actually has.")
    value: Optional[str] = Field(default=None, max_length=300,
                                 description="Value to look for (case-insensitive substring). "
                                             "Omit to list every document that has this field set.")
    profile: Optional[str] = Field(default=None, description="Restrict to one cataloguing profile")
    limit: int = Field(default=25, ge=1, le=100, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class ReadInput(BaseModel):
    """Input for reading document content."""
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(..., description="Document id from search/list results")
    first_page: int = Field(default=1, ge=1, description="First page to read (PDF only, 1-based)")
    last_page: Optional[int] = Field(default=None, ge=1, description="Last page to read inclusive (PDF only). Default: 10 pages from first_page")
    max_chars: int = Field(default=20000, ge=100, le=200000, description="Truncate returned text to this many characters")


class GraphSearchInput(BaseModel):
    """Input for fuzzy entity lookup in the corpus graph."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(..., min_length=1, max_length=300,
                       description="Entity name to look for, e.g. 'ACME' — matches the "
                                   "canonical name, any alias, or a substring of either")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")


class GraphNeighborsInput(BaseModel):
    """Input for exploring what one entity connects to."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    entity: str = Field(..., min_length=1, max_length=300,
                        description="Entity name, as it appears in a document or in "
                                    "graph_search results — resolved by fuzzy match")
    hops: int = Field(default=1, ge=1, le=3,
                      description="How many relationship steps to expand outward")


class GraphPathInput(BaseModel):
    """Input for finding how two entities are connected."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    source: str = Field(..., min_length=1, max_length=300, description="First entity name")
    target: str = Field(..., min_length=1, max_length=300, description="Second entity name")


class GraphCommunitiesInput(BaseModel):
    """Input for listing detected entity clusters, or finding one entity's cluster."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    entity: Optional[str] = Field(default=None, max_length=300,
                                  description="Entity name — return its cluster instead of "
                                              "listing all of them")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum clusters to list")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="catalog_overview",
    annotations={"title": "Collection Overview", "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def catalog_overview() -> str:
    """Get a high-level overview of the document collection.

    Call this FIRST to understand what the corpus contains before searching.

    Returns:
        str: JSON with total document count, counts by file type and doc_type,
        the 30 most frequent topics, and catalog freshness info.
    """
    catalog = _load_catalog()
    if not catalog:
        return f"Error: catalog is empty or missing at {CATALOG_PATH}. Run indexer.py first."
    topics = Counter(t for r in catalog.values() for t in (r.get("topics") or []))
    by_profile = Counter(r.get("profile") or "(none)" for r in catalog.values())
    needing_ocr = [r.get("path") for r in catalog.values()
                   if r.get("needs_ocr") or r.get("status") == "needs_ocr"]
    overview = {
        "total_documents": len(catalog),
        "by_filetype": dict(Counter(r.get("filetype", "?") for r in catalog.values())),
        "by_doc_type": dict(Counter(r.get("doc_type", "unknown") for r in catalog.values())),
        "top_topics": dict(topics.most_common(30)),
        "by_profile": dict(by_profile),
        "profile_fields": _profile_fields(catalog),
        "documents_without_text": sum(1 for r in catalog.values() if r.get("status") == "empty"),
        "documents_needing_ocr": needing_ocr[:50],
        "documents_needing_ocr_count": len(needing_ocr),
        "last_indexed_at": max((r.get("indexed_at", "") for r in catalog.values()), default=None),
        "has_graph": bool(len(_load_graph())),
        "hint": ("Documents are catalogued under profiles; each profile adds its own metadata "
                 "fields (see profile_fields). Use search_documents for keyword search, "
                 "search_by_field to query one of those fields directly, then read_document. "
                 "If has_graph is true, graph_overview shows the entities and relations "
                 "extracted across the corpus — use it for 'how are X and Y connected' "
                 "questions, which keyword search answers poorly."),
    }
    return json.dumps(overview, ensure_ascii=False, indent=2)


@mcp.tool(
    name="search_documents",
    annotations={"title": "Search Document Catalog", "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def search_documents(params: SearchInput) -> str:
    """Search the metadata catalog for documents relevant to a query.

    Matches keywords against titles, summaries, keywords, topics, entities and
    example questions. Use several specific terms for best results, then call
    read_document on the top hits to extract actual content.

    Args:
        params (SearchInput): query, optional doc_type/topic filters, limit, offset.

    Returns:
        str: JSON: {"total_matches": int, "count": int, "offset": int,
        "results": [{"doc_id", "path", "title", "doc_type", "summary",
        "topics", "relevance_score", ...}]} sorted by relevance.
    """
    catalog = _load_catalog()
    if not catalog:
        return f"Error: catalog is empty or missing at {CATALOG_PATH}. Run indexer.py first."
    terms = [t for t in re.split(r"\W+", params.query.lower()) if len(t) > 1]
    if not terms:
        return "Error: query contained no usable search terms."

    hits = []
    for rec in catalog.values():
        if params.doc_type and (rec.get("doc_type") or "").lower() != params.doc_type.lower():
            continue
        if params.topic and not any(params.topic.lower() in t.lower() for t in (rec.get("topics") or [])):
            continue
        if params.profile and (rec.get("profile") or "").lower() != params.profile.lower():
            continue
        s = _score(rec, terms)
        if s > 0:
            hits.append((s, rec))
    hits.sort(key=lambda x: x[0], reverse=True)

    page = hits[params.offset:params.offset + params.limit]
    results = [{**_summary_view(r), "relevance_score": round(s, 2)} for s, r in page]
    return json.dumps({
        "total_matches": len(hits),
        "count": len(results),
        "offset": params.offset,
        "has_more": len(hits) > params.offset + len(results),
        "results": results,
        "hint": "Use read_document(doc_id=...) to read the content of a result." if results
                else "No matches. Try broader or different terms, or call catalog_overview for available topics.",
    }, ensure_ascii=False, indent=2)


@mcp.tool(
    name="list_documents",
    annotations={"title": "List Documents", "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def list_documents(params: ListInput) -> str:
    """Browse catalog entries with optional filters (no relevance ranking).

    Useful for exploring a subfolder or a document type after catalog_overview.

    Args:
        params (ListInput): optional doc_type and path_prefix filters, limit, offset.

    Returns:
        str: JSON with total, count, offset, has_more and compact document records.
    """
    catalog = _load_catalog()
    if not catalog:
        return f"Error: catalog is empty or missing at {CATALOG_PATH}. Run indexer.py first."
    records = sorted(catalog.values(), key=lambda r: r.get("path", ""))
    if params.doc_type:
        records = [r for r in records if (r.get("doc_type") or "").lower() == params.doc_type.lower()]
    if params.path_prefix:
        records = [r for r in records if (r.get("path") or "").startswith(params.path_prefix)]
    if params.profile:
        records = [r for r in records
                   if (r.get("profile") or "").lower() == params.profile.lower()]
    page = records[params.offset:params.offset + params.limit]
    return json.dumps({
        "total": len(records),
        "count": len(page),
        "offset": params.offset,
        "has_more": len(records) > params.offset + len(page),
        "documents": [_summary_view(r) for r in page],
    }, ensure_ascii=False, indent=2)


@mcp.tool(
    name="search_by_field",
    annotations={"title": "Query One Metadata Field", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def search_by_field(params: FieldQueryInput) -> str:
    """Find documents by the value of one specific metadata field.

    Cataloguing profiles give each document family its own structured fields —
    contracts get `parties`, `expiry_date`, `governing_law`; session transcripts
    get `speakers`, `agenda_items`, `session_date`; manuals get `systems`,
    `error_codes`. Use this when the question names such a field ("which
    contracts involve ACME?", "which manuals cover error E-403?") instead of
    hoping keyword search happens to surface it.

    Call catalog_overview first to see which fields exist in this catalog.

    Args:
        params (FieldQueryInput): field name, optional value substring, optional
            profile filter, limit, offset.

    Returns:
        str: JSON with total_matches and results, each carrying the matched
        field's value alongside the compact document record.
    """
    catalog = _load_catalog()
    if not catalog:
        return f"Error: catalog is empty or missing at {CATALOG_PATH}. Run the indexer first."

    needle = (params.value or "").lower()
    hits = []
    for rec in catalog.values():
        if params.profile and (rec.get("profile") or "").lower() != params.profile.lower():
            continue
        if params.field not in rec:
            continue
        value = rec[params.field]
        if value in (None, "", [], {}):
            continue
        if needle and needle not in _flatten(value).lower():
            continue
        hits.append((rec, value))

    if not hits:
        available = sorted({f for fields in _profile_fields(catalog).values() for f in fields})
        return json.dumps({
            "total_matches": 0,
            "results": [],
            "available_fields": available[:80],
            "hint": f"No document has {params.field!r}"
                    + (f" containing {params.value!r}." if params.value else ".")
                    + " Check available_fields, or use search_documents for keyword search.",
        }, ensure_ascii=False, indent=2)

    hits.sort(key=lambda h: h[0].get("path") or "")
    page = hits[params.offset:params.offset + params.limit]
    return json.dumps({
        "total_matches": len(hits),
        "count": len(page),
        "offset": params.offset,
        "has_more": len(hits) > params.offset + len(page),
        "field": params.field,
        "results": [{**_summary_view(rec), params.field: value} for rec, value in page],
    }, ensure_ascii=False, indent=2)


@mcp.tool(
    name="get_document_metadata",
    annotations={"title": "Get Full Document Metadata", "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def get_document_metadata(doc_id: str) -> str:
    """Get the complete catalog record for one document.

    Includes full summary, all keywords, entities (people, organizations,
    places, dates) and example questions the document can answer.

    Args:
        doc_id (str): Document id from search_documents or list_documents.

    Returns:
        str: JSON catalog record, or an error message if the id is unknown.
    """
    catalog = _load_catalog()
    rec = catalog.get(doc_id)
    if not rec:
        return f"Error: no document with doc_id '{doc_id}'. Use search_documents or list_documents to find valid ids."
    return json.dumps(rec, ensure_ascii=False, indent=2)


@mcp.tool(
    name="read_document",
    annotations={"title": "Read Document Content", "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def read_document(params: ReadInput) -> str:
    """Extract and return the actual text content of a document.

    This is how you get answer context: after search_documents identifies
    relevant documents, read them here. For long PDFs read a page range
    (default 10 pages) and paginate; DOCX files are returned whole up to
    max_chars.

    Args:
        params (ReadInput): doc_id, optional first_page/last_page (PDF, 1-based
            inclusive), max_chars truncation limit.

    Returns:
        str: JSON: {"doc_id", "path", "title", "total_pages", "pages_read",
        "truncated": bool, "text": str} or an error message.
    """
    catalog = _load_catalog()
    rec = catalog.get(params.doc_id)
    if not rec:
        return f"Error: no document with doc_id '{params.doc_id}'. Use search_documents to find valid ids."

    def _read_cache() -> str | None:
        """Fallback: extracted-text copy kept by the indexer (Markdown + front matter)."""
        cache_rel = rec.get("extracted_text_path")
        if cache_rel:
            cache_file = (TEXT_CACHE / cache_rel).resolve()
            if str(cache_file).startswith(str(TEXT_CACHE.resolve())) and cache_file.exists():
                return cache_file.read_text(encoding="utf-8")
        return None

    path = (DOCS_ROOT / rec["path"]).resolve()
    if not str(path).startswith(str(DOCS_ROOT)):
        return "Error: resolved path escapes the documents root."

    text = total = None
    pages_read = "all"
    source = "live_extraction"
    if path.exists():
        from extractors import extract_text
        try:
            if rec.get("filetype") == "pdf":
                first = params.first_page - 1
                last = (params.last_page - 1) if params.last_page else first + 9
                text, total = extract_text(path, first_page=first, last_page=last)
                pages_read = f"{params.first_page}-{min(last + 1, total)}"
            else:
                text, total = extract_text(path)
        except Exception as e:
            text = _read_cache()
            source = "text_cache"
            if text is None:
                return f"Error: extraction failed for {rec['path']}: {type(e).__name__}: {e}"
    else:
        text = _read_cache()
        source = "text_cache"
        if text is None:
            return (f"Error: file missing on disk and no cached text: {rec['path']}. "
                    "Re-run the indexer to refresh the catalog.")

    truncated = len(text) > params.max_chars
    return json.dumps({
        "doc_id": params.doc_id,
        "path": rec["path"],
        "title": rec.get("title") or rec.get("filename"),
        "source": source,
        "total_pages": total if rec.get("filetype") == "pdf" else None,
        "pages_read": pages_read,
        "truncated": truncated,
        "next_hint": "Increase max_chars or request the next page range to continue reading." if truncated else None,
        "text": text[:params.max_chars],
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Graph tools — traversal over the corpus graph `doccat graph build` produces
# from a `graph:`-enabled profile's extracted relations (see server.py's
# module docstring and docs/GRAPH.md). Read-only, same as every tool above;
# every result carries the doc_ids backing it, so the normal next step is
# read_document or get_document_metadata — the graph points at evidence, it
# never replaces reading it.
# ---------------------------------------------------------------------------

_NO_GRAPH_HINT = ("No corpus graph has been built yet (looked in {dir}). Run "
                 "`doccat graph build` after indexing with a profile that has a "
                 "`graph:` block (e.g. legal, parliamentary) — see docs/GRAPH.md.")


def _resolve_entity(graph, query: str) -> tuple[object | None, str | None]:
    """Fuzzy-resolve one entity name. Returns (node, None) or (None, error_message)."""
    matches = graph.find(query, limit=5)
    if not matches:
        return None, (f"No entity matching {query!r} was found in the graph. "
                      "Try graph_search for a broader look, or check the spelling "
                      "against how a document actually names it.")
    if len(matches) > 1 and matches[0].name.lower() != query.strip().lower():
        suggestions = ", ".join(f"{m.name!r}" for m in matches[:5])
        # Still proceed with the best match — ambiguity is common and a wrong
        # guess costs one extra query, whereas refusing costs the agent a
        # round trip every time. The alternatives are listed either way.
        log.debug("Entity query %r matched several: %s — using the first", query, suggestions)
    return matches[0], None


@mcp.tool(
    name="graph_overview",
    annotations={"title": "Corpus Graph Overview", "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def graph_overview() -> str:
    """Get a high-level view of the corpus graph: entities and relations extracted
    from the documents by a `graph:`-enabled profile, consolidated across the
    whole corpus by `doccat graph build`.

    Call this before graph_search/graph_neighbors/graph_path/graph_communities to
    see whether a graph exists at all, and what kinds of entities and relations
    it actually contains — the vocabulary is whatever the profile's `graph:`
    block guided the model toward, not a fixed schema.

    Returns:
        str: JSON with node/edge/community counts, entity types, and the most
        common relation labels — or a hint to run `doccat graph build` if no
        graph exists yet.
    """
    graph = _load_graph()
    if not len(graph):
        return _NO_GRAPH_HINT.format(dir=GRAPH_DIR)
    overview = graph.overview()
    overview["hint"] = ("Use graph_search to find an entity by name, graph_neighbors to see "
                        "what it connects to, graph_path to see how two entities connect, "
                        "and graph_communities for the broader clusters they belong to.")
    return json.dumps(overview, ensure_ascii=False, indent=2)


@mcp.tool(
    name="graph_search",
    annotations={"title": "Search Graph Entities", "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def graph_search(params: GraphSearchInput) -> str:
    """Find entities in the corpus graph by name.

    The entry point when you only have a name from a document's text, not a
    resolved graph entity — use this before graph_neighbors/graph_path when the
    name might be ambiguous or might not match exactly (the graph merges
    spelling variants like "ACME Corp." and "Acme Corporation" into one entity
    with the others kept as aliases).

    Args:
        params (GraphSearchInput): query string, result limit.

    Returns:
        str: JSON list of matching entities, each with its canonical name,
        aliases, type, degree (how connected it is) and supporting doc_ids.
    """
    graph = _load_graph()
    if not len(graph):
        return _NO_GRAPH_HINT.format(dir=GRAPH_DIR)
    matches = graph.find(params.query, limit=params.limit)
    return json.dumps({
        "query": params.query,
        "count": len(matches),
        "results": [m.to_dict() for m in matches],
        "hint": "Use graph_neighbors(entity=...) on a result's name to explore its connections."
               if matches else "No match. Try a shorter or partial name, or graph_overview "
                               "to see what entity types this graph actually has.",
    }, ensure_ascii=False, indent=2)


@mcp.tool(
    name="graph_neighbors",
    annotations={"title": "Explore Entity Connections", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def graph_neighbors(params: GraphNeighborsInput) -> str:
    """Get everything one entity connects to in the corpus graph.

    Use this for "who/what is connected to X" questions — it returns every
    neighboring entity within `hops` steps (relation direction doesn't matter:
    an entity that X supplies to and one that supplies to X both count as
    connected) along with the actual relation stated between them and which
    documents state it.

    Args:
        params (GraphNeighborsInput): entity name (fuzzy-resolved), hop count.

    Returns:
        str: JSON with the resolved entity, its neighbors, and the connecting
        edges (relation label, description, weight, doc_ids) — or an error
        naming the closest entities actually in the graph.
    """
    graph = _load_graph()
    if not len(graph):
        return _NO_GRAPH_HINT.format(dir=GRAPH_DIR)
    node, error = _resolve_entity(graph, params.entity)
    if error:
        return f"Error: {error}"
    result = graph.neighbors(node.id, hops=params.hops)
    result["hint"] = ("Use read_document or get_document_metadata on the doc_ids backing "
                      "an edge to verify it against the source.")
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool(
    name="graph_path",
    annotations={"title": "Connect Two Entities", "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def graph_path(params: GraphPathInput) -> str:
    """Find how two entities are connected in the corpus graph.

    Use this for "how are X and Y related" questions when they aren't directly
    mentioned together — it returns the shortest chain of stated relations
    linking them, each hop with the actual relation and the documents that
    state it, or reports that they are not connected at all.

    Args:
        params (GraphPathInput): the two entity names (each fuzzy-resolved).

    Returns:
        str: JSON with the path (list of entities) and the edges connecting
        each consecutive pair, or an explicit "not connected" result.
    """
    graph = _load_graph()
    if not len(graph):
        return _NO_GRAPH_HINT.format(dir=GRAPH_DIR)
    source, error = _resolve_entity(graph, params.source)
    if error:
        return f"Error (source): {error}"
    target, error = _resolve_entity(graph, params.target)
    if error:
        return f"Error (target): {error}"
    path = graph.shortest_path(source.id, target.id)
    if path is None:
        return json.dumps({
            "connected": False,
            "source": source.to_dict(), "target": target.to_dict(),
            "hint": f"No chain of stated relations links {source.name!r} and "
                   f"{target.name!r} in this graph. They may simply never be "
                   "discussed together, or may belong to unrelated documents.",
        }, ensure_ascii=False, indent=2)
    path["connected"] = True
    return json.dumps(path, ensure_ascii=False, indent=2)


@mcp.tool(
    name="graph_communities",
    annotations={"title": "Entity Clusters", "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def graph_communities(params: GraphCommunitiesInput) -> str:
    """List the clusters of densely-interrelated entities detected in the corpus graph,
    or find which cluster one entity belongs to.

    This is the high-level view: before drilling into individual entities, use
    this with no `entity` to see the major groupings in the corpus (each with a
    short generated summary, when `doccat graph build --summarize` was used),
    or with an `entity` to see what it belongs to and who else is in that group.

    Args:
        params (GraphCommunitiesInput): optional entity to look up; otherwise
            limit/offset over the full list, largest clusters first.

    Returns:
        str: JSON list of clusters (id, size, summary, member names) — or, with
        `entity` set, the single cluster containing it (an entity with no
        detected cluster is reported as such, not as an error).
    """
    graph = _load_graph()
    if not len(graph):
        return _NO_GRAPH_HINT.format(dir=GRAPH_DIR)

    def _view(community) -> dict:
        members = [graph.nodes[m].name for m in community.members if m in graph.nodes]
        return {"id": community.id, "size": community.size, "summary": community.summary,
                "members": members}

    if params.entity:
        node, error = _resolve_entity(graph, params.entity)
        if error:
            return f"Error: {error}"
        community = graph.community_of(node.id)
        if community is None:
            return json.dumps({
                "entity": node.to_dict(), "community": None,
                "hint": f"{node.name!r} was not grouped into any cluster — it is isolated "
                       "or only weakly connected in this graph.",
            }, ensure_ascii=False, indent=2)
        return json.dumps({"entity": node.to_dict(), "community": _view(community)},
                         ensure_ascii=False, indent=2)

    ordered = sorted(graph.communities, key=lambda c: c.size, reverse=True)
    page = ordered[params.offset:params.offset + params.limit]
    return json.dumps({
        "total": len(ordered),
        "count": len(page),
        "offset": params.offset,
        "has_more": len(ordered) > params.offset + len(page),
        "communities": [_view(c) for c in page],
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Health check (plain HTTP GET, no MCP client required — for monitoring)
# ---------------------------------------------------------------------------

try:
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @mcp.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        catalog = _load_catalog()
        graph = _load_graph()
        ok = bool(catalog)
        return JSONResponse(
            {
                "status": "ok" if ok else "degraded",
                "documents": len(catalog),
                "catalog_path": str(CATALOG_PATH),
                "catalog_paths": [str(p) for p in CATALOG_PATHS],
                "profiles": sorted({r.get("profile") or "(none)" for r in catalog.values()}),
                "docs_root": str(DOCS_ROOT),
                "docs_root_exists": DOCS_ROOT.is_dir(),
                "text_cache_exists": TEXT_CACHE.is_dir(),
                "graph_dir": str(GRAPH_DIR),
                "graph_nodes": len(graph),
                "graph_edges": len(graph.edges),
                "graph_communities": len(graph.communities),
            },
            status_code=200 if ok else 503,
        )
except (ImportError, AttributeError):  # older SDK without custom_route
    pass


def main() -> None:
    global DOCS_ROOT, CATALOG_PATH, CATALOG_PATHS, TEXT_CACHE, GRAPH_DIR
    parser = argparse.ArgumentParser(description="Document catalog MCP server (streamable HTTP)")
    parser.add_argument("--docs-root", type=Path, default=DOCS_ROOT,
                        help="Folder with the original documents (env DOCS_ROOT)")
    parser.add_argument("--catalog", action="append", default=None, metavar="PATH",
                        help="Catalog JSONL to serve (env CATALOG_PATH). Repeatable, and "
                             "each value may be a comma-separated list or a glob — several "
                             "catalogs are merged by doc_id, so index shards can be served "
                             "without merging them into one file first.")
    parser.add_argument("--text-cache", type=Path, default=TEXT_CACHE,
                        help="Folder with extracted-text copies (env TEXT_CACHE)")
    parser.add_argument("--graph-dir", type=Path, default=None,
                        help="Folder with the corpus graph built by `doccat graph build` "
                             "(env GRAPH_DIR; default: 'graph' next to the catalog)")
    parser.add_argument("--host", default=HOST, help="Bind address (env MCP_HOST)")
    parser.add_argument("--port", type=int, default=PORT, help="Port (env MCP_PORT)")
    args = parser.parse_args()

    DOCS_ROOT = args.docs_root.resolve()
    if args.catalog:
        CATALOG_PATHS = [p for spec in args.catalog for p in _catalog_paths(spec)]
    CATALOG_PATH = CATALOG_PATHS[0]
    TEXT_CACHE = args.text_cache
    GRAPH_DIR = args.graph_dir or Path(os.getenv(
        "GRAPH_DIR", str(CATALOG_PATH.resolve().parent / "graph")))
    mcp.settings.host = args.host
    mcp.settings.port = args.port

    n = len(_load_catalog())
    if n == 0:
        log.warning("Catalog %s is missing or empty — run `doccat index` first "
                    "(the server will pick it up automatically once it exists)", CATALOG_PATH)
    if not DOCS_ROOT.is_dir():
        log.warning("DOCS_ROOT %s does not exist — read_document will rely on the text cache", DOCS_ROOT)
    n_graph = len(_load_graph())
    if n_graph == 0:
        log.info("No corpus graph at %s yet — the graph_* tools will report that until "
                 "`doccat graph build` is run (the server will pick it up automatically)",
                 GRAPH_DIR)
    log.info("doc_catalog_mcp: %d documents, %d graph entities | MCP endpoint "
             "http://%s:%d/mcp | health http://%s:%d/health",
             n, n_graph, args.host, args.port, args.host, args.port)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
