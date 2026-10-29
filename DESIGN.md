# Document Catalog MCP — System Description

## Purpose

Give an AI agent fast, cheap access to a folder containing thousands of PDF and
Office documents **without** building a traditional RAG pipeline (no chunking,
no embeddings, no vector database). Instead of retrieving text chunks by vector
similarity, the agent consults a **metadata catalog** that describes what each
document contains, decides which documents are relevant, and reads only those on
demand.

This mirrors how a person uses a library: check the card catalogue, pull the
right books, read the relevant pages.

The catalogue is only as good as its descriptions, which is why the system is
built around three ideas: descriptions are written by a **domain-aware
cataloguer** (profiles), the cataloguer for an unfamiliar repository can be
**designed by an agent that reads a sample of it** (the optimizer), and writing
the catalogue for a large corpus is a **parallel, resumable batch job** (the
pipeline). All of it is driven from **one configuration model**.

## Architecture

```
                        ┌──────────────────────────────┐
                        │  LLM (your facilities or a    │
                        │  hosted OpenAI-compatible     │
                        │  endpoint)                    │
                        └──────────▲───────────────────┘
                                   │ /v1/chat/completions
                                   │ (pooled, rate-limit aware,
                                   │  N concurrent requests)
┌──────────────┐                 ┌─┴─────────────────────┐    writes   ┌────────────────┐
│ Documents    │ ───────────────▶│  doccat pipeline      │ ──────────▶ │ catalog.jsonl  │
│ folder, or   │  scan/extract   │  ├ sources            │             │ 1 line per doc │
│ a JSON dump  │  (process pool) │  ├ profiles (routing) │             └───────┬────────┘
└──────▲───────┘                 │  ├ LLM thread pool    │                     │ reads
       │                         │  └ journal+checkpoint │                     │
       │ on-demand extraction    └───────────────────────┘                     │
       │                         ┌──────────────┐                              │
       └─────────────────────────│  server.py   │◀─────────────────────────────┘
                                 │ doc_catalog_ │
                                 │ mcp (FastMCP)│   streamable HTTP :8000/mcp
                                 └──────────────┘◀───────────── Agent / MCP client
```

### Component 1 — the cataloguing pipeline (`doccat/`)

| Module | Responsibility |
|---|---|
| `config.py` | The single settings model behind the config file, the environment and the flags |
| `sources.py` | Where work comes from: a folder tree (`FileSource`) or a JSON array dump (`JsonRecordSource`). Produces `WorkItem`s and materialises their text |
| `profiles.py` | Cataloguing profiles: schema, prompt assembly, excerpt strategies, inheritance, routing |
| `llm.py` | Pooled, thread-safe chat-completions client with shared back-pressure |
| `pipeline.py` | Orchestration: process pool for extraction, thread pool for LLM calls, progress, interrupts |
| `catalog.py` | Crash-safe, concurrency-safe JSONL store (journal + atomic checkpoints) |
| `jsonparse.py` | Tolerant parsing of malformed model JSON |
| `optimize.py` | The agent that designs a cataloguing profile from a sample of the corpus |
| `cli.py` | The `doccat` command |

The run, step by step:

1. **Scan.** Walk the tree (or read the JSON array), apply `--include/--exclude`,
   and compute each document's change key — a SHA-256 content hash, or
   mtime+size under `--change-detection mtime`. Hashing runs in a thread pool.
2. **Select.** Skip documents whose change key matches a healthy catalog record.
   Error statuses (`llm_error`, `metadata_parse_error`, `extraction_error`,
   `legacy_format`, `needs_ocr`) are always retried. `--shard I/N` keeps only
   one deterministic slice.
3. **Extract.** A process pool parses the file (pypdfium2, python-docx,
   python-pptx, openpyxl, stdlib HTML parser). Format is decided by magic bytes,
   not extension, so misnamed files from bulk dumps work. Legacy OLE2 files are
   converted through LibreOffice when it is installed.
4. **Route.** A profile is chosen from the document's path, name, type, size,
   source metadata or first extracted characters — unless `--profile` forces one.
5. **Catalogue.** The profile builds the prompt and the excerpt; one LLM call
   returns the metadata as JSON. Long documents under a `map_reduce` profile get
   one call per chunk plus a reduce call, so the summary covers the whole
   document rather than its first pages.
6. **Normalise and store.** The reply is parsed tolerantly, coerced to the
   profile's declared types, and appended to the journal; the catalog is
   rewritten atomically every `--checkpoint-every` records.

**Parallelism.** LLM calls wait on the network and extraction burns CPU, so they
are bounded separately: `--llm-workers` threads and `--extract-workers`
processes. A 429 from any worker pauses the whole pool (shared `RateLimitGate`),
and endpoint quirks — a rejected `chat_template_kwargs`, a hard token cap — are
learned once and applied everywhere.

**Resumability.** Every record hits the journal as soon as it exists, so a crash
loses at most the in-flight documents; the journal is replayed on the next
start. The first Ctrl-C drains in-flight work and saves.

**Failure handling.** Scanned PDFs with no text layer are recorded
`needs_ocr` and listed in `needs_ocr.txt`; legacy binary Office files without
LibreOffice are recorded `legacy_format` and listed in `needs_conversion.txt`;
unparseable replies are recorded `metadata_parse_error` with the raw reply saved
under `llm_errors/`. All of them are retried on the next run.

### Component 2 — cataloguing profiles (`doccat/profiles/*.yaml`)

A profile is the answer to "who is cataloguing this, and what should they record?"

* a **persona and domain rules** (`role`, `instructions`, `guidance`);
* a **controlled vocabulary** of document types;
* a **metadata schema**: the shared core fields plus domain-specific ones —
  `parties`/`expiry_date`/`risk_flags` for contracts,
  `chamber`/`speakers`/`agenda_items` for session transcripts,
  `systems`/`error_codes` for manuals;
* an **excerpt strategy**: `head_tail`, `head`, `full`, `sampled` (head + evenly
  spaced windows + tail, for long uniform texts) or `map_reduce`;
* optional **per-profile LLM settings** (model, temperature, token budget,
  `ingestion: upload` for scanned documents);
* **routing rules** that claim documents by path, name, type, size, content or
  source metadata.

Profiles inherit (`extends`), so a domain profile states only its differences
and the core schema stays identical across families — which is what lets one
catalog mix contracts, manuals and transcripts and still be searched uniformly.

Nine profiles ship with the package; `doccat profiles init` scaffolds new ones.
Full reference: [docs/PROFILES.md](docs/PROFILES.md).

### Component 2b — the profile-designing agent (`doccat/optimize.py`)

Choosing or writing a profile presupposes knowing what is in the repository,
which is usually the thing you do not know. `doccat optimize` closes that
circle in four phases: a **stratified sample** across folders and file types; a
single **analysis** call that asks what these documents are and which fields
would actually tell them apart; a **draft** rendered into profile YAML, where
every value the model proposed is sanitised and the result is loaded through
the normal profile machinery so an invalid draft fails immediately; and a
**refine** loop that catalogues a few real documents with the draft, measures
which fields came back empty or identical, and feeds that back for revision.

The output is a file the operator owns, carrying its own reasoning in comments.
Nothing is applied automatically. Full reference:
[docs/OPTIMIZE.md](docs/OPTIMIZE.md).

### Component 3 — the metadata catalog (`catalog.jsonl`)

One JSON object per line, per document:

| Field | Description |
|---|---|
| `doc_id` | Stable id (SHA-1 of the relative path, 16 hex chars) |
| `path`, `filename`, `filetype`, `size_bytes`, `modified_at` | File system facts |
| `content_hash`, `change_key` | Drive incremental re-indexing |
| `profile` | Which cataloguing profile produced this record |
| `title`, `summary` | LLM-generated title and 3–5 sentence summary |
| `description` | 2–4 paragraphs: purpose, audience, structure, sections, concrete content |
| `doc_type` | From the profile's controlled vocabulary |
| `language` | ISO 639-1 |
| `keywords`, `topics` | 8–15 search keywords; 2–5 broad subject areas |
| `entities` | people, organizations, places, dates |
| `relations` | For `graph:`-enabled profiles: `{subject, relation, object, description, strength}` between entities — see Component 3b |
| `questions_answered` | Example questions the document can answer — very effective for matching user queries |
| *profile fields* | Whatever the profile declared: `parties`, `speakers`, `systems`, … |
| `extracted_text_path` | Path of the persisted extracted-text copy |
| `needs_ocr` | PDFs with no usable text layer |
| `incomplete_fields` | Core fields the model left empty |
| `ingestion` | `parse`, `upload` or `map_reduce` |
| `status`, `indexed_at`, `char_count`, `pages_or_paragraphs` | Bookkeeping |

JSONL keeps the catalog human-readable, diffable, greppable and trivially
editable. A few thousand documents load into memory in well under a second, and
the server reloads the file whenever it changes on disk.

Alongside it, the **text cache** holds one Markdown file per document with a
YAML front-matter header (`doc_id`, source path, type, extraction date). The
server falls back to it when a source file is missing or fails to parse.

### Component 3b — corpus graph consolidation (`doccat/graph.py`)

A `graph:`-enabled profile extracts `relations` using *document-local* entity
names. `doccat graph build` is the separate, re-runnable step that resolves
those into one graph across the corpus: the same "ACME Corp" mentioned in
fifty documents collapses into one node backed by fifty `doc_id`s. Entity
resolution is tier 1 only — normalized exact match (case-fold, strip
punctuation and legal-entity suffixes); a fuzzy, embedding-based tier 2 is a
documented future upgrade, not built. Community detection runs on
`networkx`'s built-in Louvain implementation — pure Python, no new system
dependency — optionally summarized with one LLM call per community. Storage
is plain JSONL (`nodes.jsonl`/`edges.jsonl`/`communities.jsonl`), not a graph
database, rebuilt into a `networkx.MultiDiGraph` for traversal on load. Full
reference: [docs/GRAPH.md](docs/GRAPH.md).

### Component 4 — the MCP server (`server.py`)

FastMCP server named `doc_catalog_mcp`, served over **streamable HTTP** at
`http://<host>:8000/mcp` so any number of agents can connect remotely. All tools
are read-only.

| Tool | Role in the workflow |
|---|---|
| `catalog_overview` | Orientation: counts, doc types, top topics, the profiles in use and the extra fields each produced. Agents call it first |
| `search_documents` | Weighted keyword search over every metadata field, profile fields included; filters by `doc_type`, `topic`, `profile` |
| `search_by_field` | Query one field directly — "which contracts involve ACME?", "which manuals cover error E-403?" |
| `list_documents` | Browse by folder prefix, doc type or profile, unranked |
| `get_document_metadata` | The full catalog record for one document |
| `read_document` | Extracts the actual text on demand — a page range for PDFs, whole file otherwise, truncated to `max_chars` with pagination hints |
| `graph_overview` / `graph_search` / `graph_neighbors` / `graph_path` / `graph_communities` | Traverse the corpus graph Component 3b produces — relational questions ("how are X and Y connected") rather than topical ones |

The search blob is built generically from the record, so a field invented by a
new profile becomes searchable with no server change. `catalog_overview`
advertises those fields per profile, which is how an agent discovers at runtime
that `search_by_field(field="parties", ...)` is worth trying; it also reports
`has_graph` so an agent knows whether the graph tools have anything to work
with. Every graph tool result carries the `doc_id`s backing it — the graph
points at evidence, read_document still does the verifying.

`CATALOG_PATH` accepts several catalogs (comma-separated, or a glob), merged by
`doc_id` — so the shards from a distributed indexing run can be served directly.
The corpus graph hot-reloads the same way: rebuild it while the server runs and
the next query picks it up, no restart.

### Query flow (no RAG)

1. User asks: *"What penalties do our 2024 supplier contracts include?"*
2. Agent → `catalog_overview` → learns the corpus has a `legal` profile whose
   records carry `parties`, `expiry_date`, `key_clauses`, `risk_flags`.
3. Agent → `search_documents(query="supplier contract 2024 penalties",
   doc_type="contract")` → ranked candidates with summaries. Or, when the
   question names a structured field, `search_by_field(field="key_clauses",
   value="penalty")` directly.
4. Agent → `read_document(doc_id=..., first_page=..., last_page=...)` to pull
   the actual clauses.
5. Agent composes the answer, citing document paths.

The agent's own reasoning replaces vector similarity: it reads summaries and
structured fields and decides what to open, exactly as an analyst would.

## Trade-offs vs. traditional RAG

**Advantages.** No embedding model, no vector DB, no chunking pipeline to
maintain. The catalog is transparent and auditable — you can read it, grep it,
diff it in git, and correct a bad record by hand. Documents are always read
fresh, so there are no stale chunks. Indexing costs one LLM call per document
rather than an embedding per chunk. And structured, domain-specific fields
support queries that similarity search handles poorly ("contracts expiring in
2025", "sessions where Deputy X spoke").

**Limitations.** Retrieval quality depends on metadata quality: a detail absent
from the summary and keywords can be missed. Profiles are the main mitigation —
a domain schema captures exactly the details that matter for that family — along
with `questions_answered` and the agent's ability to iterate with different
terms. Very long documents still need paged reading by the agent. Purely
semantic matches ("car" vs "automobile") remain weaker than embeddings; a hybrid
upgrade is available without changing the architecture — add one embedding of
the *summary* per document (not per chunk) and blend it into the ranking.

## Sizing and performance

Indexing throughput is set by the endpoint, not by this code. At 8 s per
document serially, 5,000 documents take about 11 hours; at `-j 16` against a
vLLM server that batches, the same corpus finishes in well under an hour. The
progress line reports the live rate and ETA, so measure with `--max-docs 100`
and scale from the number you get. See [docs/SCALING.md](docs/SCALING.md).

A catalog runs roughly 1–3 KB per document: ~15 MB for 30,000 documents. Server
search latency is milliseconds (in-memory scan); `read_document` latency is
dominated by PDF parsing, typically under a second.

## Configuration

Endpoints, directories, worker counts and serving are one settings model
(`doccat/config.py`) fed from four layers, later winning: built-in defaults, a
YAML config file, environment variables, and command-line flags. The same model
drives the indexer, the optimizer and the server, so a container can be
configured by a mounted file, a systemd unit by an env file, and a one-off run
by a flag — without three different mechanisms. `doccat config show` prints what
a command will actually use, secrets masked, with the origin of every override.
Full reference: [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Security notes

Everything can run inside your facilities: documents, metadata and the LLM. The
server validates that resolved file paths stay inside `DOCS_ROOT`, runs
read-only, and ships as a non-root container and a confined systemd unit. Add
TLS and an auth layer (reverse proxy) in front of the HTTP endpoint if agents
connect from other machines. When cataloguing through a hosted endpoint, note
that document excerpts leave the building.
