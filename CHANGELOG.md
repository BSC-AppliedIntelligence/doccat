# Changelog

## 0.4.0 — relation extraction and graph-based retrieval

Catalog records already carried `entities` per document; this release adds
*relations* between them and a way to query the resulting graph, without
changing how anything already shipped behaves. Full design rationale and
usage reference: [docs/GRAPH.md](docs/GRAPH.md).

### Extraction (`doccat/profiles.py`, `doccat/pipeline.py`)

- `FieldSpec` gains `type: list, items: object` — a list of structured
  elements, not just scalars. Required before a `relations` field
  (`{subject, relation, object, description, strength}` each) could exist at
  all; every existing scalar `list` field is untouched.
- A new `graph:` profile YAML block (parallel to `match:`/`excerpt:`/`llm:`)
  turns relation extraction on: `entity_types`/`relation_types` fold into the
  prompt as vocabulary hints (never an enforced enum — an open label plus a
  free-text description sidesteps the cross-document consistency problems a
  fixed RDF-style predicate vocabulary would create), and the profile gets a
  `relations` field for free, the same way every profile already gets
  `entities` for free from `generic`. Absent `graph:`, nothing changes.
  `graph:` inherits like `excerpt:`/`llm:` (unlike `match:`).
- Extraction rides the existing single per-document LLM call by default — zero
  extra cost. `graph.dedicated_pass: true` opts a profile into a second,
  focused call grounded in the entities the main call already found, for
  document families dense enough to need it. `graph.min_confidence` drops
  relations the model itself flagged as weak.
- Two built-ins use this now: `legal` (inline) and `parliamentary` (dedicated
  pass). The other seven are untouched.

### Consolidation (`doccat/graph.py`, new)

- `doccat graph build` resolves every document's *document-local* entity
  names into *one graph across the corpus* — the same "ACME Corp" in fifty
  documents becomes one node backed by fifty `doc_id`s. Entity resolution is
  tier 1 only (normalized exact match: case-fold, strip punctuation and
  legal-entity suffixes); embedding-based fuzzy resolution is a documented
  future upgrade, not built.
- Duplicate `(source, target, relation)` triples merge across documents into
  one edge with combined `doc_ids` and accumulated weight; a relation whose
  subject and object resolve to the same node is dropped as a near-certain
  resolution artifact.
- Community detection via `networkx`'s built-in Louvain implementation — pure
  Python, no new system dependency beyond `networkx` itself. `--summarize`
  adds one LLM call per community for a short GraphRAG-style summary.
- Persistence is plain JSONL (`nodes.jsonl`/`edges.jsonl`/`communities.jsonl`)
  — greppable and diffable like `catalog.jsonl`, not an opaque graph database.
  A `networkx.MultiDiGraph` is rebuilt from these files on load.
- `doccat graph build`/`doccat graph stats` CLI, and a `graph:` section in
  `doccat.yaml` (distinct from a profile's own `graph:` block) so a deployment
  needs no flags.

### MCP server (`server.py`)

- Five new read-only tools: `graph_overview`, `graph_search`,
  `graph_neighbors`, `graph_path`, `graph_communities`. Entity names are
  fuzzy-resolved, every result carries the `doc_id`s backing it, and a catalog
  with no graph built yet gets a clear hint rather than an error.
- The graph hot-reloads the same way the catalog does — rebuild it while the
  server runs and the next query picks it up, no restart.
- `catalog_overview` reports `has_graph`; `/health` reports graph node/edge/
  community counts. `GRAPH_DIR` (env var, `--graph-dir`, or `graph.out_dir`)
  locates it.

### Profiler (`doccat/optimize.py`)

- The analysis call now also proposes `entity_types`/`relation_types` when
  the sample shows a network of named parties worth tracing — and is told
  explicitly to propose nothing when it wouldn't help. `render_profile_yaml`
  only emits a `graph:` block when a vocabulary was actually proposed, and
  reserves the `relations` field name so the agent can't shadow the
  structured schema with a plain string list.
- The refine loop gets a new model-free signal: relations whose subject or
  object isn't among that same document's own extracted entities — a cheap
  way to catch fabricated connections.

### Deployment

- `networkx>=3.0` is the one new runtime dependency, loaded only by
  `doccat graph build` and the server's graph tools.
- Containers and systemd docs cover running `doccat graph build` alongside
  indexing.

### Tests

- 88 new tests (191 total): the schema engine's list-of-objects shape, graph
  block parsing/inheritance/validation, both extraction paths through the
  real pipeline (including graceful degradation when a dedicated pass
  fails), confidence filtering, the consolidation module's resolution/
  merging/community/traversal/persistence logic, the MCP tools both as
  direct calls and over a real MCP client against a live server (confirming
  hot-reload while serving), and the optimizer's graph-vocabulary proposal
  and its new refine-loop signal.

## Unreleased

- PDF text extraction now uses pypdfium2 (BSD-3-Clause / Apache-2.0) instead of
  PyMuPDF (AGPL-3.0), so every dependency is permissively licensed ahead of the
  open-source release. Output keeps the same `[page N]` markers and page counts;
  calls into PDFium are serialised with a lock because PDFium is not thread-safe.
  New `tests/test_extractors.py` covers PDF extraction, page ranges and sniffing.
- The MCP server is now verified end to end against the real SDK: 26 checks over
  streamable HTTP (initialize, tools/list, every tool, pagination, filters,
  schema validation), plus catalog hot-reload, text-cache fallback, multi-catalog
  glob serving, and the mcpo OpenAPI proxy.
- `tests/test_server.py` uses the real FastMCP when the SDK is installed, and
  falls back to a stub only when it is not.
- DEPLOYMENT.md: how to spot and repair a mixed `mcp` install (two `dist-info`
  directories; `pip uninstall` removes one distribution per invocation, so the
  orphaned files survive the first run), and a note that mcpo's
  `Router.__init__() got an unexpected keyword argument 'on_startup'` is a
  starlette/fastapi clash rather than an MCP problem.

## 0.3.0 — one configuration file, and an agent that designs the profile

### Single configuration module (`doccat/config.py`)

- One settings model behind the config file, the environment variables and the
  command-line flags, resolved in that order of precedence: defaults → config
  file → environment → flags. The indexer, the optimizer and the MCP server all
  read it, so a deployment is configured once rather than three times.
- `doccat config init` writes a commented `doccat.yaml` covering paths, the LLM
  endpoint, indexing, serving and the optimizer. With it in place, `doccat
  index` and `doccat serve` need no flags at all.
- `doccat config show` prints the fully resolved settings with secrets masked
  (`--reveal` to see them) and names the config file plus every environment
  override in effect — the fastest way to find out why a deployment is not using
  the endpoint you thought it was. `doccat config path` prints which file would
  be loaded.
- Discovery: `--config FILE`, then `$DOCCAT_CONFIG`, then `./doccat.yaml`,
  `./doccat.yml`, `./.doccat.yaml`, `~/.config/doccat/config.yaml`,
  `/etc/doccat/config.yaml`. No file at all is fine.
- `${VAR}` and `${VAR:-fallback}` are expanded from the environment when the
  file is read, so secrets stay out of it.
- Relative paths resolve against the config file's own directory, not the
  working directory, so a config is portable.
- Unknown sections and keys are rejected at load time, and values are coerced to
  the declared types with precise errors (`indexing.llm_workers: expected an
  integer, got 'many'`).
- `ENV_MAP` in the module is now the single source of truth for environment
  variable names, shared by the CLI, the container image and the systemd units.

### Profile-designing agent (`doccat/optimize.py`)

- `doccat optimize --root DIR` reads a sample of an unfamiliar repository and
  writes the cataloguing profile for it, in four phases:
  **sample** (stratified by folder and file type, seeded and reproducible, so a
  rare document family is still represented); **analyse** (one LLM call that
  asks what these documents are and which fields would actually tell them
  apart); **draft** (rendered into profile YAML and loaded through the normal
  profile machinery, so an invalid draft fails immediately); **refine**
  (catalogue a few real documents with the draft, measure which fields came back
  empty or identical, feed that back for revision).
- Everything the model proposes is sanitised rather than trusted: names forced
  to snake_case, unknown parents and strategies replaced, type synonyms mapped,
  absurd character budgets clamped, invalid regexes dropped, core fields and
  nameless entries removed. A poor analysis yields a poorer profile, never an
  unloadable one.
- The written file carries its own provenance in comments — corpus summary,
  domain, families found, why those fields, why that strategy.
- A refine round that fails or produces an invalid profile is logged and the
  previous good draft is kept.
- Works on JSON dumps too (`--input`), and honours `--include`/`--exclude` so a
  heterogeneous repository can be given several narrow profiles.
- `--print-only`, `--sample`, `--refine`, `--trial-docs`, `--seed`, `--name`.
  Cost is `1 + (trial_docs + 1) × refine_rounds` calls — five by default.

### Deployment

- Containers pick up a config file mounted at `/app/config/doccat.yaml`
  automatically (`CONFIG_DIR` in `.env`); environment variables still override
  it, so one image serves several deployments.
- The systemd env file documents `DOCCAT_CONFIG`, so the units can be driven by
  a YAML config with only the API key left as a variable.
- `make config`, `make config-show`, `make optimize`.

### Docs

- New [docs/CONFIGURATION.md](docs/CONFIGURATION.md) and
  [docs/OPTIMIZE.md](docs/OPTIMIZE.md); README, DESIGN and DEPLOYMENT updated.

### Tests

- 34 new tests (103 total): configuration precedence, coercion, interpolation,
  path anchoring, masking and the shipped template; the optimizer's
  stratification, determinism, sanitising of hostile model output, refine loop,
  failure handling and CLI.

## 0.2.0 — domain profiles and parallel indexing

Two themes: make the catalogue's descriptions fit the documents being
catalogued, and make building one for thousands of files practical.

### Cataloguing profiles

- New `doccat/profiles/` — YAML profiles that set the cataloguer persona, domain
  rules, document-type vocabulary, metadata schema and reading strategy per
  document family. Nine ship with the package: `generic`, `legal`, `financial`,
  `parliamentary`, `technical`, `academic`, `intranet`, `minutes`,
  `long_document`.
- Profiles inherit with `extends`, so a domain profile states only its
  differences and the core schema stays identical across families.
- Profiles route themselves: `match` rules on path globs, filename/content
  regexes, file type, size, or a JSON record's own metadata, combined with
  `any_of` for OR semantics and `priority` for precedence. `--profile` forces
  one; `generic` is the fallback.
- Domain fields are declared with types (`string`, `list`, `object`, `integer`,
  `number`, `boolean`) and used both to render the JSON skeleton shown to the
  model and to coerce what comes back.
- Excerpt strategies: `head_tail`, `head`, `full`, `sampled` (head + evenly
  spaced windows + tail, for long uniform texts) and `map_reduce` (notes per
  chunk, then catalogue from the notes) — which addresses summaries and
  descriptions that previously only reflected a document's first pages.
- Per-profile LLM overrides: `model`, `temperature`, `max_tokens`, `think`, and
  `ingestion` (`parse`/`upload`/`auto`).
- `doccat profiles list | show | init | validate`, and `--dry-run` to see the
  routing for a whole corpus without spending a token.
- Records now carry `profile`, and `incomplete_fields` when the model left a
  declared core field empty.

### Parallel, resumable, deployable indexing

- Extraction runs in a process pool (`--extract-workers`), LLM calls in a thread
  pool (`--llm-workers/-j`). Started from a `forkserver` context, so forking a
  multi-threaded parent is never involved.
- One pooled `httpx.Client` shared by all workers, with a shared `RateLimitGate`:
  a 429 from any worker pauses the whole pool (honouring `Retry-After`, else
  exponential back-off with jitter). Optional `--rps` token bucket.
- Endpoint quirks are learned once and shared — a rejected `chat_template_kwargs`
  or a hard completion-token cap no longer gets rediscovered per document.
- Crash-safe catalog store: every record is journalled immediately, the full
  catalog is rewritten atomically every `--checkpoint-every` records, and a
  leftover journal is replayed on the next start.
- First Ctrl-C drains in-flight work and saves (exit 130); a second aborts.
- Progress lines with live rate and ETA (`--progress-every`).
- `--shard I/N` splits a corpus deterministically across machines; `doccat merge`
  combines the results, or the server reads the shards directly.
- `--change-detection mtime` for very large trees; `--include`/`--exclude`
  globs; `--max-docs` for trial runs.
- `doccat stats` summarises a catalog by status, profile, file type, doc type
  and ingestion mode.

### Deployment

- `deploy/Dockerfile` — one image, three roles (`serve`, `index`, `mcpo`), non-root,
  with a health check and `tini`. Optional LibreOffice via `WITH_LIBREOFFICE=1`.
- `deploy/docker-compose.yml` + `.env.example` — server, one-shot indexer and the
  Open WebUI proxy, behind compose profiles.
- `deploy/install.sh` — bare-metal systemd installer (service user, venv under
  `/opt`, units, env file).
- Generic systemd units driven by `/etc/doc-catalog-mcp.env`:
  `doc-catalog-mcp.service`, `doc-catalog-indexer.service` + `.timer`,
  `mcpo.service`. The server unit is confined (`ProtectSystem=strict`,
  `ProtectHome=read-only`, `NoNewPrivileges`).
- `pyproject.toml` with the `doccat` console script; `Makefile` for common tasks.
- Every option is settable from the environment, so containers and units need no
  command-line editing.

### MCP server

- New `search_by_field` tool: query one metadata field directly, including the
  fields a profile invented.
- `catalog_overview` now reports `by_profile` and `profile_fields`, so an agent
  discovers at runtime which structured fields exist.
- The keyword search blob is built generically from the record, so a new
  profile's fields are searchable with no server change.
- `profile` filter on `search_documents` and `list_documents`; `profile` included
  in result summaries.
- `CATALOG_PATH`/`--catalog` accept several catalogs (comma-separated, repeated,
  or a glob), merged by `doc_id` — index shards can be served without merging.
- `/health` reports the catalog paths in use and the profiles present.
- Malformed catalog lines are skipped with a warning instead of killing the load.

### Tests

- New suite (69 tests) running the whole pipeline against a mock
  OpenAI-compatible endpoint: profile loading, inheritance and routing, excerpt
  strategies, parallel execution and overlap, rate-limit back-off, crash
  recovery, sharding, CLI wiring, the server's tools, and the legacy entry
  points.

### Compatibility

- `python indexer.py --root ...` and `python json_indexer.py --input ...` keep
  working with their original flags; they now delegate to `doccat index` /
  `doccat json` and gain every new option.
- The legacy module-level helpers (`LLMConfig`, `call_llm`, `build_excerpt`,
  `parse_metadata_json`, `parse_llm_metadata`, `load_catalog`, `save_catalog`,
  `METADATA_PROMPT`, `EXCERPT_SUFFIX`) are still importable. `METADATA_PROMPT` is
  derived from the `generic` profile so it cannot drift from the real prompt.
- Existing `catalog.jsonl` files are read as they are. Records simply have no
  `profile` field until re-indexed; nothing needs migrating.
- `mcp` is pinned to `>=1.9,<2` — the 2.x SDK moves the FastMCP API.

### Fixed

- A `CatalogStore` recovering records from a journal marked itself clean, so the
  recovered records could be dropped when the store closed.

## 0.1.0 — initial version

Serial indexer (`indexer.py`), JSON-record indexer (`json_indexer.py`), shared
extractors with content sniffing and LibreOffice conversion, FastMCP server with
five read-only tools, systemd units for the intranet deployment.
