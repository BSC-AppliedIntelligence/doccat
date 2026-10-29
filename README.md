# doc_catalog_mcp

An MCP server that lets an agent work through thousands of PDF/Office documents
using an **LLM-written metadata catalog** instead of a vector store — no
embeddings, no chunking, no similarity search. The agent reads what each
document *is about* and decides which ones to open, the way an analyst uses a
library catalogue. See [DESIGN.md](DESIGN.md) for the rationale and the
architecture.

Three things make that catalog good enough to retrieve on:

* **Cataloguing profiles** — the prompt, the metadata schema and the reading
  strategy adapt to the domain and the kind of document. A contract gets
  `parties`, `expiry_date` and `risk_flags`; a session transcript gets
  `chamber`, `speakers` and `agenda_items`; a manual gets `systems` and
  `error_codes`. See [docs/PROFILES.md](docs/PROFILES.md).
* **A profile-designing agent** — `doccat optimize` reads a sample of an
  unfamiliar repository, works out what it is, and writes the profile for it,
  then tries that profile on real documents and revises it. See
  [docs/OPTIMIZE.md](docs/OPTIMIZE.md).
* **A parallel pipeline** — extraction runs in a process pool, LLM calls in a
  thread pool, writes are journalled and checkpointed, and a corpus can be split
  across machines with `--shard`. See [docs/SCALING.md](docs/SCALING.md).
* **A corpus graph** — a `graph:`-enabled profile (`legal`, `parliamentary`)
  also extracts relations between entities; `doccat graph build` resolves them
  across the whole corpus, and five MCP tools traverse it — "how are X and Y
  connected?" where keyword search alone comes up short. See
  [docs/GRAPH.md](docs/GRAPH.md).

One configuration file sets all of it — endpoints, directories, workers,
serving — with environment variables and flags layered on top. See
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

[docs/doccat_Quick_Start_Guide.pdf](docs/doccat_Quick_Start_Guide.pdf)

## Layout

| Path | What it is |
|---|---|
| `doccat/` | the library: config, profiles, LLM client, sources, pipeline, catalog store, CLI |
| `doccat/config.py` | the single configuration model behind file, env and flags |
| `doccat/optimize.py` | the agent that designs a cataloguing profile from a sample |
| `doccat/graph.py` | corpus-level graph consolidation (entity resolution, communities) |
| `doccat/profiles/*.yaml` | the built-in cataloguing profiles |
| `server.py` | the MCP server (FastMCP, streamable HTTP) |
| `extractors.py` | text extraction (pdf, docx, pptx, xlsx, md, txt, html) with content sniffing |
| `deploy/` | Dockerfile, compose stack, systemd units, installer |
| `tests/` | test suite, including a mock LLM endpoint |
| `indexer.py`, `json_indexer.py` | the original commands, kept working as thin wrappers |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

This installs the `doccat` command. `make dev` does the same plus the test
dependencies; `pip install -e '.[proxy]'` adds mcpo for Open WebUI.

Use a virtualenv rather than a shared interpreter: the server needs
`mcp>=1.9,<2` (the 2.x SDK moves the FastMCP API), and mcpo is sensitive to the
`starlette`/`fastapi` versions around it. [DEPLOYMENT.md](DEPLOYMENT.md) covers
repairing a shared environment where that has already gone wrong.

## 1. Configure it once

```bash
doccat config init
```

Edit the `doccat.yaml` it writes — the corpus directory, the catalog path, the
LLM endpoint, the worker count — then check what it resolves to:

```bash
doccat config show
```

Every command below then works with no flags at all. Flags and environment
variables still override the file when you need them to, and skipping this step
entirely is fine: pass the flags instead. Full reference in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## 2. Build the catalog

Point it at your documents and at any OpenAI-compatible endpoint (vLLM,
llama.cpp, TGI, Ollama, LM Studio, the Hugging Face router):

```bash
doccat index --root /data/documents --catalog ./catalog.jsonl --base-url http://localhost:8080/v1 --model qwen2.5-14b-instruct --llm-workers 16
```

Each document is routed to a cataloguing profile automatically. See what it
would pick, without spending a single token:

```bash
doccat index --root /data/documents --dry-run
```

Supported formats: pdf, docx, pptx, xlsx, md, txt, html. The real format is
detected from the file's magic bytes, not its extension, so misnamed files from
bulk dumps are handled correctly.

Written next to the catalog:

- `extracted_text/` — a Markdown copy of every extraction, with YAML front matter
- `llm_errors/` — raw model replies that could not be parsed, for diagnosis
- `needs_ocr.txt` — PDFs with no text layer
- `needs_conversion.txt` — legacy binary Office files (`.doc/.ppt/.xls`, possibly
  renamed). If LibreOffice (`soffice`) is installed these are converted and
  indexed automatically

Re-run the same command any time: unchanged files are skipped by content hash,
new and modified files are indexed, deleted files are removed, and records left
in an error state are retried. `--force` rebuilds everything.

### JSON dumps

A JSON array where each element carries a text field plus scalar metadata (a
dump of session transcripts, tickets, articles) goes through the same pipeline:

```bash
doccat json --input Plenos.json --field texto --catalog ./plenos.jsonl --base-url http://localhost:8080/v1 --model my-model -j 16
```

Every other field of the record is preserved under `source_metadata`, and can
drive profile routing (`match: {metadata_equals: {orga: Senado}}`).

### Using the Hugging Face inference router

```bash
doccat index --root /data/documents --base-url https://router.huggingface.co/v1 --model Qwen/Qwen3.6-35B-A3B --api-key $HF_TOKEN --bill-to BSC-LT
```

`--bill-to` (or `HF_BILL_TO`) sends `X-HF-Bill-To` so usage is billed to that
organisation; `--header NAME:VALUE` adds any other header.

## 3. Customise the cataloguing — profiles

```bash
doccat profiles list
```

```bash
doccat profiles show legal
```

```bash
doccat profiles init mydomain
```

Nine profiles ship with the package: `generic`, `legal`, `financial`,
`parliamentary`, `technical`, `academic`, `intranet`, `minutes`, and
`long_document` (map-reduce for very long texts). Write your own as YAML,
inherit from one of these, state only the differences, and load it with
`--profiles-dir`:

```yaml
name: clinical_trial
extends: academic
priority: 70
match:
  any_of:
    - path_globs: ["**/trials/**"]
    - filename_regex: "(NCT[0-9]{8}|protocol)"
prompt:
  guidance: Never infer an endpoint or a sample size that is not stated.
  fields:
    trial_id: registry identifier (NCT..., EudraCT...), or empty
    phase: "trial phase: I, II, III, IV, or empty"
    primary_endpoints:
      type: list
      description: primary endpoints exactly as defined in the protocol
```

```bash
doccat index --root /data/trials --profiles-dir ./profiles
```

[docs/PROFILES.md](docs/PROFILES.md) documents every key, the routing rules and
the excerpt strategies.

### Let an agent design the profile for you

On a repository nobody has catalogued before, you do not yet know what a good
profile would say. `doccat optimize` reads a stratified sample, works out what
the corpus is, drafts a profile for it, then catalogues a few real documents
with that draft and revises it based on what came back empty or useless:

```bash
doccat optimize --root /data/documents -o profiles/mydomain.yaml
```

The result is a YAML file you own, carrying the reasoning behind it in comments.
Review it, check its routing with `--dry-run`, then index. See
[docs/OPTIMIZE.md](docs/OPTIMIZE.md).

## 4. Scale it

```bash
doccat index --root /data/documents -j 32 --extract-workers 8 --rps 20
```

`--llm-workers/-j` sets concurrent LLM requests, `--extract-workers` the
extraction process pool. A 429 from any worker pauses the whole pool. Progress,
rate and ETA are logged as the run proceeds; Ctrl-C drains in-flight work and
saves. For very large corpora, split across machines and merge:

```bash
doccat index --root /data/documents --catalog shard_0.jsonl --shard 0/4
```

```bash
doccat merge shard_*.jsonl -o catalog.jsonl
```

[docs/SCALING.md](docs/SCALING.md) covers worker sizing, throughput maths,
resumability and sharding.

## 5. Build the corpus graph (optional)

If any profile used above has a `graph:` block (`legal` and `parliamentary`
both do), the catalog already carries `relations` between entities. Resolve
them across the whole corpus:

```bash
doccat graph build --catalog ./catalog.jsonl --summarize
```

This is where "how are X and Y connected?" questions get answered — see
[docs/GRAPH.md](docs/GRAPH.md).

## 6. Run the MCP server

```bash
doccat serve --docs-root /data/documents --catalog ./catalog.jsonl --port 8000
```

It listens at `http://<host>:8000/mcp` (streamable HTTP), serves a plain-HTTP
health check at `/health`, and reloads the catalog automatically when it changes
on disk. `--catalog` may be repeated, comma-separated, or a glob, so index
shards can be served without merging them first.

With containers:

```bash
cp .env.example .env && docker compose -f deploy/docker-compose.yml --profile index run --rm indexer
```

```bash
docker compose -f deploy/docker-compose.yml up -d server
```

On a systemd host: `sudo ./deploy/install.sh`. Full instructions in
[DEPLOYMENT.md](DEPLOYMENT.md).

## 7. Connect an agent

```json
{
  "mcpServers": {
    "doc_catalog": { "type": "http", "url": "http://localhost:8000/mcp" }
  }
}
```

## Tools exposed

| Tool | Purpose |
|---|---|
| `catalog_overview` | Corpus stats: counts, doc types, top topics, profiles in use and the extra fields each one produced |
| `search_documents` | Ranked keyword search over all metadata, including profile-specific fields; filters by `doc_type`, `topic`, `profile` |
| `search_by_field` | Query one metadata field directly — "which contracts involve ACME?", "which manuals mention error E-403?" |
| `list_documents` | Browse by folder prefix, doc type or profile |
| `get_document_metadata` | The full record for one document |
| `read_document` | Extract the actual text on demand (PDF page ranges); falls back to the `extracted_text/` cache |
| `graph_overview` / `graph_search` / `graph_neighbors` / `graph_path` / `graph_communities` | Traverse the corpus graph built by `doccat graph build` — see [docs/GRAPH.md](docs/GRAPH.md) |

## Quick test

```bash
npx @modelcontextprotocol/inspector
```

Connect it to `http://localhost:8000/mcp` and call `catalog_overview`.

## Tests

```bash
make test
```

The suite runs the whole pipeline against a mock OpenAI-compatible endpoint:
configuration precedence, profile routing and inheritance, relation extraction
(inline and dedicated-pass) and graph consolidation, the optimizer's sampling
and sanitising, parallel execution, rate-limit back-off, crash recovery,
sharding, the server's tools (including the graph tools over a real MCP
client), and the legacy entry points.

## Migrating from the old scripts

`python indexer.py --root ...` and `python json_indexer.py --input ...` still
work with their original flags — they now delegate to `doccat index` / `doccat
json` and gain every new option. Existing `catalog.jsonl` files are read as they
are; records simply have no `profile` field until they are re-indexed. See
[CHANGELOG.md](CHANGELOG.md).

## Status and next steps

- Build an evaluation set and measure this approach against a conventional RAG
  baseline on the same questions
- OCR for the documents listed in `needs_ocr.txt`, or an always-multimodal mode
- Optional hybrid upgrade: one embedding of the *summary* per document (not per
  chunk) to complement keyword search

  This code was created with aid from Artificial Intelligence
