# Scaling the cataloguing run

Cataloguing one document is a short burst of CPU (parse the PDF) followed by a
long wait on the network (the LLM call). Run serially, a 5,000-document corpus
at 8 s per document is about 11 hours; the endpoint is idle for nearly all of
it. The pipeline overlaps both phases and bounds each independently.

```
scan ──┬── hash files in a thread pool (I/O bound)
       │
       ├── N LLM worker threads, each:
       │     1. hand the file to a process pool for extraction (CPU bound)
       │     2. pick the cataloguing profile
       │     3. one LLM call — or map+reduce for long documents
       │     4. hand the record to the catalog store
       │
       └── CatalogStore: journal every record, checkpoint periodically
```

## The knobs

| Flag | Env | Default | What it does |
|---|---|---|---|
| `--llm-workers`, `-j` | `DOCCAT_LLM_WORKERS` | 8 | Concurrent LLM requests |
| `--extract-workers` | `DOCCAT_EXTRACT_WORKERS` | auto = `min(cpu, 8)` | Processes parsing documents. `1` extracts inline |
| `--rps` | `LLM_RPS` | 0 (unlimited) | Global cap on requests/second |
| `--checkpoint-every` | `DOCCAT_CHECKPOINT_EVERY` | 50 | Records between full catalog rewrites |
| `--progress-every` | `DOCCAT_PROGRESS_EVERY` | 20 | Documents between progress/ETA lines |
| `--change-detection` | — | `hash` | `mtime` is cheaper on very large trees |
| `--shard I/N` | `DOCCAT_SHARD` | — | Process one deterministic slice of the corpus |
| `--max-docs` | — | — | Stop after N documents (trial runs) |

## Choosing `--llm-workers`

This is the one that matters, and the right value comes from the endpoint, not
from the machine running `doccat`.

* **A local vLLM/TGI server** batches internally: push it until throughput stops
  improving. 16–32 is typical for a mid-sized model on one GPU; watch its
  batch-size metric rather than guessing.
* **llama.cpp or Ollama** usually serve one or two requests at a time. More than
  4 workers just queues requests and inflates per-request latency. Try `-j 4`.
* **A hosted router** (Hugging Face, or any commercial API) has its own rate
  limits. Start at 8–16, and use `--rps` to stay under a documented quota rather
  than discovering it through 429s.

Measure instead of guessing — the progress line reports the actual rate:

```bash
doccat index --root /data/documents --max-docs 100 -j 8
```

```
[100/100] 1.42 doc/s | ETA 0s | last: informe_2024.pdf (ok, profile=generic)
```

Double `-j`, run another 100 with `--force`, and compare. When doubling the
workers stops raising `doc/s`, the endpoint is saturated: stop there.

At 1.42 doc/s a 50,000-document corpus takes about 10 hours. That is the number
to plan a run against.

## Choosing `--extract-workers`

pypdfium2 and openpyxl are CPU-bound and hold the GIL, so extraction runs in a
separate **process** pool while the LLM threads wait on the network. The default
(`min(cpu_count, 8)`) is right unless:

* the corpus is mostly plain text or Markdown — extraction is nearly free, so
  `--extract-workers 1` (inline) avoids the pool entirely;
* the machine is shared — cap it so indexing does not starve other work;
* you are debugging an extraction crash — `1` gives you a clean traceback in the
  main process.

JSON sources carry their text already and never start a pool.

## Back-pressure and rate limits

The `RateLimitGate` is shared by every worker. When any request comes back 429
(or 503), the gate pauses the **whole pool** — honouring `Retry-After` when the
endpoint sends it, otherwise backing off 5, 10, 20 … seconds up to five minutes,
with jitter so the workers do not resume in lockstep. Having 32 threads each
retry on their own private schedule against an already-saturated endpoint only
makes things worse.

```
2026-03-02 04:11:07 WARNING Endpoint busy (429, hit 2/8) — pausing all workers 11s
```

Other endpoint quirks are learned once and shared: a rejected
`chat_template_kwargs`, or a hard completion-token cap discovered from a 400
response, applies to every worker from then on rather than being rediscovered
per document.

## Crash safety and resuming

Every finished record is appended to `<catalog>.journal` immediately; the full
catalog is rewritten atomically every `--checkpoint-every` records. A crash, an
OOM kill on document 8,431, or a power cut loses at most the records still in
flight. On the next start the journal is replayed into the catalog
automatically:

```
WARNING Recovered 37 record(s) from an interrupted run (catalog.jsonl.journal)
```

Ctrl-C is handled deliberately: the first one stops submitting new work, lets
in-flight documents finish, saves, and exits 130. A second Ctrl-C aborts
immediately.

Re-running is always safe and always cheap — unchanged documents are skipped by
content hash, and records left in an error state (`llm_error`,
`metadata_parse_error`, `extraction_error`, `legacy_format`, `needs_ocr`) are
retried. A run that ended with errors just wants the same command again:

```bash
doccat index --root /data/documents   # retries only what failed
```

## Very large corpora

### Cheaper change detection

Above roughly 100,000 files, hashing every file on every run costs real time.
`--change-detection mtime` uses modification time plus size instead:

```bash
doccat index --root /data/documents --change-detection mtime
```

It is cheaper and slightly less exact — a file rewritten with identical size and
a preserved mtime is missed. Sound for most corpora; keep `hash` when documents
are rewritten in place by a system that preserves timestamps.

### Sharding across machines

`--shard I/N` assigns each document to a shard by a stable hash of its
`doc_id` — no coordination, no shared state, no queue. Run one shard per
machine:

```bash
doccat index --root /data/documents --catalog shard_0.jsonl --shard 0/4 -j 16
```

```bash
doccat index --root /data/documents --catalog shard_1.jsonl --shard 1/4 -j 16
```

Then either merge them:

```bash
doccat merge shard_*.jsonl -o catalog.jsonl
```

…or skip the merge entirely and let the server read them all:

```bash
doccat serve --docs-root /data/documents --catalog 'catalogs/shard_*.jsonl'
```

Each shard is independently resumable, and re-running one shard only touches its
own slice. Because assignment is a pure function of `doc_id`, adding a machine
means re-sharding (the slices change), so choose `N` for the run.

### Narrowing the run

`--include` and `--exclude` take glob patterns against the relative path, and
compose with everything else:

```bash
doccat index --root /data/documents --include "contracts/**" --profile legal
```

```bash
doccat index --root /data/documents --exclude "archive/**" --exclude "**/tmp/*"
```

## Cost

The pipeline makes exactly one LLM call per document, except:

* `map_reduce` profiles: `chunks + 1` calls, bounded by `max_chunks`;
* `upload`/`auto` ingestion when local extraction yields nothing: one extra call;
* retries after an empty reply or a transient failure.

`doccat stats` reports the real distribution after a run:

```bash
doccat stats --catalog catalog.jsonl
```

```json
{
  "documents": 8451,
  "by_status": {"ok": 8309, "needs_ocr": 121, "llm_error": 21},
  "by_profile": {"generic": 4102, "legal": 1980, "technical": 1233, "long_document": 262},
  "by_ingestion": {"parse": 8047, "map_reduce": 262, "upload": 0},
  "incomplete": 44
}
```

Input tokens per call follow the profile's excerpt budget (`head_chars +
tail_chars`, or `max_chars`), so the excerpt settings are the cost dial: raising
`head_chars` from 8,000 to 16,000 characters roughly doubles prompt cost across
the whole corpus.

## Serving at scale

The server keeps the catalog in memory and rebuilds the search index when the
file changes on disk — a 15 MB, 30,000-record catalog loads in well under a
second and searches in milliseconds. Indexing and serving are fully decoupled:
re-run the indexer whenever you like, the server picks up the new catalog with
no restart and no downtime.
