# Configuration

Everything — endpoints, directories, worker counts, serving, the optimizer — is
set in one place. There is one configuration model behind the config file, the
environment variables and the command-line flags, so you never have to work out
which mechanism a given setting belongs to.

```bash
doccat config init          # write a commented doccat.yaml
doccat config show          # what a command will actually use, secrets masked
doccat config path          # which file would be loaded
```

Once the file is in place, commands need no flags at all:

```bash
doccat index
```

```bash
doccat serve
```

## Precedence

Later wins:

1. built-in defaults (`doccat/config.py`)
2. the config file
3. environment variables
4. command-line flags

That ordering is what makes one image work everywhere: bake the config file in,
override one value with an environment variable per deployment, and still pass a
flag for a one-off run.

```bash
doccat config show
```

```
indexing:
  llm_workers: 16
  ...
# ---
# config file: /etc/doc-catalog-mcp/doccat.yaml
# env override: llm.api_key <- $LLM_API_KEY
# env override: indexing.llm_workers <- $DOCCAT_LLM_WORKERS
```

The trailing block is the fastest way to find out why a deployment is not using
the endpoint you thought it was.

## Where the file is looked for

`--config FILE` wins. Otherwise `$DOCCAT_CONFIG`, then the first of:

```
./doccat.yaml
./doccat.yml
./.doccat.yaml
~/.config/doccat/config.yaml
/etc/doccat/config.yaml
```

No file at all is fine — defaults and the environment are then used on their own.

## The file

```yaml
paths:
  docs_root: ./documents           # the corpus to catalogue
  catalog: ./catalog.jsonl         # the catalog to build and serve
  # text_cache: ./extracted_text   # default: next to the catalog
  # error_log: ./llm_errors        # default: next to the catalog
  profiles_dir:
    - ./profiles                   # your custom cataloguing profiles

llm:
  base_url: http://localhost:8080/v1
  model: qwen2.5-14b-instruct
  api_key: ${LLM_API_KEY:-not-needed}
  temperature: 0.1
  max_tokens: 4096
  timeout: 300
  think: false
  requests_per_second: 0           # 0 = unlimited
  # bill_to: BSC-LT                # Hugging Face router: sends X-HF-Bill-To
  # headers:
  #   X-Custom-Header: value

indexing:
  llm_workers: 8
  extract_workers: 0               # 0 = auto (min(cpu, 8)); 1 = extract inline
  mode: auto                       # parse | upload | auto
  # profile: legal                 # force one profile instead of auto-routing
  checkpoint_every: 50
  progress_every: 20
  change_detection: hash           # hash | mtime
  # shard: 0/4
  include: []
  exclude: []

server:
  host: 0.0.0.0
  port: 8000
  stateless: true

optimize:
  sample_size: 12
  refine_rounds: 1
  sample_chars: 6000

# Corpus-level graph consolidation (`doccat graph build`) — distinct from a
# profile's own `graph:` block, which controls what gets *extracted* per
# document. This section controls how those results get *consolidated* into
# one corpus graph. See docs/GRAPH.md.
graph:
  # out_dir: ./graph          # default: 'graph' next to the catalog
  detect_communities: true
  min_community_size: 2
  summarize: false            # one LLM call per community if true
  community_seed: 0
```

Two rules make the file safe to move around:

**Relative paths are relative to the config file, not to the working
directory.** A config at `/etc/doc-catalog-mcp/doccat.yaml` saying
`docs_root: ./documents` always means `/etc/doc-catalog-mcp/documents`, whatever
directory you run the command from.

**Unknown sections and keys are errors.** A typo is reported at load time rather
than silently ignored:

```
Configuration error: doccat.yaml: unknown key(s) in llm: ['modle']; allowed: [...]
```

Types are coerced, so `port: "9000"` and `think: "yes"` work, and a value that
cannot be coerced is reported precisely:

```
Configuration error: indexing.llm_workers: expected an integer, got 'many'
```

## Secrets

Use `${VAR}` or `${VAR:-fallback}` anywhere in the file; they are expanded from
the environment when the file is read. Keep the key in the environment and the
shape of the deployment in the file:

```yaml
llm:
  api_key: ${LLM_API_KEY}
```

`doccat config show` masks secrets (`sk-a…op (20 chars)`). `--reveal` prints
them in full — for when you need to confirm which key is loaded.

## Environment variables

Every setting has one, and they are the names the container image and the
systemd units already use.

| Setting | Variable |
|---|---|
| `paths.docs_root` | `DOCS_ROOT` |
| `paths.catalog` | `CATALOG_PATH` |
| `paths.text_cache` | `TEXT_CACHE` |
| `paths.error_log` | `DOCCAT_ERROR_LOG` |
| `paths.profiles_dir` | `DOCCAT_PROFILES_DIR` (`:`-separated) |
| `llm.base_url` | `LLM_BASE_URL` |
| `llm.model` | `LLM_MODEL` |
| `llm.api_key` | `LLM_API_KEY` |
| `llm.temperature` | `LLM_TEMPERATURE` |
| `llm.max_tokens` | `LLM_MAX_TOKENS` |
| `llm.timeout` | `LLM_TIMEOUT` |
| `llm.think` | `LLM_THINK` |
| `llm.requests_per_second` | `LLM_RPS` |
| `llm.bill_to` | `HF_BILL_TO` |
| `indexing.llm_workers` | `DOCCAT_LLM_WORKERS` |
| `indexing.extract_workers` | `DOCCAT_EXTRACT_WORKERS` |
| `indexing.mode` | `DOCCAT_MODE` |
| `indexing.profile` | `DOCCAT_PROFILE` |
| `indexing.checkpoint_every` | `DOCCAT_CHECKPOINT_EVERY` |
| `indexing.progress_every` | `DOCCAT_PROGRESS_EVERY` |
| `indexing.change_detection` | `DOCCAT_CHANGE_DETECTION` |
| `indexing.shard` | `DOCCAT_SHARD` |
| `server.host` | `MCP_HOST` |
| `server.port` | `MCP_PORT` |
| `server.stateless` | `MCP_STATELESS` |
| `optimize.sample_size` | `DOCCAT_SAMPLE_SIZE` |
| `optimize.refine_rounds` | `DOCCAT_REFINE_ROUNDS` |
| `graph.out_dir` | `DOCCAT_GRAPH_DIR` (`doccat serve` passes it to `server.py` as `GRAPH_DIR`; set `GRAPH_DIR` directly only if running `server.py` standalone) |
| `graph.detect_communities` | `DOCCAT_GRAPH_COMMUNITIES` |
| `graph.min_community_size` | `DOCCAT_GRAPH_MIN_COMMUNITY_SIZE` |
| `graph.summarize` | `DOCCAT_GRAPH_SUMMARIZE` |

Plus `DOCCAT_CONFIG` (which file to load) and `LOG_LEVEL`.

## In deployments

**Containers.** Mount the file at `/app/config/doccat.yaml`; the entrypoint
exports `DOCCAT_CONFIG` for it automatically when it is there. `CONFIG_DIR` in
`.env` selects the host directory.

```bash
docker compose -f deploy/docker-compose.yml up -d server
```

**systemd.** Put `DOCCAT_CONFIG=/etc/doc-catalog-mcp/doccat.yaml` in
`/etc/doc-catalog-mcp.env` and keep only the API key in the env file. Both the
server and the indexer units read that env file, so they cannot disagree about
where the data is.

## From Python

```python
from doccat.config import Settings

settings = Settings.load()              # discover the file
settings = Settings.load("prod.yaml")   # or name one
print(settings.llm.model, settings.paths.resolved_text_cache())
```

`Settings.load(..., use_env=False)` ignores the environment, which is what the
tests use to assert on a file in isolation.
