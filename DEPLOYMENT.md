# Deployment Guide

Three supported ways to run this, in increasing order of effort:

1. [Containers](#1-containers-docker-compose) — one image, three roles
2. [systemd](#2-systemd-bare-metal) — `deploy/install.sh` on any systemd host
3. [Manual](#3-manual) — a venv and two commands

All three end at the same place: an MCP endpoint at `/mcp` for MCP-native
clients, an optional OpenAPI proxy for Open WebUI, and a catalog refreshed on a
schedule. Indexing and serving are decoupled — the server reloads
`catalog.jsonl` when it changes on disk, so re-indexing never needs a restart.

Whichever you choose, settings can come from one YAML file
(`doccat config init`), from environment variables, or from flags — in that
order of precedence. `doccat config show` prints what a command will actually
use and where each value came from. See
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## 1. Containers (docker compose)

```bash
cp .env.example .env
```

Edit `.env` — at minimum `DOCUMENTS_DIR`, `LLM_BASE_URL` and `LLM_MODEL`.

Build the catalog (a one-shot job; it sits behind the `index` compose profile so
`up` never starts it by accident):

```bash
docker compose -f deploy/docker-compose.yml --profile index run --rm indexer
```

If the profile used has a `graph:` block, build the corpus graph too (same
container, same `index` profile — it's `doccat`, just a different
subcommand):

```bash
docker compose -f deploy/docker-compose.yml --profile index run --rm indexer doccat graph build --summarize
```

Serve it:

```bash
docker compose -f deploy/docker-compose.yml up -d server
```

```bash
curl http://localhost:8000/health
```

Add the Open WebUI proxy when you need it:

```bash
docker compose -f deploy/docker-compose.yml --profile proxy up -d mcpo
```

### The image

One image, selected by the first argument: `serve` (default), `index`, `json`,
`mcpo`, `doccat <subcommand>`, or any command.

```bash
docker build -f deploy/Dockerfile -t doc-catalog-mcp .
```

```bash
docker run --rm doc-catalog-mcp doccat profiles list
```

Build with `--build-arg WITH_LIBREOFFICE=1` to convert legacy `.doc/.ppt/.xls`
files automatically (adds roughly 450 MB).

The container runs as an unprivileged user (uid 10001), mounts the corpus
read-only, has a `HEALTHCHECK` on `/health`, and uses `tini` to reap the
processes the extraction pool forks.

### Volumes

| Container path | Holds | Mode |
|---|---|---|
| `/data/documents` | the corpus | read-only |
| `/data/catalog` | `catalog.jsonl`, `extracted_text/`, `llm_errors/` | read-write |
| `/app/profiles` | your custom cataloguing profiles | read-only, optional |
| `/app/config` | `doccat.yaml`, to configure everything from one file | read-only, optional |

Any `*.yaml` in `/app/profiles` is picked up automatically — the entrypoint sets
`DOCCAT_PROFILES_DIR` when the directory is non-empty. Likewise, a
`/app/config/doccat.yaml` is exported as `DOCCAT_CONFIG` when it exists. The
environment variables in `.env` still override anything in that file, so one
image serves several deployments.

### Scheduling the refresh

The indexer is a one-shot job, so schedule it from outside: a host cron entry, a
Kubernetes `CronJob`, or a systemd timer.

```cron
0 2 * * * cd /opt/doc_catalog_mcp && docker compose -f deploy/docker-compose.yml --profile index run --rm indexer >> /var/log/doccat-index.log 2>&1
```

## 2. systemd (bare metal)

```bash
sudo ./deploy/install.sh
```

The installer creates a `doccat` service user, installs the code and a venv
under `/opt/doc-catalog-mcp`, drops the systemd units, and writes
`/etc/doc-catalog-mcp.env` if it is not already there. It never overwrites an
existing env file and never starts anything — review the configuration first.

```bash
sudo nano /etc/doc-catalog-mcp.env      # paths, LLM endpoint + key, worker counts
```

Everything in that env file can instead live in one YAML config — write it with
`doccat config init`, put it at `/etc/doc-catalog-mcp/doccat.yaml`, set
`DOCCAT_CONFIG` to it in the env file, and keep only the API key as a variable.

Build the catalog, then start serving:

```bash
sudo -u doccat /opt/doc-catalog-mcp/.venv/bin/doccat index --root /srv/documents -j 16
```

```bash
sudo systemctl enable --now doc-catalog-mcp
```

```bash
curl http://localhost:8010/health
```

Nightly incremental refresh, and the Open WebUI proxy:

```bash
sudo systemctl enable --now doc-catalog-indexer.timer
```

```bash
sudo systemctl enable --now mcpo
```

### What gets installed

| Unit | Role |
|---|---|
| `doc-catalog-mcp.service` | the MCP server |
| `doc-catalog-indexer.service` | one-shot incremental re-index |
| `doc-catalog-indexer.timer` | runs it nightly at 02:00 (±15 min) |
| `mcpo.service` | OpenAPI proxy for Open WebUI |

All four read `/etc/doc-catalog-mcp.env`, so the server and the indexer cannot
disagree about where the data is. The file holds the LLM API key — install it
mode 0600.

Notes carried over from the original Rocky Linux 9 deployment, which still
apply:

* Rocky 9's default `python3` is 3.9, too old for the `mcp` package.
  `sudo dnf install python3.12` first; the installer picks the newest suitable
  interpreter it finds.
* Code and venv belong in `/opt`, not `/home` — systemd cannot reliably execute
  binaries from home directories (SELinux, permissions). Documents may live in
  `/home`: they are only read.
* The indexer runs `Nice=10` with idle I/O scheduling so a long catalogue rebuild
  does not starve the server.

The server unit is confined: `ProtectSystem=strict`, `ProtectHome=read-only`,
`NoNewPrivileges`, and a single `ReadWritePaths` for the catalog directory. If
your documents live somewhere `ProtectHome=read-only` blocks, adjust that line
rather than loosening the whole unit.


## 3. Manual

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -e .
```

```bash
doccat index --root /data/documents --catalog /data/catalog.jsonl --base-url http://localhost:8080/v1 --model my-model -j 16
```

If the profile used has a `graph:` block (`legal`, `parliamentary`, or a
custom one), build the corpus graph before serving — see
[docs/GRAPH.md](docs/GRAPH.md):

```bash
doccat graph build --catalog /data/catalog.jsonl --summarize
```

```bash
doccat serve --docs-root /data/documents --catalog /data/catalog.jsonl --port 8010
```

Keep it fresh with cron:

```cron
0 2 * * * /opt/doc-catalog-mcp/.venv/bin/doccat index --root /data/documents --catalog /data/catalog.jsonl >> /var/log/doccat.log 2>&1
```

## Connecting clients

**MCP-native** (Claude Desktop, Claude Code, any MCP client):

```json
{
  "mcpServers": {
    "doc_catalog": { "type": "http", "url": "http://<host>:8010/mcp" }
  }
}
```

**Open WebUI** — its native MCP support is still experimental, its OpenAPI tool
support is stable, so put mcpo in front. In *Admin → Tools → Add connection
(OpenAPI)*:

- URL: `http://<host>:8011`
- Spec path: `openapi.json`
- Auth: none, or Bearer when `MCPO_API_KEY` is set (recommended whenever the
  port is reachable by anyone else)

Browse `http://<host>:8011/docs` for the Swagger UI listing the eleven tools
(six catalog tools plus five `graph_*` tools, present even on a catalog with
no graph built yet — they report that cleanly rather than erroring).

## Serving index shards

A corpus catalogued with `--shard` does not need merging. `CATALOG_PATH` (and
`--catalog`) accept several paths, comma-separated or as a glob:

```bash
CATALOG_PATH='/data/catalogs/shard_*.jsonl' doccat serve --docs-root /data/documents
```

Records are merged by `doc_id`, and each file is watched for changes
independently.

## Verification checklist

```bash
systemctl is-active doc-catalog-mcp mcpo
```

```bash
curl -s http://localhost:8010/health | python3 -m json.tool
```

`/health` reports document count, the catalog paths in use, the profiles present
in the catalog, and whether the documents root and text cache exist:

```json
{
  "status": "ok",
  "documents": 8451,
  "catalog_paths": ["/var/lib/doc-catalog-mcp/catalog.jsonl"],
  "profiles": ["generic", "legal", "technical"],
  "docs_root_exists": true,
  "text_cache_exists": true
}
```

```bash
doccat stats --catalog /var/lib/doc-catalog-mcp/catalog.jsonl
```

```bash
journalctl -u doc-catalog-mcp -n 20
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `status=203/EXEC` on start | `ExecStart` path wrong, or the venv lives under `/home` — deploy to `/opt` with absolute paths for both interpreter and script |
| "bad unit file setting" | Typo in the unit — `systemd-analyze verify /etc/systemd/system/doc-catalog-mcp.service` pinpoints the line |
| `curl /mcp` returns 406 | Normal: plain curl does not send MCP's SSE `Accept` headers. Use `/health` for plain-HTTP checks |
| `mcp>=1.9` not found on install | venv built with Python 3.9 — rebuild with `python3.12 -m venv` |
| `ImportError: cannot import name 'LifespanContextT' from 'mcp.shared.context'` | A mixed `mcp` install: 2.x was installed over 1.x, which drops `mcp/server/fastmcp/` from its manifest but leaves the 1.x files on disk, so the old FastMCP imports from the new `shared/context.py`. See below |
| `TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'` from mcpo | A `starlette`/`fastapi` version clash in that environment, not an MCP problem. Install mcpo in the project venv (`pip install -e '.[proxy]'`), not in a crowded shared environment |
| Port unreachable from other machines | Check `MCP_HOST=0.0.0.0`, then `firewall-cmd`, then the perimeter firewall |
| `read_document` says "file missing on disk" | `DOCS_ROOT` does not match the `--root` used at indexing time; the tool falls back to the text cache when one is configured |
| Service fails but a manual run works | SELinux — `sudo ausearch -m avc -ts recent` |
| Health says `degraded`, 0 documents | The catalog path is wrong, or the indexer has not run yet |
| Indexing crawls, endpoint idle | Too few `--llm-workers`; see [docs/SCALING.md](docs/SCALING.md) |
| Repeated `Endpoint busy (429)` | The endpoint is saturated — lower `-j`, or set `--rps` to the documented quota |
| Everything routes to `generic` | The `match` rules do not fit your folder layout — check with `--dry-run`, then adjust or force one with `--profile`, or draft one with `doccat optimize` |
| A setting is ignored | Something later in the chain overrides it — `doccat config show` names the file and every env override in effect |
| `Configuration error: unknown key(s) in ...` | A typo in the config file; the message lists the allowed keys |
| `DOCCAT_CONFIG points at a missing file` | The variable is set but the file is not there (in containers, mount it at `/app/config/doccat.yaml` and let the entrypoint set the variable) |

### Repairing a mixed `mcp` install

`pip list` showing one version is not proof the install is clean — check for
more than one distribution directory:

```bash
ls -d "$(python -c 'import site; print(site.getsitepackages()[0])')"/mcp-*.dist-info
```

Two of them means two manifests, and `pip uninstall` removes **one distribution
per invocation**, so run it until none is left, then reinstall:

```bash
pip uninstall -y mcp && pip uninstall -y mcp && pip install 'mcp>=1.9,<2'
```

The second uninstall is what removes the orphaned files, because they are listed
only in the older distribution's manifest. Verify:

```bash
python -c "from mcp.server.fastmcp import FastMCP; import server; print('ok')"
```

A project virtualenv avoids the whole problem, which is why the installer builds
one. Reserve the repair above for an environment shared with other tools.

## Security notes

Everything runs inside your facilities: documents, catalog and LLM. The server
is read-only and validates that every resolved path stays inside `DOCS_ROOT`,
but the catalog and the documents are internal data:

- keep the ports VPN-only, or put nginx with TLS in front of both;
- set `MCPO_API_KEY` whenever the OpenAPI port is reachable by others;
- run as a non-root service user (the installer and the image both do);
- `/etc/doc-catalog-mcp.env` holds the LLM API key — mode 0600, root-owned;
- when cataloguing through a hosted endpoint, remember that document excerpts
  leave the building. `--bill-to` controls billing, not data residency.
