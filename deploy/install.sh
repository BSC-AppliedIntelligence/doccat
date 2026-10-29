#!/usr/bin/env bash
# Bare-metal installer for a systemd host (Rocky/RHEL/Debian/Ubuntu).
#
#   sudo ./deploy/install.sh [--prefix /opt/doc-catalog-mcp] [--python python3.12]
#
# It creates the service user, installs the code and a venv under --prefix,
# drops the systemd units, and installs /etc/doc-catalog-mcp.env if it is not
# there yet. It never overwrites an existing env file, and never starts the
# services — review the env file first, then:
#
#   systemctl enable --now doc-catalog-mcp
#   systemctl enable --now doc-catalog-indexer.timer
set -euo pipefail

PREFIX=/opt/doc-catalog-mcp
PYTHON=""
STATE_DIR=/var/lib/doc-catalog-mcp
ENV_FILE=/etc/doc-catalog-mcp.env
SERVICE_USER=doccat

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --user)   SERVICE_USER="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ $EUID -eq 0 ]] || { echo "Run as root (sudo)." >&2; exit 1; }

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The mcp package needs Python >= 3.10; distro defaults are often older.
if [[ -z "$PYTHON" ]]; then
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1 \
           && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
            PYTHON="$candidate"; break
        fi
    done
fi
[[ -n "$PYTHON" ]] || { echo "No Python >= 3.10 found. Install python3.12 and retry." >&2; exit 1; }
echo "==> Using $("$PYTHON" --version)"

echo "==> Service user: $SERVICE_USER"
id -u "$SERVICE_USER" >/dev/null 2>&1 || useradd --system --home "$STATE_DIR" \
    --shell /usr/sbin/nologin "$SERVICE_USER"

echo "==> Installing code into $PREFIX"
# systemd cannot reliably execute binaries from /home (SELinux, permissions),
# hence /opt for both the code and the venv.
mkdir -p "$PREFIX" "$STATE_DIR"
cp -r "$SRC_DIR"/doccat "$SRC_DIR"/server.py "$SRC_DIR"/extractors.py \
      "$SRC_DIR"/indexer.py "$SRC_DIR"/json_indexer.py \
      "$SRC_DIR"/pyproject.toml "$SRC_DIR"/requirements.txt "$SRC_DIR"/README.md "$PREFIX"/

"$PYTHON" -m venv "$PREFIX/.venv"
"$PREFIX/.venv/bin/pip" install --quiet --upgrade pip
"$PREFIX/.venv/bin/pip" install --quiet -r "$PREFIX/requirements.txt"
"$PREFIX/.venv/bin/pip" install --quiet --no-deps -e "$PREFIX"

chown -R "$SERVICE_USER:$SERVICE_USER" "$PREFIX" "$STATE_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
    install -m600 -o root -g root "$SRC_DIR/deploy/doc-catalog-mcp.env.example" "$ENV_FILE"
    echo "==> Wrote $ENV_FILE — EDIT IT before starting the services"
else
    echo "==> $ENV_FILE already exists, left untouched"
fi

echo "==> Installing systemd units"
for unit in doc-catalog-mcp.service doc-catalog-indexer.service \
            doc-catalog-indexer.timer mcpo.service; do
    sed "s#/opt/doc-catalog-mcp#$PREFIX#g; s#User=doccat#User=$SERVICE_USER#; \
         s#Group=doccat#Group=$SERVICE_USER#" \
        "$SRC_DIR/deploy/$unit" > "/etc/systemd/system/$unit"
done
systemctl daemon-reload

cat <<EOF

Installed. Next:

  1. edit $ENV_FILE   (paths, LLM endpoint + key, worker counts)
  2. build the catalog:
       sudo -u $SERVICE_USER $PREFIX/.venv/bin/doccat index --root <DOCS_ROOT> -j 16
  3. start serving:
       systemctl enable --now doc-catalog-mcp
       curl http://localhost:\${MCP_PORT:-8010}/health
  4. optional — nightly refresh and the Open WebUI proxy:
       systemctl enable --now doc-catalog-indexer.timer
       systemctl enable --now mcpo
EOF
