#!/usr/bin/env bash
# Container entrypoint. The first argument selects the role:
#
#   serve            run the MCP server (default)
#   index            run `doccat index --root "$DOCS_ROOT"` once
#   json             run `doccat json --input "$JSON_INPUT"` once
#   mcpo             wrap a running MCP server as an OpenAPI service
#   doccat ...       any doccat subcommand, verbatim
#   <anything else>  executed as a plain command (bash, python, ...)
#
# Every flag is also settable through the environment (LLM_BASE_URL,
# DOCCAT_LLM_WORKERS, DOCCAT_PROFILE, ...) — see `doccat index --help`.
set -euo pipefail

role="${1:-serve}"
shift || true

# A config file mounted at /app/config/doccat.yaml is picked up automatically.
# It is only exported when it exists: DOCCAT_CONFIG pointing at a missing file
# is an error, and running without a config file is perfectly normal.
if [[ -z "${DOCCAT_CONFIG:-}" && -f /app/config/doccat.yaml ]]; then
    export DOCCAT_CONFIG=/app/config/doccat.yaml
    echo "entrypoint: using config /app/config/doccat.yaml" >&2
fi

# A profiles directory mounted at /app/profiles is picked up automatically.
if [[ -d /app/profiles ]] && compgen -G "/app/profiles/*.y*ml" > /dev/null; then
    export DOCCAT_PROFILES_DIR="${DOCCAT_PROFILES_DIR:+$DOCCAT_PROFILES_DIR:}/app/profiles"
    echo "entrypoint: using custom profiles from /app/profiles" >&2
fi

case "$role" in
    serve)
        exec doccat serve --docs-root "$DOCS_ROOT" --catalog "$CATALOG_PATH" \
             --text-cache "$TEXT_CACHE" --host "$MCP_HOST" --port "$MCP_PORT" "$@"
        ;;
    index)
        exec doccat index --root "$DOCS_ROOT" --catalog "$CATALOG_PATH" \
             --text-cache "$TEXT_CACHE" "$@"
        ;;
    json)
        : "${JSON_INPUT:?set JSON_INPUT to the path of the JSON dump}"
        exec doccat json --input "$JSON_INPUT" --field "${JSON_FIELD:-texto}" \
             --catalog "$CATALOG_PATH" --text-cache "$TEXT_CACHE" "$@"
        ;;
    mcpo)
        exec mcpo --host 0.0.0.0 --port "${MCPO_PORT:-8001}" \
             ${MCPO_API_KEY:+--api-key "$MCPO_API_KEY"} \
             --server-type streamable-http -- "${MCP_URL:?set MCP_URL}" "$@"
        ;;
    doccat)
        exec doccat "$@"
        ;;
    *)
        exec "$role" "$@"
        ;;
esac
