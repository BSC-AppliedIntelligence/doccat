# doc_catalog_mcp — one image, three roles (indexer, MCP server, mcpo proxy).
#
#   docker build -f deploy/Dockerfile -t doc-catalog-mcp .
#   docker run --rm -p 8000:8000 -v /data/docs:/data/documents:ro \
#              -v /data/catalog:/data/catalog doc-catalog-mcp serve
#
# Mount a doccat.yaml at /app/config/doccat.yaml to configure everything from
# one file; the environment variables below still override it.
#
# Build args:
#   WITH_LIBREOFFICE=1   add LibreOffice so legacy .doc/.ppt/.xls convert
#                        automatically (adds ~450 MB)
FROM python:3.12-slim AS base

ARG WITH_LIBREOFFICE=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CATALOG_PATH=/data/catalog/catalog.jsonl \
    TEXT_CACHE=/data/catalog/extracted_text \
    DOCS_ROOT=/data/documents \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

# tini reaps the child processes the extraction pool forks.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tini \
 && if [ "$WITH_LIBREOFFICE" = "1" ]; then \
        apt-get install -y --no-install-recommends libreoffice-writer libreoffice-impress \
                                                   libreoffice-calc fonts-dejavu; \
    fi \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so the layer is cached across code changes.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md /app/
COPY doccat /app/doccat
COPY server.py extractors.py indexer.py json_indexer.py /app/
RUN pip install --no-cache-dir --no-deps -e .

COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
 && mkdir -p /data/documents /data/catalog /app/profiles /app/config

# Never run the server as root: it reads a whole document corpus.
RUN useradd --create-home --uid 10001 doccat && chown -R doccat /data /app
USER doccat

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
url='http://127.0.0.1:%s/health' % os.getenv('MCP_PORT','8000'); \
sys.exit(0 if urllib.request.urlopen(url, timeout=4).status == 200 else 1)"

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
