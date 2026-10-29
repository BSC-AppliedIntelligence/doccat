# Common tasks. `make help` lists them.
.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON      ?= python3
VENV        ?= .venv
BIN         := $(VENV)/bin
ROOT        ?= ./documents
CATALOG     ?= ./catalog.jsonl
WORKERS     ?= 8
COMPOSE     := docker compose -f deploy/docker-compose.yml

.PHONY: help venv install dev test lint config config-show optimize profiles \
        index index-dry serve stats docker-build docker-up docker-down \
        docker-index clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the virtualenv
	$(PYTHON) -m venv $(VENV) && $(BIN)/pip install --quiet --upgrade pip

install: venv  ## Install the package and its dependencies
	$(BIN)/pip install -e .

dev: venv  ## Install with the development extras
	$(BIN)/pip install -e '.[dev]'

test:  ## Run the test suite
	$(BIN)/python -m pytest

config:  ## Write a starter doccat.yaml
	$(BIN)/doccat config init

config-show:  ## Print the settings every command will actually use
	$(BIN)/doccat config show

optimize:  ## Design a cataloguing profile from a sample of ROOT
	$(BIN)/doccat optimize --root $(ROOT)

profiles:  ## List the cataloguing profiles
	$(BIN)/doccat profiles list

index:  ## Build/refresh the catalog (ROOT=... WORKERS=...)
	$(BIN)/doccat index --root $(ROOT) --catalog $(CATALOG) -j $(WORKERS)

index-dry:  ## Show which profile each document would get, without calling the LLM
	$(BIN)/doccat index --root $(ROOT) --catalog $(CATALOG) --dry-run

stats:  ## Summarise the catalog
	$(BIN)/doccat stats --catalog $(CATALOG)

serve:  ## Run the MCP server locally
	$(BIN)/doccat serve --docs-root $(ROOT) --catalog $(CATALOG)

docker-build:  ## Build the container image
	$(COMPOSE) build

docker-index:  ## Run the indexing job in a container (one-shot)
	$(COMPOSE) --profile index run --rm indexer

docker-up:  ## Start the MCP server (add --profile proxy for mcpo)
	$(COMPOSE) up -d server

docker-down:  ## Stop the stack
	$(COMPOSE) down

clean:  ## Remove caches and build artefacts
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
