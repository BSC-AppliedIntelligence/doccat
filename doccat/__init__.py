"""doccat — domain-aware, parallel document cataloguing for doc_catalog_mcp.

The package splits the original monolithic `indexer.py` into reusable parts:

    doccat.config     one settings model behind the config file, env and flags
    doccat.llm        thread-safe OpenAI-compatible client (pooled, rate-limit aware)
    doccat.jsonparse  tolerant parsing of LLM JSON replies
    doccat.profiles   domain/document-kind cataloguing profiles (YAML) + routing
    doccat.catalog    crash-safe, concurrency-safe JSONL catalog store
    doccat.sources    work sources (filesystem tree, JSON array dump)
    doccat.pipeline   parallel extract + LLM pipeline
    doccat.optimize   agent that designs a cataloguing profile from a corpus sample
    doccat.graph      corpus-level graph consolidation (entity resolution, communities)
    doccat.cli        the `doccat` command line

`indexer.py` and `json_indexer.py` remain as thin, backwards-compatible CLIs.
"""

__version__ = "0.4.0"

__all__ = ["__version__"]
