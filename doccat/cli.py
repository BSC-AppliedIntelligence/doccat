#!/usr/bin/env python3
"""`doccat` — build, inspect and serve AI-readable document catalogs.

    doccat config   init|show|path
    doccat optimize --root DIR  -o profiles/mydomain.yaml
    doccat index    --root DIR  [--profile NAME] [--llm-workers N] ...
    doccat json     --input FILE --field texto   [--profile NAME] ...
    doccat profiles list|show|init|validate
    doccat merge    shard-*.jsonl --output catalog.jsonl
    doccat graph    build  --catalog catalog.jsonl
    doccat stats    --catalog catalog.jsonl
    doccat serve    --docs-root DIR --catalog catalog.jsonl --port 8000

Settings come from four places, later ones winning: built-in defaults, a config
file (`--config`, `$DOCCAT_CONFIG`, or ./doccat.yaml), environment variables,
and the flags below. `doccat config show` prints what a command will actually
use. See doccat/config.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

from . import __version__
from .catalog import CatalogStore, load_catalog, save_catalog
from .config import (CONFIG_TEMPLATE, ConfigError, Settings, discover_config_file)
from .llm import LLMConfig, LLMClient
from .pipeline import PipelineOptions, run_pipeline, write_followup_lists
from .profiles import BASE_PROFILE, CORE_FIELDS, RELATIONS_FIELD, ProfileError, load_profiles
from .sources import FileSource, JsonRecordSource

log = logging.getLogger("doccat")


# ---------------------------------------------------------------------------
# Shared argument groups
# ---------------------------------------------------------------------------

def add_llm_args(p: argparse.ArgumentParser, s: Settings) -> None:
    """Defaults come from the resolved Settings, so a flag overrides env overrides file."""
    g = p.add_argument_group("LLM endpoint")
    g.add_argument("--base-url", default=s.llm.base_url,
                   help="OpenAI-compatible endpoint base URL (config llm.base_url, env LLM_BASE_URL)")
    g.add_argument("--model", default=s.llm.model,
                   help="Model name (config llm.model, env LLM_MODEL)")
    g.add_argument("--api-key", default=s.llm.api_key,
                   help="API key, if the endpoint needs one (config llm.api_key, env LLM_API_KEY)")
    g.add_argument("--max-tokens", type=int, default=s.llm.max_tokens,
                   help="Completion token budget; raise for reasoning models (env LLM_MAX_TOKENS)")
    g.add_argument("--temperature", type=float, default=s.llm.temperature,
                   help="Sampling temperature (env LLM_TEMPERATURE)")
    g.add_argument("--timeout", type=float, default=s.llm.timeout,
                   help="Per-request timeout in seconds (env LLM_TIMEOUT)")
    g.add_argument("--think", action="store_true", default=s.llm.think,
                   help="Keep the model's reasoning mode enabled (default: disabled)")
    g.add_argument("--rps", type=float, default=s.llm.requests_per_second,
                   help="Cap the global request rate, requests/second (0 = unlimited, env LLM_RPS)")
    g.add_argument("--bill-to", default=s.llm.bill_to,
                   help="Hugging Face router billing org (sends X-HF-Bill-To, env HF_BILL_TO)")
    g.add_argument("--header", action="append", default=[], metavar="NAME:VALUE",
                   help="Extra HTTP header for LLM requests (repeatable; adds to config llm.headers)")


def add_parallel_args(p: argparse.ArgumentParser, s: Settings) -> None:
    g = p.add_argument_group("parallelism and scale")
    g.add_argument("--llm-workers", "-j", type=int, default=s.indexing.llm_workers,
                   help="Concurrent LLM requests (config indexing.llm_workers, env DOCCAT_LLM_WORKERS)")
    g.add_argument("--extract-workers", type=int, default=s.indexing.extract_workers,
                   help="Processes used for text extraction; 0 = auto (min(cpu, 8)), "
                        "1 = extract inline (env DOCCAT_EXTRACT_WORKERS)")
    g.add_argument("--checkpoint-every", type=int, default=s.indexing.checkpoint_every,
                   help="Rewrite the full catalog every N records (the journal makes every "
                        "single record crash-safe regardless, env DOCCAT_CHECKPOINT_EVERY)")
    g.add_argument("--progress-every", type=int, default=s.indexing.progress_every,
                   help="Log a progress/ETA line every N documents")
    g.add_argument("--shard", default=s.indexing.shard, metavar="I/N",
                   help="Process only shard I of N (e.g. 0/4) — run the other shards on "
                        "other machines and combine them with `doccat merge`")
    g.add_argument("--max-docs", type=int, default=None,
                   help="Stop after N documents (useful for trial runs)")


def add_profile_args(p: argparse.ArgumentParser, s: Settings) -> None:
    g = p.add_argument_group("cataloguing profiles")
    g.add_argument("--profile", default=s.indexing.profile,
                   help="Force one profile for every document instead of auto-routing "
                        "(config indexing.profile, env DOCCAT_PROFILE). Documents catalogued "
                        "under another profile are re-catalogued.")
    g.add_argument("--profiles-dir", action="append",
                   default=[str(d) for d in s.paths.profiles_dir],
                   metavar="DIR", help="Directory of custom profile YAML files (repeatable; "
                                       "config paths.profiles_dir, env DOCCAT_PROFILES_DIR)")
    g.add_argument("--profile-file", action="append", default=[], metavar="FILE",
                   help="A single profile YAML file to load (repeatable)")


def add_catalog_args(p: argparse.ArgumentParser, s: Settings) -> None:
    g = p.add_argument_group("catalog output")
    g.add_argument("--catalog", type=Path, default=s.paths.catalog,
                   help="Output JSONL catalog path (config paths.catalog, env CATALOG_PATH)")
    g.add_argument("--text-cache", type=Path, default=s.paths.text_cache,
                   help="Folder for extracted-text copies (default: 'extracted_text' next to "
                        "the catalog; env TEXT_CACHE)")
    g.add_argument("--no-text-cache", action="store_true", help="Do not keep extracted text copies")
    g.add_argument("--error-log", type=Path, default=s.paths.error_log,
                   help="Folder where unparseable LLM replies are saved for diagnosis "
                        "(default: 'llm_errors' next to the catalog)")
    g.add_argument("--no-error-log", action="store_true", help="Do not save unparseable replies")
    g.add_argument("--force", action="store_true", help="Re-catalogue everything, even unchanged")
    g.add_argument("--dry-run", action="store_true",
                   help="Show what would be catalogued, and with which profile, without "
                        "calling the LLM")


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def build_llm_config(args) -> LLMConfig:
    settings: Settings | None = getattr(args, "settings", None)
    # Headers from the config file first; --header adds to them.
    extra_headers: dict[str, str] = dict(settings.llm.headers) if settings else {}
    for h in args.header:
        name, sep, value = h.partition(":")
        if not sep or not name.strip():
            sys.exit(f"Error: --header must be NAME:VALUE, got {h!r}")
        extra_headers[name.strip()] = value.strip()
    if args.bill_to:
        extra_headers["X-HF-Bill-To"] = args.bill_to
    workers = getattr(args, "llm_workers", 8)
    return LLMConfig(
        base_url=args.base_url, model=args.model, api_key=args.api_key,
        temperature=args.temperature, max_tokens=args.max_tokens, timeout=args.timeout,
        think=args.think, extra_headers=extra_headers,
        max_connections=max(workers * 2, 16), requests_per_second=args.rps,
    )


def build_pipeline_options(args) -> PipelineOptions:
    catalog_dir = args.catalog.resolve().parent
    text_cache = None if args.no_text_cache else (args.text_cache or catalog_dir / "extracted_text")
    error_log = None if args.no_error_log else (args.error_log or catalog_dir / "llm_errors")

    shard = None
    if args.shard:
        try:
            index, total = (int(x) for x in str(args.shard).split("/"))
        except ValueError:
            sys.exit(f"Error: --shard must look like I/N, got {args.shard!r}")
        if not (total > 0 and 0 <= index < total):
            sys.exit(f"Error: --shard I/N needs 0 <= I < N, got {args.shard!r}")
        shard = (index, total)

    return PipelineOptions(
        llm_workers=max(args.llm_workers, 1),
        extract_workers=max(args.extract_workers, 0),
        mode=getattr(args, "mode", "auto"),
        force=args.force,
        text_cache=text_cache,
        error_log=error_log,
        checkpoint_every=args.checkpoint_every,
        profile_name=args.profile,
        progress_every=args.progress_every,
        max_docs=args.max_docs,
        change_detection=getattr(args, "change_detection", "hash"),
        shard=shard,
        dry_run=args.dry_run,
    )


def load_profile_set(args):
    try:
        return load_profiles(extra_dirs=args.profiles_dir or (),
                             files=args.profile_file or ())
    except ProfileError as e:
        sys.exit(f"Profile error: {e}")


def _run(source, args) -> int:
    profiles = load_profile_set(args)
    if args.profile and args.profile not in profiles:
        sys.exit(f"Error: unknown profile {args.profile!r}; available: {profiles.names()}")
    opts = build_pipeline_options(args)

    with CatalogStore(args.catalog, checkpoint_every=opts.checkpoint_every) as store:
        if opts.dry_run:
            stats = run_pipeline(source, store, profiles, None, opts)  # no LLM needed
        else:
            with LLMClient(build_llm_config(args)) as client:
                stats = run_pipeline(source, store, profiles, client, opts)
        records = store.records()

    if not opts.dry_run:
        write_followup_lists(args.catalog, records)
        log.info("Finished in %s: %d catalogued, %d skipped (unchanged), %d with errors, "
                 "%d removed | %d LLM calls",
                 _fmt(stats.seconds), stats.indexed, stats.skipped, stats.failed,
                 stats.removed, stats.llm_calls)
        if stats.by_profile:
            log.info("Profiles used: %s",
                     ", ".join(f"{k}={v}" for k, v in sorted(stats.by_profile.items())))
        if stats.failed:
            log.info("Re-run the same command to retry the %d error record(s).", stats.failed)
        if stats.interrupted:
            log.warning("Run was interrupted — progress is saved; re-run to continue.")
            return 130
    return 0


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_index(args) -> int:
    if not args.root.is_dir():
        sys.exit(f"Error: {args.root} is not a directory")
    source = FileSource(args.root.resolve(), include=args.include, exclude=args.exclude,
                        extensions=args.extension or None, follow_symlinks=args.follow_symlinks)
    return _run(source, args)


def cmd_json(args) -> int:
    if not args.input.is_file():
        sys.exit(f"Error: {args.input} is not a file")
    source = JsonRecordSource(args.input, args.field, limit=args.limit)
    return _run(source, args)


def cmd_profiles(args) -> int:
    if args.profiles_action == "init":
        return _profiles_init(args)
    profiles = load_profile_set(args)
    if args.profiles_action == "validate":
        print(f"OK — {len(profiles)} profile(s) loaded: {', '.join(profiles.names())}")
        return 0
    if args.profiles_action == "show":
        profile = profiles.get(args.name) if args.name in profiles else None
        if profile is None:
            sys.exit(f"Error: unknown profile {args.name!r}; available: {profiles.names()}")
        print(f"# profile: {profile.name}   (from {profile.source})")
        print(f"# {profile.description}")
        print(f"# priority {profile.priority} | excerpt strategy {profile.strategy} "
              f"| ingestion {profile.ingestion or 'from --mode'}")
        print()
        print(profile.build_prompt(label="<document name>",
                                   excerpt="<document text goes here>"))
        return 0
    # list
    print(f"{len(profiles)} profile(s) — highest routing priority first\n")
    for profile in profiles:
        print(profile.summary_line())
        if profile.source != "<builtin>" and not str(profile.source).startswith(
                str(Path(__file__).parent)):
            print(f"{'':<18} source: {profile.source}")
    print("\nUse `doccat profiles show NAME` to see the exact prompt a profile produces.")
    return 0


_PROFILE_TEMPLATE = """# Custom cataloguing profile. Drop this file in a directory and pass it with
# `--profiles-dir DIR` (or DOCCAT_PROFILES_DIR), then select it with
# `--profile {name}` or let its `match` rules route documents to it.
name: {name}
description: Describe the document family this profile is for.
extends: generic          # inherit the core schema; state only your differences
priority: 50              # higher wins when several profiles match

# Remove the `match` block to make this profile selectable only with --profile.
match:
  path_globs:
    - "**/{name}/**"
  # filename_regex: "(invoice|factura)"
  # content_regex: "CONFIDENTIAL"      # evaluated on the first 4000 extracted chars
  # filetypes: [pdf, docx]
  # metadata_equals: {{orga: Senado}}   # JSON sources: match on a record field
  # min_bytes: 100000

excerpt:
  strategy: head_tail     # head_tail | head | full | sampled | map_reduce
  head_chars: 8000
  tail_chars: 2000

llm:
  # model: a-bigger-model-for-this-domain
  # max_tokens: 6000
  # ingestion: parse      # parse | upload | auto

prompt:
  role: >-
    You are a cataloguer of ... . Describe each document so that ... .
  guidance: >-
    Domain rules the model must follow: what to quote verbatim, what never to infer,
    which vocabulary to use.
  doc_types: [report, other]
  fields:
    # Core fields (title, summary, description, doc_type, language, keywords,
    # topics, entities, questions_answered) are inherited — only add your own.
    my_field: what this field should contain
    my_list_field:
      type: list
      description: what each element should be
"""


def _profiles_init(args) -> int:
    name = args.name
    out = Path(args.output or f"{name}.yaml")
    if out.exists() and not args.force_write:
        sys.exit(f"Error: {out} already exists (use --force-write to overwrite)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_PROFILE_TEMPLATE.format(name=name), encoding="utf-8")
    print(f"Wrote {out}\nEdit it, then: doccat profiles show {name} --profile-file {out}")
    return 0


def cmd_config(args) -> int:
    if args.config_action == "init":
        out = Path(args.output)
        if out.exists() and not args.force_write:
            sys.exit(f"Error: {out} already exists (use --force-write to overwrite)")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(CONFIG_TEMPLATE, encoding="utf-8")
        print(f"Wrote {out}\n\nEdit it, then check what it resolves to:\n"
              f"  doccat config show --config {out}")
        return 0

    if args.config_action == "path":
        found = discover_config_file(args.config)
        print(found if found else "(no config file found — using defaults and the environment)")
        return 0

    # show
    settings: Settings = args.settings
    report = settings.as_dict(mask_secrets=not args.reveal)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        import yaml as _yaml
        source = report.pop("_source")
        print(_yaml.safe_dump(report, sort_keys=False, allow_unicode=True, width=88).rstrip())
        print("\n# ---")
        print(f"# config file: {source['config_file'] or '(none — defaults + environment)'}")
        for override in source["env_overrides"] or []:
            print(f"# env override: {override}")
    return 0


def cmd_optimize(args) -> int:
    from .optimize import design_profile

    if args.input:
        source = JsonRecordSource(args.input, args.field, limit=args.limit)
        label = str(args.input)
    else:
        if not args.root or not Path(args.root).is_dir():
            sys.exit(f"Error: {args.root} is not a directory (pass --root, or --input for a "
                     "JSON dump)")
        source = FileSource(Path(args.root).resolve(), include=args.include,
                            exclude=args.exclude)
        label = str(args.root)

    settings: Settings = args.settings
    # Existing profiles are read only to know which names `extends` may target.
    # The profiles directory is usually where this command is about to *write*,
    # so one that does not exist yet is normal, not an error.
    known_dirs = [d for d in (args.profiles_dir or []) if Path(d).is_dir()]

    log.info("Designing a cataloguing profile for %s", label)
    with LLMClient(build_llm_config(args)) as client:
        try:
            result = design_profile(
                source, client,
                sample_size=args.sample, sample_chars=settings.optimize.sample_chars,
                refine_rounds=args.refine, name=args.name, seed=args.seed,
                trial_docs=args.trial_docs, extra_profile_dirs=known_dirs,
            )
        except ProfileError as e:
            sys.exit(f"Error: {e}")
        except (ValueError, TypeError) as e:
            sys.exit(f"Error: the model did not return a usable analysis of the sample "
                     f"({e}). Try a larger --max-tokens, a stronger --model, or "
                     f"--log-level DEBUG to see the reply.")

    if args.print_only:
        print(result.yaml_text)
        return 0

    out = Path(args.output) if args.output else _default_profile_path(args, result.profile_name)
    if out.exists() and not args.force_write:
        sys.exit(f"Error: {out} already exists (use --force-write to overwrite, "
                 f"or -o to choose another path)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.yaml_text, encoding="utf-8")

    _report_optimize(result, out, args)
    return 0


def _default_profile_path(args, profile_name: str) -> Path:
    directory = Path(args.profiles_dir[0]) if args.profiles_dir else Path("profiles")
    return directory / f"{profile_name}.yaml"


def _report_optimize(result, out: Path, args) -> None:
    import yaml as _yaml

    profile = result.profile
    parent = result.analysis.get("extends") or BASE_PROFILE
    # Fields the agent itself proposed, as opposed to those inherited from the
    # profile it chose to extend — worth distinguishing when reviewing a draft.
    own = list((_yaml.safe_load(result.yaml_text).get("prompt") or {}).get("fields") or {})
    inherited = [f for f in profile.fields
                if f not in CORE_FIELDS and f not in own and f != RELATIONS_FIELD]

    print(f"\nWrote {out}")
    print(f"  profile     {result.profile_name} (extends {parent})")
    print(f"  domain      {result.analysis.get('domain', '-')}")
    print(f"  families    {', '.join(result.analysis.get('document_families') or []) or '-'}")
    print(f"  strategy    {profile.strategy}  ({result.analysis.get('strategy_reason', '-')})")
    print(f"  doc types   {_trim(profile.doc_types)}")
    print(f"  new fields  {', '.join(own) or '(none)'}")
    if inherited:
        print(f"  inherited   {_trim(inherited)}  (from {parent})")
    if profile.graph_enabled:
        print(f"  graph       entities: {_trim(profile.graph.entity_types) or '(open)'}")
        print(f"              relations: {_trim(profile.graph.relation_types) or '(open)'}")
    print(f"  sampled     {len(result.sample)} document(s), {result.llm_calls} LLM call(s)")
    for entry in result.rounds:
        trial = entry.get("trial") or {}
        bits = [f"round {entry['round']}"]
        if trial.get("always_empty"):
            bits.append(f"always empty: {_trim(trial['always_empty'])}")
        if trial.get("no_variation"):
            bits.append(f"no variation: {_trim(trial['no_variation'])}")
        if trial.get("relations_without_matching_entities"):
            bits.append("unmatched relations: "
                       f"{trial['relations_without_matching_entities']}")
        if entry.get("error"):
            bits.append(f"stopped: {entry['error']}")
        print("  refine      " + " | ".join(bits))
        if entry.get("changes"):
            print(f"              {entry['changes'][:160]}")

    target = args.input or args.root
    print("\nReview it, then use it:")
    print(f"  doccat profiles show {result.profile_name} --profile-file {out}")
    print(f"  doccat index --root {target} --profiles-dir {out.parent} --dry-run")


def _trim(values, limit: int = 8) -> str:
    """Join a list for display, summarising the tail when it is long."""
    values = list(values)
    shown = ", ".join(str(v) for v in values[:limit])
    return shown + (f" (+{len(values) - limit} more)" if len(values) > limit else "")


def cmd_merge(args) -> int:
    merged: dict[str, dict] = {}
    for path in args.inputs:
        part = load_catalog(Path(path))
        log.info("%s: %d record(s)", path, len(part))
        merged.update(part)
    save_catalog(args.output, merged)
    print(f"Merged {len(args.inputs)} catalog(s) into {args.output}: {len(merged)} records")
    return 0


def cmd_graph(args) -> int:
    if args.graph_action == "build":
        return _graph_build(args)
    return _graph_stats(args)


def _graph_build(args) -> int:
    from .graph import build_corpus_graph, save_corpus_graph, summarize_communities

    catalog = load_catalog(args.catalog)
    if not catalog:
        sys.exit(f"Error: {args.catalog} is missing or empty — run `doccat index` first")

    settings: Settings = args.settings
    out_dir = Path(args.out) if args.out else settings.graph.resolved_out_dir(args.catalog)
    detect = settings.graph.detect_communities if args.communities is None else args.communities
    min_size = args.min_community_size or settings.graph.min_community_size
    summarize = args.summarize or settings.graph.summarize
    seed = args.seed if args.seed is not None else settings.graph.community_seed

    log.info("Building the corpus graph from %d catalogued document(s)", len(catalog))
    graph, stats = build_corpus_graph(catalog.values(), detect_communities=detect,
                                      min_community_size=min_size, community_seed=seed)
    log.info("Resolved %d document(s) into %d node(s), %d edge(s), %d community/communities "
             "(%d entity mention(s), %d relation mention(s), %d self-relation(s) dropped)",
             stats.documents_scanned, stats.nodes, stats.edges, stats.communities,
             stats.raw_entity_mentions, stats.raw_relation_mentions,
             stats.self_relations_dropped)

    if summarize and graph.communities:
        log.info("Summarising %d community/communities (one LLM call each)",
                 len(graph.communities))
        with LLMClient(build_llm_config(args)) as client:
            done = summarize_communities(graph, client)
        log.info("Summarised %d/%d communities", done, len(graph.communities))
    elif summarize:
        log.info("No communities to summarise")

    save_corpus_graph(graph, out_dir)
    print(f"Wrote {out_dir} ({stats.nodes} nodes, {stats.edges} edges, "
         f"{stats.communities} communities)")
    if not stats.edges:
        # `entities` is always extracted (it's a core field), so nodes alone
        # prove nothing; it's the absence of *relations* that means no profile
        # in this catalog has a `graph:` block doing real extraction yet.
        print("No relations found — graph extraction needs a profile with a `graph:` "
             "block (see `doccat profiles show legal`), and the affected documents "
             "re-indexed under it. Nothing to build yet.")
    return 0


def _graph_stats(args) -> int:
    from .graph import load_corpus_graph

    settings: Settings = args.settings
    out_dir = Path(args.out) if args.out else settings.graph.resolved_out_dir(args.catalog)
    graph = load_corpus_graph(out_dir)
    if not len(graph):
        sys.exit(f"Error: no graph at {out_dir} — run `doccat graph build` first")
    print(json.dumps(graph.overview(), ensure_ascii=False, indent=2))
    return 0


def cmd_stats(args) -> int:
    catalog = load_catalog(args.catalog)
    if not catalog:
        sys.exit(f"Error: {args.catalog} is missing or empty")
    records = list(catalog.values())
    report = {
        "catalog": str(args.catalog),
        "documents": len(records),
        "by_status": dict(Counter(r.get("status", "unknown") for r in records)),
        "by_profile": dict(Counter(r.get("profile", "(pre-profiles)") for r in records)),
        "by_filetype": dict(Counter(r.get("filetype", "?") for r in records)),
        "by_doc_type": dict(Counter(r.get("doc_type", "unknown") for r in records).most_common(20)),
        "by_ingestion": dict(Counter(r.get("ingestion", "-") for r in records)),
        "needs_ocr": sum(1 for r in records if r.get("needs_ocr")),
        "incomplete": sum(1 for r in records if r.get("incomplete_fields")),
        "last_indexed_at": max((r.get("indexed_at", "") for r in records), default=None),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args) -> int:
    settings: Settings = args.settings
    # server.py reads its configuration from the environment, so hand it the
    # resolved settings that way — a config file therefore configures the
    # server as well as the indexer, with no second mechanism.
    os.environ.setdefault("DOCS_ROOT", str(args.docs_root))
    os.environ.setdefault("CATALOG_PATH", str(args.catalog))
    os.environ.setdefault("TEXT_CACHE", str(args.text_cache or settings.paths.resolved_text_cache()))
    graph_dir = args.graph_dir or settings.graph.resolved_out_dir(args.catalog)
    os.environ.setdefault("GRAPH_DIR", str(graph_dir))
    os.environ.setdefault("MCP_HOST", args.host)
    os.environ.setdefault("MCP_PORT", str(args.port))
    os.environ.setdefault("MCP_STATELESS", "true" if settings.server.stateless else "false")
    args.text_cache = args.text_cache or settings.paths.resolved_text_cache()
    sys.argv = [
        "server.py", "--docs-root", str(args.docs_root), "--catalog", str(args.catalog),
        "--host", args.host, "--port", str(args.port), "--graph-dir", str(graph_dir),
    ]
    if args.text_cache:
        sys.argv += ["--text-cache", str(args.text_cache)]
    import server  # imported late: the server's deps are not needed for indexing
    server.main()
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser(settings: Settings | None = None) -> argparse.ArgumentParser:
    """Build the parser with every default taken from the resolved settings."""
    s = settings if settings is not None else Settings.load()
    parser = argparse.ArgumentParser(
        prog="doccat",
        description="Build, inspect and serve AI-readable document catalogs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", action="version", version=f"doccat {__version__}")
    parser.add_argument("--config", "-c", default=None, metavar="FILE",
                        help="Configuration file (default: $DOCCAT_CONFIG, then ./doccat.yaml, "
                             "~/.config/doccat/config.yaml, /etc/doccat/config.yaml). "
                             "`doccat config show` prints what is in effect.")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"),
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.set_defaults(settings=s)
    sub = parser.add_subparsers(dest="command", required=True)

    # index -----------------------------------------------------------------
    p_index = sub.add_parser("index", help="Catalogue a folder tree of documents",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_index.add_argument("--root", type=Path, default=s.paths.docs_root,
                         help="Folder containing the documents (config paths.docs_root, env DOCS_ROOT)")
    p_index.add_argument("--mode", choices=["auto", "parse", "upload"], default=s.indexing.mode,
                         help="How to give documents to the LLM: extract text locally, upload "
                              "the raw file, or auto (parse, upload only when no text is "
                              "extractable). A profile's llm.ingestion overrides this.")
    p_index.add_argument("--include", action="append", default=list(s.indexing.include),
                         metavar="GLOB", help="Only index paths matching this glob (repeatable)")
    p_index.add_argument("--exclude", action="append", default=list(s.indexing.exclude),
                         metavar="GLOB", help="Skip paths matching this glob (repeatable)")
    p_index.add_argument("--extension", action="append", default=[], metavar="EXT",
                         help="Restrict to these extensions (repeatable, default: all supported)")
    p_index.add_argument("--follow-symlinks", action="store_true")
    p_index.add_argument("--change-detection", choices=["hash", "mtime"],
                         default=s.indexing.change_detection,
                         help="How to detect changed files between runs: exact content hash, "
                              "or the cheaper mtime+size (worth it above ~100k files)")
    add_profile_args(p_index, s)
    add_llm_args(p_index, s)
    add_parallel_args(p_index, s)
    add_catalog_args(p_index, s)
    p_index.set_defaults(func=cmd_index)

    # json ------------------------------------------------------------------
    p_json = sub.add_parser("json", help="Catalogue a JSON array dump of text records",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_json.add_argument("--input", required=True, type=Path, help="JSON file (array of objects)")
    p_json.add_argument("--field", default="texto", help="Field holding the text to catalogue")
    p_json.add_argument("--limit", type=int, default=None, help="Only read the first N records")
    add_profile_args(p_json, s)
    add_llm_args(p_json, s)
    add_parallel_args(p_json, s)
    add_catalog_args(p_json, s)
    p_json.set_defaults(func=cmd_json, mode="parse")

    # profiles --------------------------------------------------------------
    p_prof = sub.add_parser("profiles", help="List, inspect, scaffold and validate profiles")
    psub = p_prof.add_subparsers(dest="profiles_action", required=True)
    for action, helptext in (("list", "List every available profile"),
                             ("validate", "Load all profiles and report errors")):
        sp = psub.add_parser(action, help=helptext)
        add_profile_args(sp, s)
        sp.set_defaults(profiles_action=action)
    sp_show = psub.add_parser("show", help="Print the exact prompt a profile produces")
    sp_show.add_argument("name")
    add_profile_args(sp_show, s)
    sp_show.set_defaults(profiles_action="show")
    sp_init = psub.add_parser("init", help="Scaffold a custom profile YAML file")
    sp_init.add_argument("name", help="Name of the new profile")
    sp_init.add_argument("--output", "-o", help="Where to write it (default: ./<name>.yaml)")
    sp_init.add_argument("--force-write", action="store_true", help="Overwrite an existing file")
    sp_init.set_defaults(profiles_action="init")
    p_prof.set_defaults(func=cmd_profiles)

    # config ----------------------------------------------------------------
    p_config = sub.add_parser("config", help="Create and inspect the configuration file")
    csub = p_config.add_subparsers(dest="config_action", required=True)
    c_init = csub.add_parser("init", help="Write a commented starter config file")
    c_init.add_argument("--output", "-o", default="doccat.yaml", help="Where to write it")
    c_init.add_argument("--force-write", action="store_true", help="Overwrite an existing file")
    c_init.set_defaults(config_action="init")
    c_show = csub.add_parser("show", help="Print the settings a command would actually use")
    c_show.add_argument("--format", choices=["yaml", "json"], default="yaml")
    c_show.add_argument("--reveal", action="store_true",
                        help="Print secrets in full instead of masking them")
    c_show.set_defaults(config_action="show")
    c_path = csub.add_parser("path", help="Print which config file would be loaded")
    c_path.set_defaults(config_action="path")
    p_config.set_defaults(func=cmd_config)

    # optimize --------------------------------------------------------------
    p_opt = sub.add_parser(
        "optimize", help="Analyse a sample of the repository and design a cataloguing profile",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a stratified sample of the documents, works out what the corpus "
                    "is, and writes a cataloguing profile tuned to it — then tries that "
                    "profile on real documents and revises it.")
    p_opt.add_argument("--root", type=Path, default=s.paths.docs_root,
                       help="Folder containing the documents")
    p_opt.add_argument("--input", type=Path, default=None,
                       help="Analyse a JSON array dump instead of a folder")
    p_opt.add_argument("--field", default="texto", help="With --input: the field holding the text")
    p_opt.add_argument("--limit", type=int, default=None, help="With --input: read first N records")
    p_opt.add_argument("--sample", type=int, default=s.optimize.sample_size,
                       help="How many documents to read (config optimize.sample_size)")
    p_opt.add_argument("--refine", type=int, default=s.optimize.refine_rounds,
                       help="Critique-and-improve rounds; 0 skips the trial entirely")
    p_opt.add_argument("--trial-docs", type=int, default=3,
                       help="Documents catalogued for real in each refine round")
    p_opt.add_argument("--name", default=None, help="Name for the profile (default: the agent's)")
    p_opt.add_argument("--seed", type=int, default=0, help="Sampling seed (same seed, same sample)")
    p_opt.add_argument("--output", "-o", default=None,
                       help="Where to write the profile (default: <profiles-dir>/<name>.yaml)")
    p_opt.add_argument("--force-write", action="store_true", help="Overwrite an existing file")
    p_opt.add_argument("--print-only", action="store_true",
                       help="Print the profile to stdout instead of writing a file")
    p_opt.add_argument("--include", action="append", default=[], metavar="GLOB",
                       help="Only sample paths matching this glob (repeatable)")
    p_opt.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                       help="Skip paths matching this glob (repeatable)")
    add_profile_args(p_opt, s)
    add_llm_args(p_opt, s)
    p_opt.set_defaults(func=cmd_optimize)

    # merge -----------------------------------------------------------------
    p_merge = sub.add_parser("merge", help="Merge shard catalogs into one")
    p_merge.add_argument("inputs", nargs="+", help="Catalog JSONL files to merge")
    p_merge.add_argument("--output", "-o", required=True, type=Path, help="Merged catalog path")
    p_merge.set_defaults(func=cmd_merge)

    # graph -------------------------------------------------------------
    p_graph = sub.add_parser(
        "graph", help="Consolidate per-document relations into one corpus graph",
        description="Resolves every document's local entities/relations into one graph "
                    "across the corpus — see docs/GRAPH.md.")
    gsub = p_graph.add_subparsers(dest="graph_action", required=True)

    g_build = gsub.add_parser("build", help="Build (or rebuild) the corpus graph")
    g_build.add_argument("--catalog", type=Path, default=s.paths.catalog)
    g_build.add_argument("--out", "-o", default=None,
                         help="Output directory (config graph.out_dir; default: "
                              "'graph' next to the catalog)")
    g_build.add_argument("--communities", action=argparse.BooleanOptionalAction, default=None,
                         help="Detect communities (config graph.detect_communities)")
    g_build.add_argument("--min-community-size", type=int, default=None,
                         help="Smallest cluster reported as a community (config "
                              "graph.min_community_size, default 2)")
    g_build.add_argument("--summarize", action="store_true", default=None,
                         help="Generate a one-paragraph summary per community — one LLM "
                              "call each (config graph.summarize)")
    g_build.add_argument("--seed", type=int, default=None,
                         help="Community-detection seed (config graph.community_seed)")
    add_llm_args(g_build, s)
    g_build.set_defaults(graph_action="build")

    g_stats = gsub.add_parser("stats", help="Summarise an already-built corpus graph")
    g_stats.add_argument("--catalog", type=Path, default=s.paths.catalog,
                         help="Only used to locate the default graph directory")
    g_stats.add_argument("--out", "-o", default=None, help="Graph directory to read")
    g_stats.set_defaults(graph_action="stats")

    p_graph.set_defaults(func=cmd_graph)

    # stats -----------------------------------------------------------------
    p_stats = sub.add_parser("stats", help="Summarise a catalog")
    p_stats.add_argument("--catalog", type=Path, default=s.paths.catalog)
    p_stats.set_defaults(func=cmd_stats)

    # serve -----------------------------------------------------------------
    p_serve = sub.add_parser("serve", help="Run the MCP server over streamable HTTP")
    p_serve.add_argument("--docs-root", type=Path, default=s.paths.docs_root)
    p_serve.add_argument("--catalog", type=Path, default=s.paths.catalog)
    p_serve.add_argument("--text-cache", type=Path, default=s.paths.text_cache)
    p_serve.add_argument("--graph-dir", type=Path, default=None,
                         help="Corpus graph directory built by `doccat graph build` "
                              "(config graph.out_dir; default: 'graph' next to the catalog)")
    p_serve.add_argument("--host", default=s.server.host)
    p_serve.add_argument("--port", type=int, default=s.server.port)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def _preparse(argv: list[str] | None) -> tuple[str | None, str]:
    """Read --config/--log-level before the real parser exists.

    The real parser's defaults come from the settings, and the settings come
    from the config file, so the config file has to be known first.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", "-c", default=None)
    pre.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    known, _ = pre.parse_known_args(argv)
    return known.config, known.log_level


def main(argv: list[str] | None = None) -> int:
    config_file, log_level = _preparse(argv)
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
    try:
        settings = Settings.load(config_file)
    except ConfigError as e:
        sys.exit(f"Configuration error: {e}")
    if settings.config_file:
        log.debug("Configuration loaded from %s", settings.config_file)

    args = build_parser(settings).parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.log_level != "DEBUG":
        # httpx logs one INFO line per request; at thousands of documents that
        # buries the progress output. Raise --log-level DEBUG to get it back.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
