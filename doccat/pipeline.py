"""Parallel cataloguing pipeline.

Cataloguing one document is a short burst of CPU (parse the PDF) followed by a
long wait on the network (the LLM call). Doing that serially is what made the
original indexer take a day per few thousand documents. The pipeline overlaps
both:

    scan ─┬─ hash files in a thread pool
          │
          ├─ N LLM worker threads, each:
          │     1. hands the file to a *process* pool for extraction
          │        (pypdfium2/openpyxl are CPU-bound and hold the GIL)
          │     2. picks the cataloguing profile for the document
          │     3. runs one — or, for long documents, map+reduce — LLM calls
          │     4. hands the finished record to the CatalogStore
          │
          └─ CatalogStore journals every record and checkpoints periodically

Concurrency is bounded on both axes (`--llm-workers`, `--extract-workers`), the
shared `RateLimitGate` pauses every worker when the endpoint pushes back, and a
Ctrl-C drains the in-flight work and saves rather than losing it.

Runs are resumable and shardable: unchanged documents are skipped by content
hash, an interrupted run replays its journal, and `--shard i/n` splits one
corpus deterministically across several machines.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import re
import signal
import threading
import time
import zlib
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from extractors import MIME_TYPES

from .catalog import CatalogStore, write_path_list
from .jsonparse import parse_metadata
from .llm import LLMClient, LLMError
from .profiles import RELATIONS_FIELD, MatchContext, Profile, ProfileSet
from .sources import Extraction, WorkItem, extract_job

log = logging.getLogger("doccat.pipeline")

#: A PDF averaging fewer than this many extracted characters per page has no
#: usable text layer (scanned / image-only).
OCR_MIN_CHARS_PER_PAGE = 20
#: Files larger than this are never uploaded (base64 inflates size by ~33%).
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
#: Statuses that are retried on the next run instead of being skipped.
RETRYABLE_STATUSES = frozenset({
    "legacy_format", "extraction_error", "metadata_parse_error", "llm_error",
    "empty", "needs_ocr", "error",
})


@dataclass
class PipelineOptions:
    """Everything that is not the source, the profiles or the LLM connection."""

    llm_workers: int = 8
    extract_workers: int = 0          # 0 = auto (min(cpu_count, 8)); 1 = inline
    mode: str = "auto"                # parse | upload | auto
    force: bool = False
    text_cache: Path | None = None
    error_log: Path | None = None
    checkpoint_every: int = 50
    profile_name: str | None = None   # force one profile instead of auto-routing
    prune_deleted: bool = True
    progress_every: int = 20
    max_docs: int | None = None
    change_detection: str = "hash"    # hash (exact) | mtime (fast on huge trees)
    shard: tuple[int, int] | None = None   # (index, total)
    dry_run: bool = False
    retry_statuses: frozenset[str] = RETRYABLE_STATUSES


@dataclass
class Stats:
    """Outcome of a run."""

    total: int = 0
    todo: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    removed: int = 0
    llm_calls: int = 0
    by_profile: dict[str, int] = field(default_factory=dict)
    interrupted: bool = False
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class _Progress:
    """Thread-safe counters + a rate/ETA line for long runs."""

    def __init__(self, total: int, every: int) -> None:
        self.total = total
        self.every = max(every, 1)
        self.done = 0
        self.started = time.monotonic()
        self._lock = threading.Lock()

    def tick(self, label: str, status: str, profile: str) -> None:
        with self._lock:
            self.done += 1
            done, total = self.done, self.total
            elapsed = time.monotonic() - self.started
        if done % self.every and done != total:
            return
        rate = done / elapsed if elapsed else 0.0
        eta = (total - done) / rate if rate else 0.0
        log.info("[%d/%d] %.2f doc/s | ETA %s | last: %s (%s, profile=%s)",
                 done, total, rate, _fmt_duration(eta), label, status, profile)


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def needs_ocr(filetype: str, text: str, units: int) -> bool:
    """True when a PDF has pages but virtually no extractable text per page."""
    if filetype != "pdf" or units <= 0:
        return False
    real_text = re.sub(r"\[page \d+\]", "", text).strip()
    return (len(real_text) / units) < OCR_MIN_CHARS_PER_PAGE


def _shard_of(doc_id: str, total: int) -> int:
    """Deterministic, stable shard assignment (same everywhere, no coordination)."""
    return zlib.crc32(doc_id.encode("ascii")) % total


# ---------------------------------------------------------------------------
# Per-document work
# ---------------------------------------------------------------------------

class _Worker:
    """Cataloguing logic for a single document. Stateless apart from its collaborators."""

    def __init__(self, source, profiles: ProfileSet, client: LLMClient,
                 opts: PipelineOptions, stats: Stats, stats_lock: threading.Lock) -> None:
        self.source = source
        self.profiles = profiles
        self.client = client
        self.opts = opts
        self.stats = stats
        self._stats_lock = stats_lock
        self.forced_profile: Profile | None = (
            profiles.get(opts.profile_name) if opts.profile_name else None)

    # -- profile routing ---------------------------------------------------
    def choose_profile(self, item: WorkItem, text: str) -> Profile:
        if self.forced_profile is not None:
            return self.forced_profile
        ctx = MatchContext(
            # `path` is the catalog path — "folder/file.pdf" for a tree,
            # "dump.json#key" for a JSON record — so routing rules read the same
            # way whatever the source.
            path=item.base_record.get("path") or item.key,
            filetype=item.filetype, size_bytes=item.size_bytes,
            source_kind=item.source_kind, text_head=text[:4000],
            metadata=item.metadata,
        )
        return self.profiles.select(ctx)

    # -- LLM calls ---------------------------------------------------------
    def _count_call(self) -> None:
        with self._stats_lock:
            self.stats.llm_calls += 1

    def _complete(self, profile: Profile, prompt: str, *, file_path: Path | None = None) -> str:
        self._count_call()
        return self.client.complete(prompt, file_path=file_path,
                                    mime_types=MIME_TYPES, **profile.llm_kwargs())

    def get_metadata(self, item: WorkItem, profile: Profile, text: str,
                     mode: str) -> tuple[dict, str, str | None]:
        """Returns (metadata, ingestion_used, raw_reply_for_diagnosis)."""
        path = item.path
        can_upload = (path is not None and path.is_file()
                      and item.size_bytes <= MAX_UPLOAD_BYTES)
        mode = profile.ingestion or mode
        dedicated = profile.graph_enabled and profile.graph.dedicated_pass
        exclude = frozenset({RELATIONS_FIELD}) if dedicated else frozenset()

        if mode == "upload" and can_upload:
            try:
                raw = self._complete(profile, profile.build_prompt(
                    item.label, exclude_fields=exclude), file_path=path)
                metadata = profile.normalize(parse_metadata(raw))
                if dedicated:
                    self._run_dedicated_relations(profile, item, excerpt="", metadata=metadata)
                return metadata, "upload", raw
            except (LLMError, ValueError) as e:
                log.warning("Upload ingestion failed for %s (%s) — falling back to parse",
                            item.label, e)

        if not text.strip():
            if mode in ("auto", "upload") and can_upload:
                raw = self._complete(profile, profile.build_prompt(
                    item.label, exclude_fields=exclude), file_path=path)
                metadata = profile.normalize(parse_metadata(raw))
                if dedicated:
                    self._run_dedicated_relations(profile, item, excerpt="", metadata=metadata)
                return metadata, "upload", raw
            raise ValueError("no extractable text and upload not attempted")

        if profile.wants_map_reduce(text):
            chunks = profile.chunks(text)
            notes = [self._complete(profile, profile.build_map_prompt(
                item.label, chunk, i, len(chunks))) for i, chunk in enumerate(chunks, 1)]
            raw = self._complete(profile, profile.build_reduce_prompt(item.label, notes))
            metadata = profile.normalize(parse_metadata(raw))
            metadata["indexed_chunks"] = len(chunks)
            if dedicated:
                self._run_dedicated_relations(profile, item, excerpt="\n\n".join(notes),
                                              metadata=metadata)
            return metadata, "map_reduce", raw

        excerpt = profile.excerpt(text)
        raw = self._complete(profile, profile.build_prompt(item.label, excerpt,
                                                            exclude_fields=exclude))
        metadata = profile.normalize(parse_metadata(raw))
        if dedicated:
            self._run_dedicated_relations(profile, item, excerpt=excerpt, metadata=metadata)
        return metadata, "parse", raw

    def _run_dedicated_relations(self, profile: Profile, item: WorkItem, *,
                                 excerpt: str, metadata: dict) -> None:
        """Second LLM call for `graph.dedicated_pass` profiles; merges into `metadata` in place.

        Best-effort: a failure here is logged and leaves `relations` empty
        rather than failing the whole document — the main metadata call
        already succeeded by the time this runs, and losing relations on one
        document is far cheaper than losing the document.
        """
        relations_prompt = profile.build_relations_prompt(
            item.label, excerpt, metadata.get("entities"))
        try:
            raw = self._complete(profile, relations_prompt)
            parsed = parse_metadata(raw)
            spec = profile.fields[RELATIONS_FIELD]
            metadata[RELATIONS_FIELD] = spec.coerce(parsed.get(RELATIONS_FIELD, []))
        except (LLMError, ValueError, TypeError) as e:
            log.warning("Dedicated relation extraction failed for %s: %s", item.label, e)
            metadata.setdefault(RELATIONS_FIELD, [])

    # -- the whole document ------------------------------------------------
    def process(self, item: WorkItem, extraction: Extraction) -> dict:
        record: dict = dict(item.base_record)
        record.update({
            "pages_or_paragraphs": extraction.units,
            "char_count": len(extraction.text),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "status": "ok",
        })

        text = extraction.text
        profile = self.choose_profile(item, text)
        record["profile"] = profile.name

        if self.opts.text_cache is not None and text.strip():
            try:
                record["extracted_text_path"] = self.source.save_text_cache(
                    self.opts.text_cache, item, text)
            except OSError as e:
                log.warning("Could not write text cache for %s: %s", item.label, e)

        if extraction.legacy:
            record["status"] = "legacy_format"
            record["extraction_error"] = extraction.error
            record["summary"] = ("Legacy binary Office file (.doc/.ppt/.xls, possibly "
                                 "renamed) — install LibreOffice or convert it manually, "
                                 "then re-run the indexer.")
            return record

        if needs_ocr(item.filetype, text, extraction.units):
            record["needs_ocr"] = True
            if (profile.ingestion or self.opts.mode) != "upload":
                record["status"] = "needs_ocr"
                record["summary"] = ("PDF with no usable text layer (scanned/image-only) "
                                     "— requires OCR or a multimodal model via "
                                     "--mode upload.")
                return record

        raw: str | None = None
        try:
            metadata, ingestion, raw = self.get_metadata(item, profile, text, self.opts.mode)
            record.update(metadata)
            record["ingestion"] = ingestion
            dropped = profile.filter_relations(record)
            if dropped:
                record["relations_below_confidence"] = dropped
            missing = profile.missing_core_fields(record)
            if missing:
                record["incomplete_fields"] = missing
        except (ValueError, TypeError) as e:
            log.error("Could not obtain/parse LLM metadata for %s: %s", item.label, e)
            record["status"] = "metadata_parse_error"
            record["error_detail"] = str(e)[:500]
            self._save_raw(record, item, raw)
        except LLMError as e:
            log.error("LLM error for %s: %s", item.label, e)
            record["status"] = "llm_error"
            record["error_detail"] = str(e)[:500]

        if extraction.error:
            record["extraction_error"] = extraction.error
            # A failed local extraction that the LLM rescued (upload ingestion)
            # stays "ok"; only an otherwise-failed record is relabelled.
            if record["status"] != "ok":
                record["status"] = "extraction_error"
        return record

    def _save_raw(self, record: dict, item: WorkItem, raw: str | None) -> None:
        if raw is None or self.opts.error_log is None:
            return
        try:
            self.opts.error_log.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", item.key).strip("_") or "record"
            out = self.opts.error_log / f"{safe}_{item.doc_id}.txt"
            out.write_text(raw, encoding="utf-8")
            record["raw_reply_path"] = str(out)
        except OSError as e:  # pragma: no cover - best effort
            log.warning("Could not save raw reply for %s: %s", item.label, e)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def select_work(items: list[WorkItem], store: CatalogStore,
                opts: PipelineOptions) -> tuple[list[WorkItem], int]:
    """Split scanned items into (to do, skipped) according to change keys and status."""
    todo: list[WorkItem] = []
    skipped = 0
    for item in items:
        if opts.shard and _shard_of(item.doc_id, opts.shard[1]) != opts.shard[0]:
            continue
        existing = store.get(item.doc_id)
        if existing and not opts.force:
            unchanged = (existing.get("change_key") or existing.get("content_hash")) == item.change_key
            healthy = existing.get("status") not in opts.retry_statuses
            # Naming a profile explicitly re-catalogues anything indexed under
            # a different one, so switching profiles does not need --force.
            profile_ok = not opts.profile_name or existing.get("profile") == opts.profile_name
            if unchanged and healthy and profile_ok:
                skipped += 1
                continue
        todo.append(item)
    if opts.max_docs is not None:
        todo = todo[: opts.max_docs]
    return todo, skipped


def run_pipeline(source, store: CatalogStore, profiles: ProfileSet, client: LLMClient,
                 opts: PipelineOptions) -> Stats:
    """Catalogue everything `source` yields, in parallel, into `store`."""
    started = time.monotonic()
    stats = Stats()
    stats_lock = threading.Lock()

    items = source.scan(hash_workers=max(opts.llm_workers, 8),
                        change_detection=opts.change_detection)
    stats.total = len(items)

    if opts.prune_deleted and not opts.shard and opts.max_docs is None:
        gone = store.ids() - {i.doc_id for i in items}
        if gone and getattr(source, "kind", "") == "files":
            stats.removed = store.remove(gone)
            if stats.removed:
                log.info("Removed %d record(s) for documents no longer present", stats.removed)

    todo, stats.skipped = select_work(items, store, opts)
    stats.todo = len(todo)
    log.info("%d document(s) to catalogue, %d unchanged and skipped", len(todo), stats.skipped)

    if opts.dry_run:
        _report_plan(todo, profiles, opts)
        stats.seconds = time.monotonic() - started
        return stats
    if not todo:
        stats.seconds = time.monotonic() - started
        return stats

    if opts.text_cache is not None:
        opts.text_cache.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()
    _install_sigint(stop)

    extract_workers = opts.extract_workers or min(os.cpu_count() or 4, 8)
    worker = _Worker(source, profiles, client, opts, stats, stats_lock)
    progress = _Progress(len(todo), opts.progress_every)

    pool: ProcessPoolExecutor | None = None
    inline_extraction = extract_workers <= 1 or getattr(source, "kind", "") != "files"
    if not inline_extraction:
        pool = ProcessPoolExecutor(max_workers=extract_workers,
                                   mp_context=_mp_context())
        log.info("Extraction: %d process(es) | LLM: %d thread(s)",
                 extract_workers, opts.llm_workers)
    else:
        log.info("Extraction: inline | LLM: %d thread(s)", opts.llm_workers)

    def handle(item: WorkItem) -> dict | None:
        if stop.is_set():
            return None
        if pool is not None and item.path is not None:
            text, units, error, legacy = pool.submit(extract_job, str(item.path)).result()
            extraction = Extraction(text=text, units=units, error=error, legacy=legacy)
        else:
            extraction = source.load_text(item)
        if stop.is_set():
            return None
        return worker.process(item, extraction)

    try:
        with ThreadPoolExecutor(max_workers=opts.llm_workers,
                                thread_name_prefix="doccat") as executor:
            futures: dict[Future, WorkItem] = {executor.submit(handle, item): item
                                               for item in todo}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record = future.result()
                except Exception as e:  # noqa: BLE001 - one bad doc must not stop the run
                    log.error("Failed to catalogue %s: %s", item.label, e)
                    with stats_lock:
                        stats.failed += 1
                    progress.tick(item.label, "crashed", "-")
                    continue
                if record is None:      # cancelled by Ctrl-C
                    continue
                store.add(record)
                with stats_lock:
                    if record.get("status") == "ok":
                        stats.indexed += 1
                    else:
                        stats.failed += 1
                    name = record.get("profile", "generic")
                    stats.by_profile[name] = stats.by_profile.get(name, 0) + 1
                progress.tick(item.label, record.get("status", "?"),
                              record.get("profile", "-"))
    except KeyboardInterrupt:  # pragma: no cover - interactive path
        stop.set()
        stats.interrupted = True
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        store.checkpoint()

    if stop.is_set():
        stats.interrupted = True
    stats.seconds = time.monotonic() - started
    return stats


def _mp_context():
    """A start method that is safe to use from a multi-threaded parent.

    `fork` in a process that already has worker threads is deprecated and can
    deadlock; `forkserver` forks from a clean, single-threaded helper instead.
    """
    for method in ("forkserver", "spawn"):
        if method in multiprocessing.get_all_start_methods():
            return multiprocessing.get_context(method)
    return multiprocessing.get_context()  # pragma: no cover - platform default


def _report_plan(todo: list[WorkItem], profiles: ProfileSet, opts: PipelineOptions) -> None:
    """--dry-run: show which profile each document would get, without calling the LLM."""
    from collections import Counter

    forced = profiles.get(opts.profile_name) if opts.profile_name else None
    counts: Counter[str] = Counter()
    for item in todo:
        if forced is not None:
            name = forced.name
        else:
            name = profiles.select(MatchContext(
                path=item.base_record.get("path") or item.key, filetype=item.filetype,
                size_bytes=item.size_bytes, source_kind=item.source_kind,
                metadata=item.metadata)).name
        counts[name] += 1
        if len(todo) <= 200:
            log.info("  %-12s %s", name, item.key)
    log.info("Dry run — %d document(s) would be catalogued:", len(todo))
    for name, n in counts.most_common():
        log.info("    %-16s %d", name, n)
    if profiles.needs_text_for_routing() and forced is None:
        log.info("Note: some profiles route on document content; a dry run cannot "
                 "evaluate those rules, so the real run may pick differently.")


def _install_sigint(stop: threading.Event) -> None:
    """First Ctrl-C drains in-flight work and saves; a second one aborts hard."""
    try:
        previous = signal.getsignal(signal.SIGINT)
    except ValueError:  # pragma: no cover - not the main thread
        return

    def handler(signum, frame):  # pragma: no cover - interactive path
        if stop.is_set():
            signal.signal(signal.SIGINT, previous)
            raise KeyboardInterrupt
        stop.set()
        log.warning("Interrupted — finishing documents already in flight and saving. "
                    "Press Ctrl-C again to abort immediately.")

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:  # pragma: no cover - not the main thread
        pass


def write_followup_lists(catalog_path: Path, records: list[dict]) -> None:
    """Write needs_ocr.txt / needs_conversion.txt next to the catalog."""
    write_path_list(catalog_path, records, "needs_ocr.txt",
                    lambda r: r.get("needs_ocr") or r.get("status") == "needs_ocr",
                    "need OCR")
    write_path_list(catalog_path, records, "needs_conversion.txt",
                    lambda r: r.get("status") == "legacy_format",
                    "are legacy Office files needing conversion")
