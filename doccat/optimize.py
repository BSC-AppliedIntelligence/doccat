"""An agent that designs the cataloguing profile for a corpus before you index it.

Picking the right profile by hand means knowing what is in the repository — and
the whole reason to catalogue a repository is that nobody does. `doccat
optimize` closes that loop: it reads a stratified sample of the documents, works
out what kind of corpus this is, and writes a cataloguing profile tuned to it.

    doccat optimize --root /data/documents -o profiles/mydomain.yaml

The agent runs in four phases:

    1. **Sample**   — a stratified draw across folders, file types and sizes, so
                      the sample looks like the corpus rather than like its
                      alphabetically-first folder.
    2. **Analyse**  — one LLM call over the sample: what domain is this, what
                      document families are present, what would a researcher
                      need recorded, which fields would actually discriminate
                      between these documents, how much of each should be read.
    3. **Draft**    — the analysis is turned into a real profile YAML, validated
                      by loading it through the normal profile machinery. An
                      invalid draft is a hard error, not a file you discover is
                      broken 10,000 documents later.
    4. **Refine**   — the draft is used to catalogue a few sample documents for
                      real; the agent sees its own output, spots fields that came
                      back empty or useless, and revises the profile. Repeat
                      `--refine N` times.

The result is a file you own and can edit — not a hidden prompt. Nothing is
applied automatically: review it, then pass `--profiles-dir` to your index run.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .jsonparse import parse_metadata
from .llm import LLMClient, LLMError
from .profiles import (BASE_PROFILE, CORE_FIELDS, RELATIONS_FIELD, Profile,
                       ProfileError, load_profiles)
from .sources import WorkItem

log = logging.getLogger("doccat.optimize")

__all__ = ["SampledDocument", "OptimizeResult", "sample_documents",
           "design_profile", "render_profile_yaml"]

#: Excerpt strategies the agent is allowed to choose.
_STRATEGIES = ("head_tail", "head", "full", "sampled", "map_reduce")

#: Field types the agent is allowed to declare, and the synonyms models reach
#: for. Anything else becomes a string — a wrong type is recoverable, an
#: unloadable profile is not.
_FIELD_TYPES = {
    "string": "string", "str": "string", "text": "string",
    "list": "list", "array": "list", "list[str]": "list", "string[]": "list",
}


@dataclass
class SampledDocument:
    """One document drawn from the corpus, with enough text to reason about."""

    key: str                  # relative path, or "dump.json#record"
    filetype: str
    size_bytes: int
    text: str
    truncated: bool = False
    item: WorkItem | None = None   # kept so the refine phase can re-catalogue it

    def render(self, index: int) -> str:
        tail = "\n[...truncated...]" if self.truncated else ""
        return (f"--- document {index} ---\n"
                f"path: {self.key}\n"
                f"type: {self.filetype}   size: {self.size_bytes} bytes\n"
                f"content:\n{self.text}{tail}\n")


@dataclass
class OptimizeResult:
    """What the agent produced, plus how it got there."""

    profile_name: str
    yaml_text: str
    analysis: dict = field(default_factory=dict)
    sample: list[SampledDocument] = field(default_factory=list)
    rounds: list[dict] = field(default_factory=list)   # one entry per refine pass
    llm_calls: int = 0

    @property
    def profile(self) -> Profile:
        """The drafted profile, loaded through the normal machinery."""
        return _load_draft(self.yaml_text, self.profile_name)


# ---------------------------------------------------------------------------
# Phase 1 — sampling
# ---------------------------------------------------------------------------

def sample_documents(source, size: int = 12, *, sample_chars: int = 6000,
                     seed: int = 0, max_scan: int | None = None) -> list[SampledDocument]:
    """Draw a stratified sample of `size` documents and extract their text.

    Stratification is by (top-level folder, file type): a corpus of 9,000
    invoices and 40 contracts should still put a contract in front of the agent,
    because the contracts are exactly the documents a generic profile handles
    worst. Within each stratum the draw is random but seeded, so a rerun on an
    unchanged corpus reads the same documents.
    """
    items = source.scan(hash_workers=8, change_detection="mtime")
    if not items:
        return []
    if max_scan is not None and len(items) > max_scan:
        items = items[:max_scan]

    strata: dict[tuple[str, str], list[WorkItem]] = defaultdict(list)
    for item in items:
        key = item.base_record.get("path") or item.key
        folder = key.split("/")[0] if "/" in key else "."
        strata[(folder, item.filetype)].append(item)

    rng = random.Random(seed)
    for bucket in strata.values():
        rng.shuffle(bucket)

    # Round-robin across strata so no single folder dominates the sample.
    picked: list[WorkItem] = []
    buckets = [list(b) for _, b in sorted(strata.items())]
    while buckets and len(picked) < size:
        for bucket in list(buckets):
            if not bucket:
                buckets.remove(bucket)
                continue
            picked.append(bucket.pop())
            if len(picked) >= size:
                break

    sampled: list[SampledDocument] = []
    for item in picked:
        extraction = source.load_text(item)
        text = (extraction.text or "").strip()
        if not text:
            log.info("Sample: %s has no extractable text — skipping it", item.key)
            continue
        sampled.append(SampledDocument(
            key=item.base_record.get("path") or item.key,
            filetype=item.filetype,
            size_bytes=item.size_bytes,
            text=text[:sample_chars],
            truncated=len(text) > sample_chars,
            item=item,
        ))
    log.info("Sampled %d document(s) from %d, across %d stratum/strata",
             len(sampled), len(items), len(strata))
    return sampled


# ---------------------------------------------------------------------------
# Phase 2 — analysis
# ---------------------------------------------------------------------------

_ANALYSIS_PROMPT = """You are designing the cataloguing strategy for a document \
repository.

An AI agent will later answer questions about this repository WITHOUT a vector \
store. It sees only a metadata record per document — a summary, a description, \
keywords, and whatever structured fields we decide to extract — and from those \
records alone it must decide which documents to open. Your job is to decide what \
those records should contain so that decision is easy.

Below is a sample of {n} documents from the repository. Study what they actually \
are, what varies between them, and what a person searching this repository would \
be looking for.

Then return ONLY a JSON object (no markdown fences, no commentary) with these fields:

{{
  "corpus_summary": "2-4 sentences: what this repository is, who produced it, what it is for",
  "domain": "short domain label, e.g. 'public procurement', 'clinical research', 'municipal government'",
  "document_families": ["the distinct kinds of document present in the sample, named concretely"],
  "dominant_language": "ISO 639-1 code of the language most documents are written in",
  "profile_name": "lowercase_snake_case name for the cataloguing profile, derived from the domain",
  "description": "one line describing the profile",
  "extends": "which built-in profile to build on: one of {builtins}",
  "role": "the persona sentence the cataloguer should be given: 'You are ...'. Be specific to this domain.",
  "guidance": "2-5 sentences of domain rules the cataloguer must follow: what to quote verbatim, what must never be inferred, which vocabulary to use, what people search for in this repository",
  "doc_types": ["controlled vocabulary of document types for THIS repository, 5-12 entries, snake_case, ending with 'other'"],
  "fields": [
    {{"name": "snake_case_field", "type": "string or list",
      "description": "exactly what the model should put here, and what to do when it is absent"}}
  ],
  "strategy": "one of: {strategies}",
  "strategy_reason": "one sentence on why that strategy suits these documents",
  "head_chars": 8000,
  "tail_chars": 2000,
  "match_path_globs": ["glob patterns matching where these documents live, or empty"],
  "match_filename_regex": "regex matching their file names, or empty string",
  "entity_types": ["kinds of entity worth extracting as a graph node for THIS repository, e.g. 'organization', 'person', 'role' — or empty, see below"],
  "relation_types": ["short snake_case relation labels worth looking for between those entities, e.g. 'supplies_to', 'party_to' — or empty, see below"],
  "reasoning": "2-4 sentences explaining the field choices — why these fields discriminate between documents in THIS repository"
}}

Rules for "fields": propose 4-10 fields BEYOND the core ones, which already \
exist and must not be repeated: {core}. Each proposed field must be something \
that (a) actually appears in these documents, and (b) a searcher would use to \
tell two documents of this corpus apart. A field that would be empty or \
identical for most of these documents is worthless — leave it out.

Rules for "strategy": "head_tail" reads the opening and the closing; "head" only \
the opening; "full" the whole (short) document; "sampled" spreads windows across \
a long uniform text; "map_reduce" takes notes chunk by chunk then catalogues from \
the notes — use it only when documents are long AND their middle matters.

Rules for "entity_types"/"relation_types": these documents may be worth representing \
as a graph — who/what relates to whom/what — in addition to the metadata fields above. \
Propose a vocabulary ONLY if these documents actually state concrete relationships \
between named entities that a reader would want to trace (who supplies whom, who \
reports to whom, who proposed what, which organisation owns which). Leave BOTH lists \
empty when they would not: documents that are self-contained (a manual, an FAQ, a \
single-author report with no network of named parties) gain nothing from a graph, and \
a forced vocabulary produces relations nobody asked for. When in doubt, leave them empty.

{documents}
"""

_REFINE_PROMPT = """You designed the cataloguing profile below, and it was then \
used to catalogue {n} real documents from the repository. The resulting metadata \
records follow.

Judge your own output. Look for:
  - fields that came back empty, or nearly empty, across documents — they are
    either badly named, badly described, or simply not present in this corpus;
  - fields whose values are vague, generic or identical between documents — they
    do not help an agent choose;
  - facts visible in the records that no field captured, and should have one;
  - doc_type values the model had to force into "other", meaning the vocabulary
    is wrong;
  - guidance the model evidently ignored, which usually means it was too vague.

Then return ONLY a JSON object in EXACTLY the same schema as before, containing \
the IMPROVED profile. Keep what worked; change what did not. If the profile is \
already good, return it unchanged and say so in "reasoning".

--- the profile you designed ---
{profile_yaml}

--- what it produced on real documents ---
{records}
"""


def _analysis_prompt(sample: list[SampledDocument], builtins: Iterable[str]) -> str:
    documents = "\n".join(doc.render(i) for i, doc in enumerate(sample, 1))
    return _ANALYSIS_PROMPT.format(
        n=len(sample),
        builtins=", ".join(sorted(builtins)),
        strategies=", ".join(_STRATEGIES),
        core=", ".join(CORE_FIELDS),
        documents=documents,
    )


# ---------------------------------------------------------------------------
# Phase 3 — drafting a valid profile
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"[^a-z0-9_]+")


def _safe_name(raw: Any, fallback: str = "custom") -> str:
    name = _NAME_RE.sub("_", str(raw or "").strip().lower()).strip("_")
    return name or fallback


def _clean_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def render_profile_yaml(analysis: dict, *, name: str | None = None,
                        available: Iterable[str] = ()) -> tuple[str, str]:
    """Turn the agent's analysis into profile YAML. Returns (yaml_text, name).

    Everything the model proposed is sanitised here rather than trusted: names
    are forced to snake_case, `extends` must be a profile that exists, the
    strategy must be one this code implements, and core fields are dropped if
    the model repeated them. A model that returns nonsense produces a poorer
    profile, never an unloadable one.
    """
    available = set(available) or {BASE_PROFILE}
    profile_name = _safe_name(name or analysis.get("profile_name"), "custom")

    parent = str(analysis.get("extends") or BASE_PROFILE).strip()
    if parent not in available or parent == profile_name:
        parent = BASE_PROFILE

    strategy = str(analysis.get("strategy") or "head_tail").strip()
    if strategy not in _STRATEGIES:
        log.warning("Agent proposed unknown strategy %r — falling back to head_tail", strategy)
        strategy = "head_tail"

    fields_out: dict[str, Any] = {}
    for spec in analysis.get("fields") or []:
        if not isinstance(spec, dict):
            continue
        field_name = _safe_name(spec.get("name"), "")
        description = str(spec.get("description") or "").strip()
        if not field_name or not description:
            continue
        if field_name in CORE_FIELDS or field_name == RELATIONS_FIELD:
            # `entities`/`doc_type`/etc. are inherited, not redefined here; `relations`
            # is reserved for the structured shape the `graph:` block below injects —
            # letting the agent redefine it as a plain string list would silently
            # shadow that richer schema.
            continue
        field_type = _FIELD_TYPES.get(str(spec.get("type") or "string").strip().lower(),
                                      "string")
        fields_out[field_name] = ({"type": "list", "description": description}
                                  if field_type == "list" else description)

    doc_types = [_safe_name(t) for t in _clean_list(analysis.get("doc_types"))]
    doc_types = [t for t in dict.fromkeys(doc_types) if t]
    if doc_types and "other" not in doc_types:
        doc_types.append("other")

    prompt: dict[str, Any] = {}
    if analysis.get("role"):
        prompt["role"] = str(analysis["role"]).strip()
    if analysis.get("guidance"):
        prompt["guidance"] = str(analysis["guidance"]).strip()
    if doc_types:
        prompt["doc_types"] = doc_types
    if fields_out:
        prompt["fields"] = fields_out

    excerpt: dict[str, Any] = {"strategy": strategy}
    for key, default in (("head_chars", 8000), ("tail_chars", 2000)):
        try:
            value = int(analysis.get(key, default))
        except (TypeError, ValueError):
            value = default
        excerpt[key] = max(500, min(value, 200_000))
    if strategy == "full":
        excerpt.pop("tail_chars", None)
        excerpt["max_chars"] = max(excerpt.pop("head_chars"), 8000)

    spec: dict[str, Any] = {
        "name": profile_name,
        "description": str(analysis.get("description")
                           or analysis.get("corpus_summary") or "").strip()[:300],
        "extends": parent,
        "priority": 50,
        "excerpt": excerpt,
        "prompt": prompt,
    }

    globs = _clean_list(analysis.get("match_path_globs"))
    filename_regex = str(analysis.get("match_filename_regex") or "").strip()
    alternatives = []
    if globs:
        alternatives.append({"path_globs": globs})
    if filename_regex and _valid_regex(filename_regex):
        alternatives.append({"filename_regex": filename_regex})
    if alternatives:
        spec["match"] = {"any_of": alternatives} if len(alternatives) > 1 else alternatives[0]

    entity_types = [_safe_name(t) for t in _clean_list(analysis.get("entity_types"))]
    entity_types = [t for t in dict.fromkeys(entity_types) if t]
    relation_types = [_safe_name(t) for t in _clean_list(analysis.get("relation_types"))]
    relation_types = [t for t in dict.fromkeys(relation_types) if t]
    # Only turn relation extraction on when the agent actually proposed a
    # vocabulary for it — an empty `graph: {}` would still enable it (see
    # Profile.graph_enabled), which is the opposite of "leave both empty when
    # a graph wouldn't help" in the prompt's own instructions.
    if entity_types or relation_types:
        spec["graph"] = {}
        if entity_types:
            spec["graph"]["entity_types"] = entity_types
        if relation_types:
            spec["graph"]["relation_types"] = relation_types

    header = _header(analysis, profile_name)
    body = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=88)
    return header + body, profile_name


def _valid_regex(pattern: str) -> bool:
    try:
        re.compile(pattern)
        return True
    except re.error:
        log.warning("Agent proposed an invalid filename_regex %r — dropping it", pattern)
        return False


def _header(analysis: dict, profile_name: str) -> str:
    lines = [f"# Cataloguing profile drafted by `doccat optimize` for: {profile_name}",
             "#",
             "# Review it before indexing a whole repository with it — it is a starting",
             "# point produced from a sample, not a finished specification.",
             "#"]
    for label, key in (("Corpus", "corpus_summary"), ("Domain", "domain"),
                       ("Families", "document_families"), ("Why these fields", "reasoning"),
                       ("Why this strategy", "strategy_reason"),
                       ("Graph entity types", "entity_types"),
                       ("Graph relation types", "relation_types")):
        value = analysis.get(key)
        if not value:
            continue
        text = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        wrapped = _wrap(text, 74)
        lines.append(f"# {label}: {wrapped[0]}")
        lines.extend(f"#   {line}" for line in wrapped[1:])
    lines.append("")
    return "\n".join(lines) + "\n"


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width and current:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def _load_draft(yaml_text: str, profile_name: str) -> Profile:
    """Load a drafted profile through the real machinery, so invalid drafts fail here."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="doccat_draft_") as tmp:
        path = Path(tmp) / f"{profile_name}.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        return load_profiles(extra_dirs=[tmp]).get(profile_name)


# ---------------------------------------------------------------------------
# Phase 4 — trial and refinement
# ---------------------------------------------------------------------------

def _trial(profile: Profile, sample: list[SampledDocument], client: LLMClient,
           limit: int = 3) -> list[dict]:
    """Catalogue a few sampled documents with the draft profile, for real."""
    from extractors import MIME_TYPES

    records: list[dict] = []
    for doc in sample[:limit]:
        prompt = profile.build_prompt(Path(doc.key).name, profile.excerpt(doc.text))
        try:
            raw = client.complete(prompt, mime_types=MIME_TYPES, **profile.llm_kwargs())
            metadata = profile.normalize(parse_metadata(raw))
        except (LLMError, ValueError, TypeError) as e:
            log.warning("Trial cataloguing failed for %s: %s", doc.key, e)
            records.append({"path": doc.key, "error": str(e)[:200]})
            continue
        metadata["path"] = doc.key
        records.append(metadata)
    return records


def _unmatched_relation_count(record: dict) -> int:
    """How many of a record's relations name a subject/object absent from its
    own `entities` field — a cheap, model-free signal that the cataloguer is
    inventing connections rather than reporting ones visible in the text.
    A record with no extracted entity names at all is skipped rather than
    penalised: there is nothing to check a relation against.
    """
    relations = record.get(RELATIONS_FIELD)
    if not isinstance(relations, list) or not relations:
        return 0
    entities = record.get("entities") or {}
    names: set[str] = set()
    if isinstance(entities, dict):
        for values in entities.values():
            if isinstance(values, list):
                names.update(str(v).strip().lower() for v in values if isinstance(v, str))
    if not names:
        return 0
    unmatched = 0
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        for role in ("subject", "object"):
            value = str(rel.get(role, "")).strip().lower()
            if value and not any(value in n or n in value for n in names):
                unmatched += 1
                break
    return unmatched


def _critique(records: list[dict]) -> dict:
    """Local, model-free diagnosis of a trial run — reported and fed to the agent."""
    if not records:
        return {}
    usable = [r for r in records if "error" not in r]
    empty: dict[str, int] = defaultdict(int)
    identical: dict[str, set] = defaultdict(set)
    for record in usable:
        for key, value in record.items():
            if key == "path":
                continue
            if value in (None, "", [], {}):
                empty[key] += 1
            else:
                identical[key].add(json.dumps(value, ensure_ascii=False, sort_keys=True)[:200])
    n = len(usable) or 1
    critique = {
        "documents_catalogued": len(usable),
        "failures": len(records) - len(usable),
        "always_empty": sorted(k for k, c in empty.items() if c == n),
        "often_empty": sorted(k for k, c in empty.items() if 0 < c < n and c / n >= 0.5),
        "no_variation": sorted(k for k, values in identical.items()
                               if len(values) == 1 and n > 1 and k not in empty),
    }
    unmatched_relations = sum(_unmatched_relation_count(r) for r in usable)
    if unmatched_relations:
        critique["relations_without_matching_entities"] = unmatched_relations
    return critique


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

def design_profile(source, client: LLMClient, *, sample_size: int = 12,
                   sample_chars: int = 6000, refine_rounds: int = 1,
                   name: str | None = None, seed: int = 0,
                   trial_docs: int = 3,
                   extra_profile_dirs: Iterable[str | Path] = ()) -> OptimizeResult:
    """Sample a corpus, analyse it, and draft (then refine) a cataloguing profile."""
    available = load_profiles(extra_dirs=extra_profile_dirs).names()

    log.info("Phase 1/4 — sampling up to %d document(s)", sample_size)
    sample = sample_documents(source, sample_size, sample_chars=sample_chars, seed=seed)
    if not sample:
        raise ProfileError("no documents with extractable text were found to sample — "
                           "check --root, or index with --mode upload for scanned files")

    log.info("Phase 2/4 — analysing the sample (%d document(s), %d chars)",
             len(sample), sum(len(d.text) for d in sample))
    calls = 0
    raw = client.complete(_analysis_prompt(sample, available))
    calls += 1
    analysis = parse_metadata(raw)

    log.info("Phase 3/4 — drafting the profile")
    yaml_text, profile_name = render_profile_yaml(analysis, name=name, available=available)
    profile = _load_draft(yaml_text, profile_name)
    log.info("Drafted profile %r: %d extra field(s), strategy %s, extends %s",
             profile_name, len([f for f in profile.fields if f not in CORE_FIELDS]),
             profile.strategy, analysis.get("extends", BASE_PROFILE))

    rounds: list[dict] = []
    for round_no in range(1, max(refine_rounds, 0) + 1):
        log.info("Phase 4/4 — refinement round %d/%d: cataloguing %d sample document(s)",
                 round_no, refine_rounds, min(trial_docs, len(sample)))
        records = _trial(profile, sample, client, limit=trial_docs)
        calls += len(records)
        critique = _critique(records)
        if critique:
            log.info("Trial: %s", json.dumps(critique, ensure_ascii=False))

        prompt = _REFINE_PROMPT.format(
            n=len(records),
            profile_yaml=yaml_text,
            records=json.dumps(records, ensure_ascii=False, indent=2)[:40_000],
        )
        if critique:
            prompt += ("\n--- mechanical check of those records ---\n"
                       + json.dumps(critique, ensure_ascii=False, indent=2)
                       + "\nFields listed under always_empty or no_variation earn their "
                         "place only if you can explain why; otherwise replace them. A "
                         "non-zero relations_without_matching_entities means some relation's "
                         "subject or object was not among the entities extracted from that "
                         "same document — tighten the graph guidance so it only reports "
                         "relations between entities actually named in the text, or drop "
                         "the graph vocabulary entirely if this keeps happening.")
        try:
            revised = parse_metadata(client.complete(prompt))
            calls += 1
        except (LLMError, ValueError, TypeError) as e:
            log.warning("Refinement round %d failed (%s) — keeping the current draft",
                        round_no, e)
            rounds.append({"round": round_no, "trial": critique, "error": str(e)[:200]})
            break

        try:
            new_yaml, profile_name = render_profile_yaml(revised, name=name or profile_name,
                                                         available=available)
            profile = _load_draft(new_yaml, profile_name)
        except (ProfileError, yaml.YAMLError) as e:
            log.warning("Refinement round %d produced an invalid profile (%s) — "
                        "keeping the previous draft", round_no, e)
            rounds.append({"round": round_no, "trial": critique, "error": str(e)[:200]})
            break

        yaml_text, analysis = new_yaml, revised
        rounds.append({"round": round_no, "trial": critique,
                       "changes": str(revised.get("reasoning", ""))[:500]})

    return OptimizeResult(profile_name=profile_name, yaml_text=yaml_text, analysis=analysis,
                          sample=sample, rounds=rounds, llm_calls=calls)
