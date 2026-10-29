"""Cataloguing profiles: domain- and document-kind-specific prompting.

A *profile* is a YAML file describing how one family of documents should be
catalogued: the cataloguer persona, extra instructions, the controlled
vocabulary of document types, the **metadata schema** (base fields plus any
domain-specific fields), how much of the document to show the model, and which
sampling strategy to use for long texts.

Profiles compose. Every profile `extends` another (`generic` by default) and
deep-merges onto it, so a domain profile only states its differences.

Profiles route themselves. Each one may declare `match` rules (path globs,
filename/content regexes, file types, source-record metadata, size bounds);
the indexer picks, for every document, the highest-priority profile whose
rules all match, falling back to `generic`. A profile with no `match` block is
only ever used when named explicitly with `--profile`.

    from doccat.profiles import load_profiles, MatchContext

    profiles = load_profiles(["./my_profiles"])
    profile = profiles.select(MatchContext(path="plenos/DSC-2024-1.pdf",
                                           filetype="pdf", size_bytes=120_000))
    prompt = profile.build_prompt(label="DSC-2024-1.pdf", excerpt=text)
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

log = logging.getLogger("doccat.profiles")

BUILTIN_DIR = Path(__file__).parent / "profiles"
BASE_PROFILE = "generic"

#: Fields every catalog record is expected to carry, whatever the profile.
CORE_FIELDS = ("title", "summary", "description", "doc_type", "language",
               "keywords", "topics", "entities", "questions_answered")

#: The name of the field a `graph:`-enabled profile gets for free (see GraphConfig).
RELATIONS_FIELD = "relations"


class ProfileError(Exception):
    """A profile file is missing, malformed, or refers to an unknown parent."""


# ---------------------------------------------------------------------------
# Schema description
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    """One field of the metadata schema a profile asks the model to produce."""

    name: str
    description: str = ""
    type: str = "string"            # string | list | object | integer | number | boolean
    fields: dict[str, "FieldSpec"] = field(default_factory=dict)  # for type: object
    items: str = "string"           # for type: list — "string" (default) or "object"

    @classmethod
    def parse(cls, name: str, spec: Any) -> "FieldSpec":
        if isinstance(spec, str):
            return cls(name=name, description=spec, type="string")
        if not isinstance(spec, dict):
            raise ProfileError(f"field {name!r}: expected a string or a mapping, "
                               f"got {type(spec).__name__}")
        ftype = str(spec.get("type", "object" if "fields" in spec else "string"))
        items = str(spec.get("items", "string"))
        if items not in ("string", "object"):
            raise ProfileError(f"field {name!r}: items must be 'string' or 'object', "
                               f"got {items!r}")
        # The same `fields:` key names an object's own sub-fields (type: object)
        # or, for a list of structured elements (type: list, items: object), the
        # schema of each element — a relation {subject, relation, object, ...}
        # is exactly this second case.
        sub = {k: cls.parse(k, v) for k, v in (spec.get("fields") or {}).items()}
        if items == "object" and ftype != "list":
            raise ProfileError(f"field {name!r}: items: object only applies to type: list")
        if ftype == "list" and items == "object" and not sub:
            raise ProfileError(f"field {name!r}: type: list with items: object needs "
                               f"a non-empty 'fields' mapping describing each element")
        return cls(name=name, description=str(spec.get("description", "")),
                   type=ftype, fields=sub, items=items)

    def placeholder(self) -> Any:
        """The value shown in the JSON skeleton handed to the model."""
        if self.type == "object":
            return {k: v.placeholder() for k, v in self.fields.items()}
        if self.type == "list" and self.items == "object":
            return [{k: v.placeholder() for k, v in self.fields.items()}]
        if self.type == "list":
            return [self.description or f"{self.name} values"]
        if self.type in ("integer", "number"):
            return self.description or 0
        if self.type == "boolean":
            return self.description or True
        return self.description or f"{self.name}"

    def coerce(self, value: Any) -> Any:
        """Best-effort normalisation of a model-produced value to this field's type."""
        if value is None:
            return None
        if self.type == "list" and self.items == "object":
            if not isinstance(value, list):
                return []
            out = []
            for element in value:
                if isinstance(element, dict):
                    out.append({k: (self.fields[k].coerce(v) if k in self.fields else v)
                               for k, v in element.items()})
                # A non-dict element (the model returned a bare string instead of
                # the structured shape) is dropped rather than forced — the same
                # best-effort posture every other coercion here already takes.
            return out
        if self.type == "list":
            if isinstance(value, list):
                return [v for v in value if v not in (None, "")]
            if isinstance(value, str):
                parts = [p.strip() for p in re.split(r"[;\n]|,(?![^(]*\))", value)]
                return [p for p in parts if p]
            return [value]
        if self.type == "object":
            if not isinstance(value, dict):
                return {}
            out = {}
            for key, val in value.items():
                sub = self.fields.get(key)
                out[key] = sub.coerce(val) if sub else val
            return out
        if self.type in ("integer", "number"):
            try:
                return int(value) if self.type == "integer" else float(value)
            except (TypeError, ValueError):
                return None
        if self.type == "string" and not isinstance(value, str):
            return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) \
                else str(value)
        return value


# ---------------------------------------------------------------------------
# Graph extraction (relations between the entities a profile already extracts)
# ---------------------------------------------------------------------------

#: The schema every graph-enabled profile's `relations` field gets, unless the
#: profile author writes their own `relations` field explicitly. Deliberately
#: not a fixed predicate vocabulary (see docs/GRAPH.md Decision 1): `relation`
#: is an open, model-chosen label and `description` is the fact that actually
#: matters, the same way `doc_type` is a label but `summary` is the content.
_RELATION_FIELD_SPEC: dict[str, Any] = {
    "type": "list",
    "items": "object",
    "fields": {
        "subject": "the first entity, exactly as named in the document",
        "relation": "a short snake_case label for the relationship, e.g. "
                    "'supplies_to', 'reports_to', 'party_to'",
        "object": "the second entity, exactly as named in the document",
        "description": "one sentence stating the relationship as the document "
                       "states it — this is what an agent actually reads",
        "strength": {"type": "number",
                    "description": "0-1: how central this relation is to the document"},
    },
}


@dataclass
class GraphConfig:
    """A profile's relation-extraction settings (the `graph:` YAML block).

    Presence of this (even empty) on a `Profile` is what turns relation
    extraction on; its absence is the default and changes nothing about how a
    profile behaves — see docs/GRAPH.md Decision 5.
    """

    entity_types: list[str] = field(default_factory=list)
    relation_types: list[str] = field(default_factory=list)
    dedicated_pass: bool = False      # a second LLM call just for relations
    min_confidence: float = 0.0       # drop relations below this `strength`

    def entity_guidance(self) -> str:
        """Vocabulary hint for the `entities` field — a hint, never an enforced enum."""
        if not self.entity_types:
            return ""
        return ("Entities in this kind of document are typically: "
               + ", ".join(self.entity_types) + ".")

    def relation_guidance(self) -> str:
        """Vocabulary hint for the `relations` field — a hint, never an enforced enum."""
        if not self.relation_types:
            return ("Only record a relation that the document actually states; "
                    "never infer one from context alone.")
        return ("Prefer these relation labels where they genuinely fit: "
               + ", ".join(self.relation_types) + ". Use another short snake_case "
               "label when none of these fit — this list is a hint, not a "
               "constraint. Only record a relation that the document actually "
               "states; never infer one from context alone.")

    def guidance(self) -> str:
        """Both hints combined — used when `relations` is in the same call as `entities`."""
        return " ".join(p for p in (self.entity_guidance(), self.relation_guidance()) if p)


_GRAPH_KEYS = {"entity_types", "relation_types", "dedicated_pass", "min_confidence"}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

@dataclass
class MatchContext:
    """What is known about a document when a profile has to be chosen."""

    path: str = ""                       # path relative to the root (or record label)
    filename: str = ""
    filetype: str = ""                   # pdf, docx, json_record, ...
    size_bytes: int = 0
    source_kind: str = "files"           # files | json
    text_head: str = ""                  # first chars of extracted text, if available
    metadata: dict[str, Any] = field(default_factory=dict)  # JSON record's other fields

    def __post_init__(self) -> None:
        if not self.filename and self.path:
            self.filename = Path(self.path).name


@dataclass
class MatchRules:
    """Conditions under which a profile claims a document.

    Every rule stated at this level must hold (AND). `any_of` holds a list of
    nested rule sets of which at least one must hold (OR) — that is how a
    profile says "the folder *or* the file name tells me this is mine".
    """

    path_globs: list[str] = field(default_factory=list)
    filename_regex: str | None = None
    content_regex: str | None = None
    filetypes: list[str] = field(default_factory=list)
    source_kinds: list[str] = field(default_factory=list)
    metadata_equals: dict[str, Any] = field(default_factory=dict)
    metadata_regex: dict[str, str] = field(default_factory=dict)
    min_bytes: int | None = None
    max_bytes: int | None = None
    any_of: list["MatchRules"] = field(default_factory=list)

    @classmethod
    def parse(cls, spec: dict | None) -> "MatchRules":
        spec = spec or {}
        unknown = set(spec) - {f.name for f in cls.__dataclass_fields__.values()}
        if unknown:
            raise ProfileError(f"unknown match rule(s): {sorted(unknown)}")
        alternatives = spec.get("any_of") or []
        if not isinstance(alternatives, list):
            raise ProfileError("match.any_of must be a list of rule sets")
        return cls(
            any_of=[cls.parse(alt) for alt in alternatives],
            path_globs=list(spec.get("path_globs") or []),
            filename_regex=spec.get("filename_regex"),
            content_regex=spec.get("content_regex"),
            filetypes=[str(t).lower().lstrip(".") for t in (spec.get("filetypes") or [])],
            source_kinds=list(spec.get("source_kinds") or []),
            metadata_equals=dict(spec.get("metadata_equals") or {}),
            metadata_regex=dict(spec.get("metadata_regex") or {}),
            min_bytes=spec.get("min_bytes"),
            max_bytes=spec.get("max_bytes"),
        )

    @property
    def empty(self) -> bool:
        return not any([self.path_globs, self.filename_regex, self.content_regex,
                        self.filetypes, self.source_kinds, self.metadata_equals,
                        self.metadata_regex, self.min_bytes, self.max_bytes,
                        self.any_of])

    def matches(self, ctx: MatchContext) -> bool:
        if self.empty:
            return False
        path = ctx.path.replace("\\", "/")
        # Also try the path with a leading slash so a `**/contracts/**` pattern
        # matches a top-level `contracts/...` folder, which is what people mean.
        candidates = (path, "/" + path.lstrip("/"), ctx.filename)
        if self.path_globs and not any(
                fnmatch.fnmatch(c, g) for g in self.path_globs for c in candidates):
            return False
        if self.filename_regex and not re.search(self.filename_regex, ctx.filename, re.I):
            return False
        if self.filetypes and ctx.filetype.lower().lstrip(".") not in self.filetypes:
            return False
        if self.source_kinds and ctx.source_kind not in self.source_kinds:
            return False
        if self.min_bytes is not None and ctx.size_bytes < self.min_bytes:
            return False
        if self.max_bytes is not None and ctx.size_bytes > self.max_bytes:
            return False
        for key, expected in self.metadata_equals.items():
            if str(ctx.metadata.get(key, "")).strip().lower() != str(expected).strip().lower():
                return False
        for key, pattern in self.metadata_regex.items():
            if not re.search(pattern, str(ctx.metadata.get(key, "")), re.I):
                return False
        if self.content_regex:
            if not ctx.text_head or not re.search(self.content_regex, ctx.text_head, re.I):
                return False
        if self.any_of and not any(alt.matches(ctx) for alt in self.any_of):
            return False
        return True

    def needs_text(self) -> bool:
        """True when this rule can only be evaluated after text extraction."""
        return self.content_regex is not None or any(a.needs_text() for a in self.any_of)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """{role}

{instructions}

Return ONLY a JSON object with exactly these fields — no markdown fences, no \
commentary, no explanation before or after:

{schema}
{guidance}
Document: {label}
"""


@dataclass
class Profile:
    """A resolved (inheritance already applied) cataloguing profile."""

    name: str
    description: str = ""
    priority: int = 0
    match: MatchRules = field(default_factory=MatchRules)
    # prompt
    role: str = "You are a document cataloguer."
    instructions: str = ""
    guidance: str = ""
    doc_types: list[str] = field(default_factory=list)
    fields: dict[str, FieldSpec] = field(default_factory=dict)
    output_language: str | None = None
    map_instructions: str = ""
    reduce_instructions: str = ""
    # excerpting
    strategy: str = "head_tail"        # head_tail | head | full | sampled | map_reduce
    head_chars: int = 8000
    tail_chars: int = 2000
    max_chars: int = 60000
    chunk_chars: int = 24000
    chunk_overlap: int = 500
    max_chunks: int = 8
    samples: int = 3
    # llm overrides
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    think: bool | None = None
    ingestion: str | None = None       # parse | upload | auto  (overrides the CLI default)
    source: str = "<builtin>"
    graph: GraphConfig | None = None   # None = no relation extraction (the default)

    @property
    def graph_enabled(self) -> bool:
        return self.graph is not None

    # -- prompt construction ----------------------------------------------
    def schema_json(self, exclude: frozenset[str] = frozenset()) -> str:
        skeleton = {name: spec.placeholder() for name, spec in self.fields.items()
                   if name not in exclude}
        text = json.dumps(skeleton, ensure_ascii=False, indent=2)
        return text.replace("{doc_types}", ", ".join(self.doc_types))

    def _render(self, text: str) -> str:
        return (text or "").replace("{doc_types}", ", ".join(self.doc_types)).strip()

    def build_prompt(self, label: str, excerpt: str | None = None,
                     extra_context: str = "",
                     exclude_fields: frozenset[str] = frozenset()) -> str:
        """The single-shot cataloguing prompt for one document.

        `exclude_fields` drops fields from the requested schema — used to omit
        `relations` from the main call when a profile's `graph.dedicated_pass`
        asks for it in a separate, focused call instead (see
        `build_relations_prompt`).
        """
        guidance_parts = [self._render(self.guidance)]
        if self.graph is not None:
            # The relation vocabulary hint only makes sense when `relations` is
            # actually part of this call's schema.
            graph_guidance = (self.graph.entity_guidance() if RELATIONS_FIELD in exclude_fields
                              else self.graph.guidance())
            if graph_guidance:
                guidance_parts.append(graph_guidance)
        if self.output_language:
            guidance_parts.append(
                f"Write all free-text fields in {self.output_language}, whatever the "
                "language of the document itself.")
        if extra_context:
            guidance_parts.append(extra_context.strip())
        guidance = "\n\n".join(p for p in guidance_parts if p)

        prompt = _PROMPT_TEMPLATE.format(
            role=self._render(self.role),
            instructions=self._render(self.instructions),
            schema=self.schema_json(exclude=exclude_fields),
            guidance=f"\n{guidance}\n" if guidance else "",
            label=label,
        )
        if excerpt:
            prompt += f"\nDocument content:\n---\n{excerpt}\n---"
        return prompt

    def build_relations_prompt(self, label: str, excerpt: str, entities: dict | None) -> str:
        """A dedicated (second) call that extracts only relations.

        Used when `graph.dedicated_pass` is set — grounds the extraction in
        the entities the main call already found (passed in as `entities`,
        the record's `entities` field) rather than re-deriving them from
        scratch, so the two calls agree on who is who.
        """
        spec = self.fields.get(RELATIONS_FIELD)
        if spec is None:  # pragma: no cover - only reachable via direct misuse
            raise ValueError(f"profile {self.name!r} has no {RELATIONS_FIELD!r} field")
        schema = json.dumps({RELATIONS_FIELD: spec.placeholder()}, ensure_ascii=False, indent=2)
        known = (json.dumps(entities, ensure_ascii=False) if entities
                else "(none identified yet — read the document for yourself)")
        parts = [
            self._render(self.role),
            f"You already identified these entities in this document: {known}",
            "Now identify the relationships this document actually states between "
            "them (or involving another entity you notice on this closer read).",
        ]
        if self.graph is not None:
            relation_guidance = self.graph.relation_guidance()
            if relation_guidance:
                parts.append(relation_guidance)
        parts.append("Return ONLY a JSON object with exactly this field — no markdown "
                     "fences, no commentary, no explanation before or after:")
        parts.append(schema)
        parts.append(f"Document: {label}")
        prompt = "\n\n".join(parts)
        if excerpt:
            prompt += f"\nDocument content:\n---\n{excerpt}\n---"
        return prompt

    def build_map_prompt(self, label: str, chunk: str, index: int, total: int) -> str:
        """Prompt for one chunk of a long document (map phase)."""
        instructions = self._render(self.map_instructions) or (
            "Read this part of a longer document and write dense factual notes about it: "
            "what it covers, its sections, the concrete procedures, data, rules, figures, "
            "names and dates it contains. Plain prose or bullets, no JSON, no preamble. "
            "Do not speculate about parts you have not seen.")
        return (f"{self._render(self.role)}\n\n{instructions}\n\n"
                f"Document: {label}\nPart {index} of {total}.\n"
                f"---\n{chunk}\n---")

    def build_reduce_prompt(self, label: str, notes: Iterable[str]) -> str:
        """Prompt that turns per-chunk notes into the final metadata record."""
        joined = "\n\n".join(f"[notes on part {i}]\n{n}" for i, n in enumerate(notes, 1))
        extra = self._render(self.reduce_instructions) or (
            "The material below is a set of notes covering the whole document, part by "
            "part. Catalogue the document as a whole from them; cover the entire document, "
            "not only its beginning.")
        return self.build_prompt(label, excerpt=joined, extra_context=extra)

    # -- excerpting --------------------------------------------------------
    def excerpt(self, text: str) -> str:
        """Reduce a document's text to what the model should see in one call."""
        strategy = self.strategy
        if strategy in ("map_reduce", "full"):
            return text[: self.max_chars]
        if strategy == "head":
            return text[: self.head_chars]
        if strategy == "sampled":
            return self._sampled(text)
        # head_tail (default)
        if len(text) <= self.head_chars + self.tail_chars:
            return text
        return (text[: self.head_chars]
                + "\n[...middle of document omitted...]\n"
                + text[-self.tail_chars:])

    def _sampled(self, text: str) -> str:
        """Head + evenly spaced samples from the middle + tail.

        Useful for long, uniformly structured documents (transcripts, registers,
        catalogues) where the middle carries as much signal as the edges.
        """
        if len(text) <= self.head_chars + self.tail_chars:
            return text
        middle_start, middle_end = self.head_chars, len(text) - self.tail_chars
        n = max(self.samples, 0)
        budget = max(self.max_chars - self.head_chars - self.tail_chars, 0)
        parts = [text[: self.head_chars]]
        if n and budget:
            window = min(budget // n, max(middle_end - middle_start, 1) // max(n, 1))
            if window > 200:
                step = (middle_end - middle_start) // (n + 1)
                for i in range(1, n + 1):
                    start = middle_start + step * i
                    parts.append(f"\n[...jumping to offset {start} of {len(text)}...]\n")
                    parts.append(text[start:start + window])
        parts.append("\n[...jumping to the end of the document...]\n")
        parts.append(text[-self.tail_chars:])
        return "".join(parts)

    def chunks(self, text: str) -> list[str]:
        """Split text for the map phase, honouring `chunk_chars` and `max_chunks`."""
        size = max(self.chunk_chars, 1000)
        overlap = max(min(self.chunk_overlap, size // 4), 0)
        out: list[str] = []
        pos = 0
        while pos < len(text) and len(out) < self.max_chunks:
            out.append(text[pos:pos + size])
            pos += size - overlap
        if pos < len(text) and out:
            out[-1] += "\n[...document continues beyond the indexed portion...]"
        return out or [text]

    @property
    def uses_map_reduce(self) -> bool:
        return self.strategy == "map_reduce"

    def wants_map_reduce(self, text: str) -> bool:
        """Map-reduce only pays off once the text exceeds a single excerpt."""
        return self.uses_map_reduce and len(text) > self.head_chars + self.tail_chars

    # -- output normalisation ---------------------------------------------
    def normalize(self, metadata: dict) -> dict:
        """Coerce model output to the declared types; keep unknown keys as-is."""
        out: dict = {}
        for key, value in metadata.items():
            spec = self.fields.get(key)
            out[key] = spec.coerce(value) if spec else value
        return out

    def missing_core_fields(self, metadata: dict) -> list[str]:
        """Core fields the profile declares but the model did not return."""
        return [f for f in CORE_FIELDS
                if f in self.fields and not metadata.get(f)]

    def filter_relations(self, metadata: dict) -> int:
        """Drop relations below `graph.min_confidence`, in place. Returns how many were dropped.

        A relation with no `strength` at all is kept — the threshold only
        screens out what the model itself flagged as weak, it never penalises
        a model that simply didn't fill the (optional) field in.
        """
        if not self.graph_enabled or self.graph.min_confidence <= 0:
            return 0
        relations = metadata.get(RELATIONS_FIELD)
        if not isinstance(relations, list):
            return 0
        kept = [r for r in relations
               if not isinstance(r, dict) or r.get("strength") is None
               or r["strength"] >= self.graph.min_confidence]
        dropped = len(relations) - len(kept)
        if dropped:
            metadata[RELATIONS_FIELD] = kept
        return dropped

    def llm_kwargs(self) -> dict:
        """Per-call overrides for `LLMClient.complete`."""
        kwargs = {}
        if self.model:
            kwargs["model"] = self.model
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.think is not None:
            kwargs["think"] = self.think
        return kwargs

    def summary_line(self) -> str:
        extra = [f for f in self.fields if f not in CORE_FIELDS]
        bits = [f"strategy={self.strategy}"]
        if extra:
            bits.append("extra fields: " + ", ".join(extra))
        if self.graph_enabled:
            bits.append("graph extraction" + (" (dedicated pass)" if self.graph.dedicated_pass else ""))
        if not self.match.empty:
            bits.append("auto-routed")
        return f"{self.name:<18} p{self.priority:<4} {self.description} [{'; '.join(bits)}]"


# ---------------------------------------------------------------------------
# Loading and inheritance
# ---------------------------------------------------------------------------

_SIMPLE_KEYS = {
    "description": "description", "priority": "priority",
}
_PROMPT_KEYS = {"role", "instructions", "guidance", "doc_types", "output_language",
                "map_instructions", "reduce_instructions"}
_EXCERPT_KEYS = {"strategy", "head_chars", "tail_chars", "max_chars", "chunk_chars",
                 "chunk_overlap", "max_chunks", "samples"}
_LLM_KEYS = {"model", "temperature", "max_tokens", "think", "ingestion"}
_TOP_KEYS = {"name", "description", "extends", "priority", "match", "prompt", "excerpt",
             "llm", "remove_fields", "graph"}
_STRATEGIES = {"head_tail", "head", "full", "sampled", "map_reduce"}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge `override` onto `base`; nested mappings merge, everything else replaces."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ProfileError(f"{path}: invalid YAML — {e}") from e
    if not isinstance(data, dict):
        raise ProfileError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise ProfileError(f"{path}: unknown top-level key(s) {sorted(unknown)}; "
                           f"allowed: {sorted(_TOP_KEYS)}")
    data.setdefault("name", path.stem)
    return data


def _resolve(name: str, raw: dict[str, dict], seen: tuple[str, ...] = ()) -> dict:
    """Return `name`'s raw spec with its inheritance chain applied."""
    if name in seen:
        raise ProfileError(f"circular profile inheritance: {' -> '.join((*seen, name))}")
    spec = raw.get(name)
    if spec is None:
        raise ProfileError(f"unknown profile {name!r}; available: {sorted(raw)}")
    parent = spec.get("extends", BASE_PROFILE if name != BASE_PROFILE else None)
    if not parent:
        return spec
    merged = _deep_merge(_resolve(parent, raw, (*seen, name)), spec)
    merged.pop("extends", None)
    # Routing and field pruning are never inherited, and never merged: a child
    # that states `match` replaces its parent's rules outright, and a child that
    # states none is simply not auto-routed.
    for key in ("match", "remove_fields"):
        if key in spec:
            merged[key] = spec[key]
        else:
            merged.pop(key, None)
    merged["name"] = spec["name"]
    if "description" in spec:
        merged["description"] = spec["description"]
    return merged


def _build(spec: dict, source: str) -> Profile:
    prompt = dict(spec.get("prompt") or {})
    excerpt = dict(spec.get("excerpt") or {})
    llm = dict(spec.get("llm") or {})

    for block, allowed, label in ((prompt, _PROMPT_KEYS | {"fields"}, "prompt"),
                                  (excerpt, _EXCERPT_KEYS, "excerpt"),
                                  (llm, _LLM_KEYS, "llm")):
        unknown = set(block) - allowed
        if unknown:
            raise ProfileError(f"profile {spec.get('name')!r}: unknown {label} key(s) "
                               f"{sorted(unknown)}; allowed: {sorted(allowed)}")

    fields_spec = dict(prompt.pop("fields", None) or {})
    for gone in spec.get("remove_fields") or []:
        fields_spec.pop(gone, None)
    fields = {name: FieldSpec.parse(name, value) for name, value in fields_spec.items()}

    graph_spec = spec.get("graph")
    graph_config: GraphConfig | None = None
    if graph_spec is not None:
        if not isinstance(graph_spec, dict):
            raise ProfileError(f"profile {spec.get('name')!r}: 'graph' must be a mapping, "
                               f"got {type(graph_spec).__name__}")
        unknown = set(graph_spec) - _GRAPH_KEYS
        if unknown:
            raise ProfileError(f"profile {spec.get('name')!r}: unknown graph key(s) "
                               f"{sorted(unknown)}; allowed: {sorted(_GRAPH_KEYS)}")
        try:
            min_confidence = float(graph_spec.get("min_confidence", 0.0))
        except (TypeError, ValueError):
            raise ProfileError(f"profile {spec.get('name')!r}: graph.min_confidence "
                               f"must be a number") from None
        graph_config = GraphConfig(
            entity_types=[str(t) for t in (graph_spec.get("entity_types") or [])],
            relation_types=[str(t) for t in (graph_spec.get("relation_types") or [])],
            dedicated_pass=bool(graph_spec.get("dedicated_pass", False)),
            min_confidence=min_confidence,
        )
        # A profile gets a `relations` field for free the moment it opts into
        # graph extraction — the same way every profile already gets `entities`
        # for free from `generic` — unless it wrote its own, which wins.
        if RELATIONS_FIELD not in fields:
            fields[RELATIONS_FIELD] = FieldSpec.parse(RELATIONS_FIELD, _RELATION_FIELD_SPEC)

    strategy = str(excerpt.get("strategy", "head_tail"))
    if strategy not in _STRATEGIES:
        raise ProfileError(f"profile {spec.get('name')!r}: unknown excerpt strategy "
                           f"{strategy!r}; allowed: {sorted(_STRATEGIES)}")
    ingestion = llm.get("ingestion")
    if ingestion not in (None, "parse", "upload", "auto"):
        raise ProfileError(f"profile {spec.get('name')!r}: ingestion must be "
                           f"parse|upload|auto, got {ingestion!r}")

    profile = Profile(
        name=str(spec["name"]),
        description=str(spec.get("description", "")),
        priority=int(spec.get("priority", 0)),
        match=MatchRules.parse(spec.get("match")),
        fields=fields,
        source=source,
        graph=graph_config,
    )
    for key in _PROMPT_KEYS:
        if key in prompt and prompt[key] is not None:
            setattr(profile, key, prompt[key])
    for key in _EXCERPT_KEYS:
        if key in excerpt and excerpt[key] is not None:
            setattr(profile, key, excerpt[key])
    for key in _LLM_KEYS:
        if key in llm and llm[key] is not None:
            setattr(profile, key, llm[key])
    return profile


@dataclass
class ProfileSet:
    """All known profiles, with routing."""

    profiles: dict[str, Profile]

    def __iter__(self) -> Iterator[Profile]:
        return iter(sorted(self.profiles.values(), key=lambda p: (-p.priority, p.name)))

    def __len__(self) -> int:
        return len(self.profiles)

    def __contains__(self, name: object) -> bool:
        return name in self.profiles

    def names(self) -> list[str]:
        return sorted(self.profiles)

    def get(self, name: str) -> Profile:
        try:
            return self.profiles[name]
        except KeyError:
            raise ProfileError(f"unknown profile {name!r}; available: {self.names()}") from None

    @property
    def default(self) -> Profile:
        return self.get(BASE_PROFILE)

    def routed(self) -> list[Profile]:
        """Profiles that can claim documents automatically, highest priority first."""
        return [p for p in self if not p.match.empty]

    def needs_text_for_routing(self) -> bool:
        """True when any routed profile matches on document content."""
        return any(p.match.needs_text() for p in self.routed())

    def select(self, ctx: MatchContext, default: str | None = None) -> Profile:
        """Pick the highest-priority profile whose match rules all hold."""
        for profile in self.routed():
            if profile.match.matches(ctx):
                return profile
        return self.get(default or BASE_PROFILE)


def load_profiles(extra_dirs: Iterable[str | Path] = (),
                  files: Iterable[str | Path] = ()) -> ProfileSet:
    """Load built-in profiles, then any user profiles (which may override them).

    `extra_dirs` are scanned for `*.yaml`/`*.yml`; `files` are loaded directly.
    A user profile reusing a built-in name replaces it entirely.
    """
    raw: dict[str, dict] = {}
    origin: dict[str, str] = {}

    def ingest(path: Path) -> None:
        spec = _read_yaml(path)
        name = str(spec["name"])
        if name in raw and origin[name] != str(path):
            log.info("Profile %r from %s overrides the one from %s", name, path, origin[name])
        raw[name] = spec
        origin[name] = str(path)

    for path in sorted(BUILTIN_DIR.glob("*.yaml")):
        ingest(path)
    for directory in extra_dirs:
        d = Path(directory)
        if not d.is_dir():
            raise ProfileError(f"profile directory not found: {d}")
        for path in sorted([*d.glob("*.yaml"), *d.glob("*.yml")]):
            ingest(path)
    for f in files:
        path = Path(f)
        if not path.is_file():
            raise ProfileError(f"profile file not found: {path}")
        ingest(path)

    if BASE_PROFILE not in raw:
        raise ProfileError(f"the {BASE_PROFILE!r} profile is missing from {BUILTIN_DIR}")

    return ProfileSet({name: _build(_resolve(name, raw), origin[name]) for name in raw})
