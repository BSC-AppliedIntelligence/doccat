"""Tolerant parsing of the JSON metadata replies produced by an LLM.

Models return broken JSON in predictable ways: markdown fences, commentary
around the object, unescaped quotes/newlines copied from the source document
(very common with transcripts full of quoted speech), replies truncated by the
token budget, and stray `{...}` fragments left over from a reasoning phase.

`parse_metadata` applies increasingly forgiving strategies, stopping at the
first that yields a dict that looks like document metadata.
"""

from __future__ import annotations

import json
import re

try:  # optional, strongly recommended
    from json_repair import repair_json
except ImportError:  # pragma: no cover - exercised only without the dependency
    repair_json = None

__all__ = ["parse_metadata", "parse_metadata_json", "find_brace_blocks", "repair_available"]

# A repaired object is accepted as "our" metadata only if it carries at least
# one of these keys, so unrelated braced snippets are not mistaken for a reply.
_METADATA_KEYS = {
    "title", "summary", "description", "doc_type", "language",
    "keywords", "topics", "entities", "questions_answered",
}


def repair_available() -> bool:
    """True when the optional `json_repair` dependency is installed."""
    return repair_json is not None


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    return text.strip()


def parse_metadata_json(raw: str) -> dict:
    """Strict-ish parse: tolerate markdown fences and surrounding commentary only."""
    text = _strip_fences(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)  # first { ... last }
        if match:
            return json.loads(match.group(0))
        raise


def find_brace_blocks(text: str) -> list[str]:
    """Return every top-level balanced ``{...}`` substring of `text`.

    String/escape state is tracked so braces inside string values do not
    confuse the scan; unlike a greedy ``\\{.*\\}`` regex this cannot merge two
    unrelated objects into one unparseable blob. An unterminated final block
    (reply truncated mid-object) is returned as-is so `json_repair` can close it.
    """
    blocks: list[str] = []
    depth = 0
    start: int | None = None
    in_string = escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(text[start:i + 1])
                start = None
    if depth > 0 and start is not None:
        blocks.append(text[start:])
    return blocks


def parse_metadata(raw: str) -> dict:
    """Parse an LLM metadata reply, repairing common malformations.

    Strategy, each step tried only if the previous one failed:

    1. `parse_metadata_json` — the fast path for clean replies.
    2. `json_repair` over the whole reply.
    3. `json_repair` over each top-level ``{...}`` block, preferring the first
       one that actually looks like the metadata schema.

    Raises `ValueError` / `json.JSONDecodeError` when nothing usable is found.
    """
    try:
        return parse_metadata_json(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    if repair_json is None:
        return parse_metadata_json(raw)  # re-raise the original, clearer error

    text = _strip_fences(raw)

    whole = repair_json(text, return_objects=True)
    if isinstance(whole, dict) and whole and _METADATA_KEYS & whole.keys():
        return whole

    best: dict | None = None
    for block in find_brace_blocks(text):
        obj = repair_json(block, return_objects=True)
        if isinstance(obj, dict) and obj:
            if _METADATA_KEYS & obj.keys():
                return obj
            if best is None:
                best = obj

    if isinstance(whole, dict) and whole:
        return whole
    if best is not None:
        return best
    raise ValueError("could not extract usable JSON metadata from LLM reply "
                     "(no repairable {...} block matching the metadata schema)")
