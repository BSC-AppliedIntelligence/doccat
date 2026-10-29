"""Shared fixtures: a mock OpenAI-compatible endpoint and a small document tree."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class MockLLM:
    """A tiny threaded /v1/chat/completions server.

    Records every request and can be told to fail the first N calls with 429 so
    the shared rate-limit gate can be exercised.
    """

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        self.fail_with_429 = 0
        self.break_refine = False       # make only the refine call unparseable
        self.reply: dict | str | None = None
        self.max_concurrent = 0
        self._active = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # silence
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                with server.lock:
                    server.requests.append(payload)
                    server._active += 1
                    server.max_concurrent = max(server.max_concurrent, server._active)
                    throttle = server.fail_with_429 > 0
                    if throttle:
                        server.fail_with_429 -= 1
                try:
                    if throttle:
                        body = b'{"error": "engine_overloaded"}'
                        self.send_response(429)
                        self.send_header("Retry-After", "1")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    prompt = payload["messages"][0]["content"]
                    if not isinstance(prompt, str):
                        prompt = " ".join(p.get("text", "") for p in prompt)
                    content = server._make_reply(prompt)
                    body = json.dumps({
                        "choices": [{"message": {"content": content},
                                     "finish_reason": "stop"}]
                    }).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                finally:
                    with server.lock:
                        server._active -= 1

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    #: Overrides for the analysis JSON returned to `doccat optimize`.
    analysis_override: dict | None = None
    #: Override for the relations list returned by a dedicated relation-extraction call.
    relations_reply: list | None = None

    def _make_reply(self, prompt: str) -> str:
        if isinstance(self.reply, str):
            return self.reply
        if "no JSON, no preamble" in prompt or "write dense factual notes" in prompt:
            return "Notes about this part of the document."
        if "You are designing the cataloguing strategy" in prompt:
            return json.dumps(self._analysis(prompt))
        if "You designed the cataloguing profile below" in prompt and self.break_refine:
            return "I would rather not answer in JSON."
        if "You designed the cataloguing profile below" in prompt:
            revised = self._analysis(prompt)
            revised["fields"].append({"name": "renewal_terms", "type": "string",
                                      "description": "renewal conditions, or empty"})
            revised["reasoning"] = "Added renewal_terms; dropped nothing."
            return json.dumps(revised)
        if "already identified these entities in this document" in prompt:
            # The dedicated (second) relation-extraction call.
            return json.dumps({"relations": self.relations_reply if self.relations_reply
                               is not None else [
                {"subject": "ACME", "relation": "supplies_to", "object": "Contoso",
                 "description": "ACME supplies widgets to Contoso.", "strength": 0.9},
            ]})
        base = {
            "title": "Mock title",
            "summary": "A mock summary of the document.",
            "description": "A mock description.",
            "doc_type": "report",
            "language": "en",
            "keywords": ["alpha", "beta"],
            "topics": ["testing"],
            "entities": {"people": [], "organizations": ["ACME"], "places": [], "dates": []},
            "questions_answered": ["What is this?"],
        }
        if isinstance(self.reply, dict):
            base.update(self.reply)
        # Echo profile-specific fields so routing can be asserted from the catalog.
        if "legal document cataloguer" in prompt:
            base["parties"] = ["ACME", "Contoso"]
        if "parliamentary proceedings" in prompt:
            base["chamber"] = "Congreso de los Diputados"
        # A graph-enabled profile's *inline* (non-dedicated) call asks for
        # `relations` in the same schema as everything else — answer it there.
        if '"relations"' in prompt:
            base["relations"] = [
                {"subject": "ACME", "relation": "supplies_to", "object": "Contoso",
                 "description": "ACME supplies widgets to Contoso.", "strength": 0.9},
                {"subject": "Contoso", "relation": "pays", "object": "ACME",
                 "description": "Contoso pays ACME under the agreement.", "strength": 0.1},
            ]
        return json.dumps(base)

    def _analysis(self, prompt: str) -> dict:
        base = {
            "corpus_summary": "Supplier contracts and installation manuals for one company.",
            "domain": "procurement and support",
            "document_families": ["supply contracts", "installation manuals"],
            "dominant_language": "en",
            "profile_name": "Company Docs!",          # deliberately messy: must be sanitised
            "description": "Contracts and manuals of one company.",
            "extends": "legal",
            "role": "You are a company records cataloguer.",
            "guidance": "Quote amounts verbatim. Never infer a party.",
            "doc_types": ["Supply Contract", "installation_manual", "other"],
            "fields": [
                {"name": "counterparties", "type": "list",
                 "description": "the other companies involved"},
                {"name": "Effective Date", "type": "string",
                 "description": "start date, ISO 8601, or empty"},
                {"name": "title", "type": "string", "description": "core field, must be dropped"},
            ],
            "strategy": "head_tail",
            "strategy_reason": "Key terms sit at the start and the signature block at the end.",
            "head_chars": 9000,
            "tail_chars": 2500,
            "match_path_globs": ["**/contracts/**", "**/manuals/**"],
            "match_filename_regex": "(contract|manual)",
            "reasoning": "Counterparty and effective date are how these are looked up.",
        }
        if self.analysis_override:
            base.update(self.analysis_override)
        return base

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def call_count(self) -> int:
        with self.lock:
            return len(self.requests)

    def prompts(self) -> list[str]:
        with self.lock:
            out = []
            for r in self.requests:
                content = r["messages"][0]["content"]
                out.append(content if isinstance(content, str)
                           else " ".join(p.get("text", "") for p in content))
            return out

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def mock_llm():
    server = MockLLM()
    yield server
    server.close()


@pytest.fixture
def docs_tree(tmp_path: Path) -> Path:
    """A small corpus with files that route to different profiles."""
    root = tmp_path / "docs"
    (root / "contracts").mkdir(parents=True)
    (root / "manuals").mkdir(parents=True)
    (root / "misc").mkdir(parents=True)
    (root / "contracts" / "supply_contract_2024.md").write_text(
        "# Supply contract\nACME and Contoso agree...\n" + "clause. " * 200, encoding="utf-8")
    (root / "contracts" / "nda_acme.txt").write_text("NDA between ACME and Contoso.",
                                                     encoding="utf-8")
    (root / "manuals" / "installation_guide.md").write_text(
        "# Installation guide\nRun the installer.\n" + "step. " * 200, encoding="utf-8")
    (root / "misc" / "notes.txt").write_text("Random notes about nothing in particular.",
                                             encoding="utf-8")
    (root / "misc" / "readme.html").write_text("<html><body><h1>Hello</h1></body></html>",
                                               encoding="utf-8")
    return root
