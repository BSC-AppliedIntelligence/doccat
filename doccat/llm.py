"""Thread-safe client for OpenAI-compatible `/chat/completions` endpoints.

Built for the indexing pipeline, where dozens of worker threads hammer one
endpoint at once:

* one pooled `httpx.Client` is shared by every worker (keep-alive connections
  instead of a fresh TCP+TLS handshake per document);
* endpoint quirks discovered by one worker (a rejected `chat_template_kwargs`,
  a hard completion-token cap) are recorded once and applied by all the others;
* a shared `RateLimitGate` turns a 429 from any worker into a short pause for
  *all* of them, so an overloaded endpoint is not stampeded;
* an optional token bucket caps the global request rate.

Everything here is provider-agnostic: vLLM, llama.cpp, TGI, Ollama, LM Studio
and the Hugging Face router all speak this API.
"""

from __future__ import annotations

import base64
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

log = logging.getLogger("doccat.llm")

MIME_FALLBACK = "application/octet-stream"


class EmptyLLMReply(Exception):
    """The endpoint answered 200 but with no usable text content."""


class LLMError(RuntimeError):
    """The endpoint could not be made to answer this request."""


@dataclass
class LLMConfig:
    """Connection + sampling settings shared by every worker."""

    base_url: str
    model: str
    api_key: str = "not-needed"
    temperature: float = 0.1
    max_tokens: int = 4096           # generous: reasoning models think before answering
    timeout: float = 300.0
    max_retries: int = 3
    max_rate_limit_retries: int = 8  # separate, more patient budget for 429s
    think: bool = False              # keep reasoning mode on (Qwen3 etc.)
    extra_headers: dict[str, str] = field(default_factory=dict)
    max_connections: int = 64        # httpx pool size; raise with --llm-workers
    requests_per_second: float = 0.0  # 0 = unlimited


class TokenBucket:
    """Simple thread-safe rate limiter (`requests_per_second`, burst = 1 s)."""

    def __init__(self, rate: float) -> None:
        self.rate = rate
        self._tokens = rate
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.rate <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.rate, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait = (1 - self._tokens) / self.rate
            time.sleep(wait)


class RateLimitGate:
    """Shared back-pressure: one worker's 429 pauses every worker.

    Endpoints that reply 429 ("model busy", `engine_overloaded`) are usually
    saturated, so having 32 threads each retry on their own private schedule
    only makes it worse. Workers call `wait()` before every request and
    `penalize()` when throttled; the pause applies to the whole pool.
    """

    def __init__(self, max_pause: float = 300.0) -> None:
        self._blocked_until = 0.0
        self._consecutive = 0
        self._lock = threading.Lock()
        self._max_pause = max_pause

    def wait(self) -> None:
        while True:
            with self._lock:
                remaining = self._blocked_until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 5.0))

    def penalize(self, retry_after: float | None = None) -> float:
        """Register a throttling event; returns the pause applied, in seconds."""
        with self._lock:
            self._consecutive += 1
            if retry_after is not None:
                pause = min(retry_after, self._max_pause)
            else:
                pause = min(5 * 2 ** (self._consecutive - 1), self._max_pause)
            pause += random.uniform(0, pause * 0.25)  # de-synchronise the pool
            self._blocked_until = max(self._blocked_until, time.monotonic() + pause)
            return pause

    def succeed(self) -> None:
        """A request went through — decay the penalty level."""
        with self._lock:
            if self._consecutive:
                self._consecutive = max(0, self._consecutive - 1)


class LLMClient:
    """Pooled, thread-safe chat-completions client."""

    def __init__(self, cfg: LLMConfig, gate: RateLimitGate | None = None) -> None:
        self.cfg = cfg
        self.gate = gate or RateLimitGate()
        self._bucket = TokenBucket(cfg.requests_per_second)
        self._lock = threading.Lock()
        # Endpoint quirks learned at runtime and shared by all workers.
        self._supports_template_kwargs = True
        self._token_cap: int | None = None
        self._client = httpx.Client(
            timeout=cfg.timeout,
            limits=httpx.Limits(max_connections=cfg.max_connections,
                                max_keepalive_connections=cfg.max_connections),
            headers={"Authorization": f"Bearer {cfg.api_key}", **cfg.extra_headers},
        )

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- internals ---------------------------------------------------------
    def _note_token_cap(self, cap: int) -> None:
        with self._lock:
            self._token_cap = cap if self._token_cap is None else min(self._token_cap, cap)

    def _effective_max_tokens(self, requested: int) -> int:
        with self._lock:
            cap = self._token_cap
        return min(requested, cap) if cap else requested

    def _disable_template_kwargs(self) -> None:
        with self._lock:
            self._supports_template_kwargs = False

    @staticmethod
    def _extract_reply_text(data: dict) -> str:
        """Get the model's text out of a chat.completions response.

        Reasoning models put text in `reasoning_content` when `content` is
        empty (thinking consumed the whole token budget); the JSON parser can
        still dig the object out of that.
        """
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        content = (msg.get("content") or "").strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if content:
            return content
        reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
        if reasoning and "{" in reasoning:
            return reasoning
        raise EmptyLLMReply(
            f"empty content (finish_reason={choice.get('finish_reason')!r}). "
            "Likely a reasoning model that spent its whole token budget thinking — "
            "raise --max-tokens or run without --think.")

    def _build_content(self, prompt: str, file_path: Path | None,
                       mime_types: dict[str, str] | None) -> object:
        if file_path is None:
            return prompt
        b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")
        mime = (mime_types or {}).get(file_path.suffix.lower(), MIME_FALLBACK)
        return [
            {"type": "text", "text": prompt},
            {"type": "file", "file": {"filename": file_path.name,
                                      "file_data": f"data:{mime};base64,{b64}"}},
        ]

    # -- public API --------------------------------------------------------
    def complete(self, prompt: str, *, file_path: Path | None = None,
                 model: str | None = None, temperature: float | None = None,
                 max_tokens: int | None = None, think: bool | None = None,
                 mime_types: dict[str, str] | None = None) -> str:
        """Run one completion and return the reply text.

        Per-call overrides let a cataloguing profile use its own model,
        temperature or token budget without a second client.
        """
        cfg = self.cfg
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        budget = self._effective_max_tokens(max_tokens or cfg.max_tokens)
        want_think = cfg.think if think is None else think

        payload: dict = {
            "model": model or cfg.model,
            "messages": [{"role": "user",
                          "content": self._build_content(prompt, file_path, mime_types)}],
            "temperature": cfg.temperature if temperature is None else temperature,
            "max_tokens": budget,
        }

        attempt = 0          # network / 5xx / empty-reply budget
        rate_limit_hits = 0  # 429 budget (separate and more patient)
        last_err: Exception | None = None

        while attempt < cfg.max_retries and rate_limit_hits < cfg.max_rate_limit_retries:
            with self._lock:
                use_template_kwargs = self._supports_template_kwargs
            if not want_think and use_template_kwargs:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            else:
                payload.pop("chat_template_kwargs", None)

            self.gate.wait()
            self._bucket.acquire()
            try:
                resp = self._client.post(url, json=payload)
                if resp.status_code == 400 and "chat_template_kwargs" in payload:
                    log.info("Endpoint rejected chat_template_kwargs — disabling it for all workers")
                    self._disable_template_kwargs()
                    payload.pop("chat_template_kwargs")
                    resp = self._client.post(url, json=payload)
                if resp.status_code == 400:
                    # e.g. "max_completion_tokens is limited to 32768 for <model>"
                    m = re.search(r"limited to (\d+)", resp.text)
                    if m and payload["max_tokens"] > int(m.group(1)):
                        cap = int(m.group(1))
                        self._note_token_cap(cap)
                        payload["max_tokens"] = cap
                        log.info("Endpoint caps completion tokens at %d — retrying", cap)
                        resp = self._client.post(url, json=payload)
                resp.raise_for_status()
                self.gate.succeed()
                return self._extract_reply_text(resp.json())

            except EmptyLLMReply as e:
                last_err = e
                attempt += 1
                budget = self._effective_max_tokens(budget * 2)
                if payload["max_tokens"] == budget:  # already at the cap, no point retrying
                    break
                payload["max_tokens"] = budget
                log.warning("LLM reply empty (attempt %d/%d): %s — retrying with max_tokens=%d",
                            attempt, cfg.max_retries, e, budget)

            except httpx.HTTPStatusError as e:
                last_err = e
                status = e.response.status_code
                if status == 429 or status == 503:
                    rate_limit_hits += 1
                    retry_after = e.response.headers.get("Retry-After")
                    hint = float(retry_after) if (retry_after or "").isdigit() else None
                    pause = self.gate.penalize(hint)
                    log.warning("Endpoint busy (%d, hit %d/%d) — pausing all workers %.0fs",
                                status, rate_limit_hits, cfg.max_rate_limit_retries, pause)
                    continue
                if 400 <= status < 500:
                    raise LLMError(
                        f"LLM endpoint rejected the request ({status}) — "
                        f"{e.response.text[:300]}"
                        + (" The endpoint likely does not accept file attachments; "
                           "use ingestion mode 'parse'." if file_path else "")) from e
                attempt += 1
                wait = 2 ** attempt
                log.warning("LLM call failed (attempt %d/%d): %s — retrying in %ds",
                            attempt, cfg.max_retries, e, wait)
                time.sleep(wait)

            except (httpx.HTTPError, KeyError, IndexError) as e:
                last_err = e
                attempt += 1
                wait = 2 ** attempt
                log.warning("LLM call failed (attempt %d/%d): %s — retrying in %ds",
                            attempt, cfg.max_retries, e, wait)
                time.sleep(wait)

        if rate_limit_hits >= cfg.max_rate_limit_retries:
            raise LLMError(f"LLM endpoint still overloaded after {rate_limit_hits} "
                           f"rate-limit retries: {last_err}")
        raise LLMError(f"LLM call failed after {attempt} attempts: {last_err}")
