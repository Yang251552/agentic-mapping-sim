"""Stage 5 LLM client: one forced `go_to` tool call per decision, every call cached in a JSONL file in the repo.

Replay (default) answers only from the cache and raises CacheMiss on a missing key; it never calls the API or falls
back silently. Live mode answers from the cache when it can and otherwise calls the API, appending the call
(arguments, usage, latency, model snapshot fields) to the cache. Cache key = sha256 of the canonical request body
plus the retry attempt number. A hard spend cap is checked against every live call recorded in llm_cache/.
Stdlib only (urllib), so the replay path needs no API key and no extra package.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

API = "https://api.openai.com/v1/chat/completions"
# USD per 1M tokens (input, cached input, output), standard tier, developers.openai.com/api/docs/pricing, 2026-10-06
PRICES = {"gpt-4.1-mini": (0.40, 0.10, 1.60), "gpt-5.6-luna": (0.20, 0.02, 1.20), "gpt-4o-mini": (0.15, 0.075, 0.60)}
CACHE_DIR = pathlib.Path("llm_cache")


def secret(name: str) -> str:
    """Environment first (local runs); on the AWS instance the value lives only in ~/amap/.secrets.env (mode 600),
    which remote_run.sh validates but never exports, so no shell or log in the job ever holds it."""
    if os.environ.get(name):
        return os.environ[name]
    f = pathlib.Path(os.environ.get("AMAP_SECRETS", pathlib.Path.home() / "amap" / ".secrets.env"))
    if f.exists():
        for line in f.read_text().splitlines():
            k, _, v = line.partition("=")
            if k == name and v:
                return v
    raise KeyError(f"{name} not set")


class CacheMiss(RuntimeError):
    pass


class InfraError(RuntimeError):
    """Rate limit / 5xx / timeout still failing after the backoff budget: the episode is aborted_infra."""


class BudgetExceeded(RuntimeError):
    pass


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def price_of(model: str) -> tuple:
    base = next((m for m in sorted(PRICES, key=len, reverse=True) if model.startswith(m)), None)
    if base is None:
        raise KeyError(f"no price for {model}: add it to PRICES from the official pricing page")
    return PRICES[base]


def cost_usd(model: str, usage: dict) -> float:
    p_in, p_cached, p_out = price_of(model)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    return ((usage.get("prompt_tokens", 0) - cached) * p_in + cached * p_cached + usage.get("completion_tokens", 0) * p_out) / 1e6


def spent_usd(cache_dir=CACHE_DIR) -> float:
    total = 0.0
    for f in sorted(pathlib.Path(cache_dir).glob("*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            total += r.get("cost_usd", 0.0)
    return total


def go_to_tool(candidate_ids) -> dict:
    return {"type": "function", "function": {
        "name": "go_to", "strict": True,
        "description": "Choose the next exploration target from the candidate list.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "required": ["candidate_id", "reason", "predicted_beyond"],
                       "properties": {
                           "candidate_id": {"type": "string", "enum": list(candidate_ids)},
                           "reason": {"type": "string", "description": "One sentence: why this target serves the task now."},
                           "predicted_beyond": {"type": "string", "description": "What you expect to find there (room type or objects)."}}}}}


def request_body(model: str, system: str, user: str, candidate_ids, max_tokens=300) -> dict:
    return {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "tools": [go_to_tool(candidate_ids)],
            "tool_choice": {"type": "function", "function": {"name": "go_to"}}}


class LLMClient:
    def __init__(self, cache_file, live=False, max_usd=4.0, max_backoff_tries=6, timeout=60):
        self.cache_file = pathlib.Path(cache_file)
        self.live, self.max_usd, self.tries, self.timeout = live, max_usd, max_backoff_tries, timeout
        self.cache = {}
        if self.cache_file.exists():
            for line in self.cache_file.open():
                r = json.loads(line)
                self.cache[r["req_sha256"]] = r

    def call(self, body: dict, attempt: int = 0) -> dict:
        """Returns the cache record: {"req_sha256", "args" (parsed tool arguments or None), "usage", ...}."""
        key = sha256(canonical(body) + f"#attempt={attempt}")
        if key in self.cache:
            return self.cache[key]
        if not self.live:
            raise CacheMiss(f"no cached LLM answer for key {key[:16]} (replay mode never calls the API)")
        if spent_usd(self.cache_file.parent) >= self.max_usd:
            raise BudgetExceeded(f"LLM spend cap ${self.max_usd} reached")
        resp, latency = self._post(body)
        msg = resp["choices"][0]["message"]
        args = None
        try:
            args = json.loads(msg["tool_calls"][0]["function"]["arguments"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            pass
        rec = {"req_sha256": key, "attempt": attempt, "args": args, "raw": msg.get("tool_calls") and msg["tool_calls"][0]["function"]["arguments"],
               "model": resp.get("model"), "system_fingerprint": resp.get("system_fingerprint"),
               "usage": resp.get("usage", {}), "cost_usd": round(cost_usd(body["model"], resp.get("usage", {})), 8),
               "latency_s": round(latency, 3), "created": resp.get("created")}
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_file.open("a") as f:
            f.write(canonical(rec) + "\n")
        self.cache[key] = rec
        return rec

    def _post(self, body):
        data = json.dumps(body).encode()
        delay = 2.0
        for i in range(self.tries):
            req = urllib.request.Request(API, data=data, headers={
                "Authorization": "Bearer " + secret("OPENAI_API_KEY"), "Content-Type": "application/json"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r), time.time() - t0
            except urllib.error.HTTPError as e:
                if e.code != 429 and e.code < 500:
                    raise RuntimeError(f"OpenAI HTTP {e.code}: {e.read()[:300]!r}") from None
            except (urllib.error.URLError, OSError, http.client.HTTPException):   # OSError covers socket.timeout, which on
                pass                                                             # Python 3.9 is not a TimeoutError (held-out 10-07)
            if i + 1 < self.tries:
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise InfraError(f"OpenAI unavailable after {self.tries} tries")
