"""Cached Anthropic caller shared by the pipeline agents and LLM classifiers.

PERSISTENT, SHARED cache: responses are stored on disk keyed by
(model, temperature, max_tokens, prompt). This makes every FinanceBench script
interruption-safe — kill it and restart, and already-computed calls return instantly
from disk instead of re-hitting the API. The cache is also shared across the
case-study run AND the behavioral-eval agent runs (they all import this module), so
repeated runs are cheap.

Cache location (stable absolute path so it survives restarts and is shared across the
temp workspaces the eval copies the package into):
    $FB_LLM_CACHE  (if set)  else  ~/.cache/financebench_casestudy/

temperature defaults to 0.0. Disk writes are atomic (temp + os.replace), so the cache
is safe under the ThreadPoolExecutor in run_casestudy and across concurrent processes.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import sys
import threading

SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5"

_CACHE_DIR = pathlib.Path(
    os.environ.get("FB_LLM_CACHE", pathlib.Path.home() / ".cache" / "financebench_casestudy")
)
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_client = None
_mem: dict[str, str] = {}
_lock = threading.Lock()
_hits = 0
_misses = 0


def get_client():
    global _client
    if _client is None:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            print("ERROR: set ANTHROPIC_API_KEY", file=sys.stderr)
            sys.exit(1)
        import anthropic
        _client = anthropic.Anthropic(api_key=key)
    return _client


def _key(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    h = hashlib.sha256(repr((model, temperature, max_tokens, prompt)).encode("utf-8"))
    return h.hexdigest()


def call(prompt: str, model: str = HAIKU, temperature: float = 0.0, max_tokens: int = 1000) -> str:
    global _hits, _misses
    k = _key(prompt, model, temperature, max_tokens)
    # 1. in-memory
    with _lock:
        if k in _mem:
            _hits += 1
            return _mem[k]
    # 2. on-disk (survives restarts / shared across runs)
    path = _CACHE_DIR / f"{k}.txt"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        with _lock:
            _mem[k] = text
            _hits += 1
        return text
    # 3. live call -> persist atomically
    resp = get_client().messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    with _lock:
        _mem[k] = text
        _misses += 1
    return text


def cache_stats() -> dict:
    """(hits, misses, dir) — useful for progress lines; high hits on a restart means
    the persistent cache is doing its job."""
    with _lock:
        return {"hits": _hits, "misses": _misses, "dir": str(_CACHE_DIR)}
