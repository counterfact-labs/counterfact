"""
LLM provider abstraction with response caching.

Provides a unified interface for calling LLMs (Google Gemini, Anthropic Claude)
with persistent response caching to minimize cost and latency during repeated
diagnostic runs.

Configuration:
    API keys are read from environment variables:
      - GOOGLE_API_KEY or GEMINI_API_KEY — for Google Gemini
      - ANTHROPIC_API_KEY — for Anthropic Claude

    Model names default to stable aliases but can be overridden:
      - COUNTERFACT_GEMINI_MODEL — default: "gemini-2.5-flash"
      - COUNTERFACT_ANTHROPIC_MODEL — default: "claude-sonnet-4-20250514"

    Cache location can be overridden:
      - COUNTERFACT_CACHE_DIR — default: ~/.cache/counterfact/

Dependencies: none (LLM SDKs are optional, imported on demand)
"""

import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

_ANTHROPIC_MODEL = os.environ.get("COUNTERFACT_ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
_GEMINI_MODEL = os.environ.get("COUNTERFACT_GEMINI_MODEL", "gemini-2.5-flash")


def _get_cache_dir() -> str:
    """Get the cache directory, creating it if necessary.

    Priority: COUNTERFACT_CACHE_DIR env var > ~/.counterfact > project-local fallback.
    """
    candidates = [
        os.environ.get("COUNTERFACT_CACHE_DIR", ""),
        os.path.join(os.path.expanduser("~"), ".counterfact"),
    ]
    for path in candidates:
        if not path:
            continue
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except OSError:
            continue
    # Last resort: use a local directory
    fallback = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".counterfact_cache")
    os.makedirs(fallback, exist_ok=True)
    return fallback


_LLM_CACHE_FILE = os.path.join(_get_cache_dir(), "llm_cache.json")
_llm_cache: dict = {}


def load_cache() -> None:
    """Load the LLM response cache from disk."""
    global _llm_cache
    if not _llm_cache and os.path.exists(_LLM_CACHE_FILE):
        try:
            with open(_LLM_CACHE_FILE, "r") as f:
                _llm_cache = json.loads(f.read())
        except Exception:
            _llm_cache = {}


def save_cache() -> None:
    """Persist the LLM response cache to disk."""
    try:
        with open(_LLM_CACHE_FILE, "w") as f:
            f.write(json.dumps(_llm_cache))
    except Exception:
        pass


def get_cache_key(prompt: str, temperature: float) -> str:
    """Generate a deterministic cache key from prompt + temperature."""
    return hashlib.sha256(f"{prompt}|{temperature}".encode()).hexdigest()[:16]


def call_llm(prompt: str, temperature: float = 0.1) -> str:
    """Unified LLM call with caching and provider fallback.

    Tries Google Gemini first, falls back to Anthropic Claude.
    Responses are cached to disk for reproducibility and cost savings.

    Args:
        prompt: The prompt to send to the LLM.
        temperature: Sampling temperature (0.0–1.0).

    Returns:
        The LLM response text, or "" if no provider is available.
    """
    load_cache()
    key = get_cache_key(prompt, temperature)
    if key in _llm_cache:
        return _llm_cache[key]

    # Try Google Gemini first
    google_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
    if google_key:
        try:
            from google import genai  # type: ignore
            client = genai.Client(api_key=google_key)
            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=2000,
                ),
            )
            result = response.text or ""
            _llm_cache[key] = result
            save_cache()
            return result
        except Exception as e:
            logger.warning("Gemini call failed: %s", str(e)[:100])

    # Fall back to Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        try:
            import anthropic  # type: ignore
            anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            anthropic_response = anthropic_client.messages.create(
                model=_ANTHROPIC_MODEL,
                max_tokens=2000,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            block = anthropic_response.content[0]
            result = getattr(block, "text", str(block))
            _llm_cache[key] = result
            save_cache()
            return result
        except Exception as e:
            logger.warning("Anthropic call failed: %s", str(e)[:100])

    return ""
