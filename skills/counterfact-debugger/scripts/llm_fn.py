"""
Default LLM caller for counterfact quality classifiers.

Resolves a provider from environment variables and returns a function with the
signature counterfact expects:  ``(prompt: str, temperature: float = 0.1) -> str``.

Provider resolution (first match wins):
  1. ANTHROPIC_API_KEY            -> Anthropic Claude
  2. GOOGLE_API_KEY / GEMINI_API_KEY -> Google Gemini

Override the model with COUNTERFACT_MODEL.

Usable two ways:
  - Imported:   from llm_fn import make_llm_fn; fn = make_llm_fn()
  - Standalone: python llm_fn.py "your prompt"   (prints the response; smoke test)

Returns None from make_llm_fn() if no key is set, so callers can degrade to
structural-only classifiers instead of crashing.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Optional


def make_llm_fn(provider: Optional[str] = None) -> Optional[Callable[[str, float], str]]:
    """Build an ``(prompt, temperature) -> str`` LLM caller from the environment.

    Args:
        provider: Force a provider ("anthropic" or "google"). If None, auto-detect
            from whichever API key is present.

    Returns:
        A caller function, or None if no usable key/provider is available.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    google_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

    if provider == "anthropic" or (provider is None and anthropic_key):
        if not anthropic_key:
            raise RuntimeError("provider=anthropic requires ANTHROPIC_API_KEY")
        return _anthropic_caller(anthropic_key)

    if provider == "google" or (provider is None and google_key):
        if not google_key:
            raise RuntimeError("provider=google requires GOOGLE_API_KEY or GEMINI_API_KEY")
        return _google_caller(google_key)

    if provider is not None:
        raise ValueError(f"Unknown provider {provider!r}; use 'anthropic' or 'google'.")

    return None  # no key set — caller should degrade to structural-only


def _anthropic_caller(api_key: str) -> Callable[[str, float], str]:
    try:
        import anthropic  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install counterfact[anthropic]  (anthropic not installed)") from e

    client = anthropic.Anthropic(api_key=api_key)
    model = os.environ.get("COUNTERFACT_MODEL", "claude-sonnet-4-6")

    def caller(prompt: str, temperature: float = 0.1) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        block = resp.content[0]
        return getattr(block, "text", str(block))

    caller.model = model  # type: ignore[attr-defined]
    return caller


def _google_caller(api_key: str) -> Callable[[str, float], str]:
    try:
        from google import genai  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install counterfact[google]  (google-genai not installed)") from e

    client = genai.Client(api_key=api_key)
    model = os.environ.get("COUNTERFACT_MODEL", "gemini-2.5-flash")

    def caller(prompt: str, temperature: float = 0.1) -> str:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=2000,
            ),
        )
        return resp.text or ""

    caller.model = model  # type: ignore[attr-defined]
    return caller


def describe() -> str:
    """One-line description of the resolved provider, for logging."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return f"Anthropic ({os.environ.get('COUNTERFACT_MODEL', 'claude-sonnet-4-6')})"
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return f"Google Gemini ({os.environ.get('COUNTERFACT_MODEL', 'gemini-2.5-flash')})"
    return "none (structural-only checks)"


if __name__ == "__main__":  # pragma: no cover
    fn = make_llm_fn()
    if fn is None:
        print("No LLM key found. Set ANTHROPIC_API_KEY or GOOGLE_API_KEY.", file=sys.stderr)
        sys.exit(1)
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Reply with the single word: ok"
    print(f"[provider: {describe()}]", file=sys.stderr)
    print(fn(prompt, 0.0))
