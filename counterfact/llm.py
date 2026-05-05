"""
LLM provider and response caching.

This module provides a unified interface for calling LLMs (Gemini, Anthropic)
and handles persistent response caching to minimize cost and latency during
repeated diagnostic runs.
"""

import os
import json
import hashlib

# Load configuration (assumes root config.py is reachable)
try:
    from config import get_anthropic_api_key, get_google_api_key, get_api_key
except ImportError:
    # Fallback for standalone usage
    def get_api_key(): return ""
    def get_anthropic_api_key(): return os.environ.get("ANTHROPIC_API_KEY", "")
    def get_google_api_key(): return os.environ.get("GOOGLE_API_KEY", "")

_ANTHROPIC_MODEL = "claude-opus-4-6"
_GEMINI_MODEL = "gemini-2.5-flash"

_LLM_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".llm_cache.json")
_llm_cache: dict = {}

def load_cache():
    global _llm_cache
    if not _llm_cache and os.path.exists(_LLM_CACHE_FILE):
        try:
            with open(_LLM_CACHE_FILE, "r") as f:
                _llm_cache = json.loads(f.read())
        except Exception:
            _llm_cache = {}

def save_cache():
    try:
        with open(_LLM_CACHE_FILE, "w") as f:
            f.write(json.dumps(_llm_cache))
    except Exception:
        pass

def get_cache_key(prompt: str, temperature: float) -> str:
    return hashlib.sha256(f"{prompt}|{temperature}".encode()).hexdigest()[:16]

def call_llm(prompt: str, temperature: float = 0.1) -> str:
    """Unified LLM call with caching and provider fallback."""
    load_cache()
    key = get_cache_key(prompt, temperature)
    if key in _llm_cache:
        return _llm_cache[key]

    # Try Google Gemini first
    google_key = get_google_api_key()
    if google_key:
        try:
            from google import genai
            client = genai.Client(api_key=google_key)
            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=2000,
                ),
            )
            result = response.text
            _llm_cache[key] = result
            save_cache()
            return result
        except Exception as e:
            print(f"  ⚠️ Gemini failed: {str(e)[:50]}")

    # Fall back to Anthropic
    anthropic_key = get_anthropic_api_key()
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            response = client.messages.create(
                model=_ANTHROPIC_MODEL,
                max_tokens=2000,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text
            _llm_cache[key] = result
            save_cache()
            return result
        except Exception as e:
            print(f"  ⚠️ Anthropic failed: {str(e)[:50]}")

    return ""
