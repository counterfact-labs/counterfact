import json
import os
from unittest.mock import MagicMock, patch

import pytest

from counterfact.llm import call_llm, get_cache_key, load_cache, save_cache


@pytest.fixture(autouse=True)
def clean_cache():
    from counterfact import llm
    llm._llm_cache = {}
    yield
    llm._llm_cache = {}

def test_get_cache_key():
    key1 = get_cache_key("test prompt", 0.5)
    key2 = get_cache_key("test prompt", 0.5)
    key3 = get_cache_key("test prompt", 0.1)

    assert key1 == key2
    assert key1 != key3

def test_call_llm_cached():
    key = get_cache_key("test prompt", 0.5)
    from counterfact import llm
    llm._llm_cache[key] = "cached response"

    with patch("counterfact.llm.load_cache"):
        res = call_llm("test prompt", 0.5)

    assert res == "cached response"

def test_load_cache(tmp_path):
    cache_data = {"key": "value"}
    cache_file = tmp_path / ".llm_cache.json"
    cache_file.write_text(json.dumps(cache_data))

    with patch("counterfact.llm._LLM_CACHE_FILE", str(cache_file)):
        from counterfact import llm
        llm._llm_cache = {}
        load_cache()
        assert llm._llm_cache == {"key": "value"}

def test_save_cache(tmp_path):
    cache_file = tmp_path / ".llm_cache.json"

    with patch("counterfact.llm._LLM_CACHE_FILE", str(cache_file)):
        from counterfact import llm
        llm._llm_cache = {"key": "value"}
        save_cache()

    assert json.loads(cache_file.read_text()) == {"key": "value"}

@patch.dict(os.environ, {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": ""})
def test_call_llm_no_keys():
    res = call_llm("test", 0.1)
    assert res == ""

@patch.dict(os.environ, {"GOOGLE_API_KEY": "google_key", "GEMINI_API_KEY": ""})
def test_call_llm_google_success():
    import sys
    mock_google = MagicMock()
    mock_genai = MagicMock()
    mock_google.genai = mock_genai
    mock_client = mock_genai.Client.return_value
    class MockResponse:
        text = "google output"
    mock_client.models.generate_content.return_value = MockResponse()

    with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
        with patch("counterfact.llm.save_cache"):
            res = call_llm("test google", 0.1)
    assert res == "google output"

@patch.dict(os.environ, {"GOOGLE_API_KEY": "google_key", "GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": ""})
def test_call_llm_google_failure():
    import sys
    mock_google = MagicMock()
    mock_genai = MagicMock()
    mock_google.genai = mock_genai
    mock_client = mock_genai.Client.return_value
    mock_client.models.generate_content.side_effect = Exception("API Error")

    with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
        res = call_llm("test google fail", 0.1)
    assert res == ""

@patch.dict(os.environ, {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": "anthropic_key"})
def test_call_llm_anthropic_success():
    import sys
    mock_anthropic = MagicMock()
    mock_client = mock_anthropic.Anthropic.return_value
    mock_response = mock_client.messages.create.return_value

    mock_block = MagicMock()
    mock_block.text = "anthropic output"
    mock_response.content = [mock_block]

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        with patch("counterfact.llm.save_cache"):
            res = call_llm("test anthropic", 0.1)
    assert res == "anthropic output"

@patch.dict(os.environ, {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": "anthropic_key"})
def test_call_llm_anthropic_failure():
    import sys
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.side_effect = Exception("API Error")

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        res = call_llm("test anthropic fail", 0.1)
    assert res == ""
