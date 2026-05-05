import pytest
import os
import json
from unittest.mock import patch, MagicMock, mock_open
import counterfact
import counterfact.cli
from counterfact.cli import _load_trace, _resolve_llm_fn, main

def test_load_trace_list(tmp_path):
    trace_data = [{"node": "a", "output": "b"}]
    file_path = tmp_path / "trace.json"
    file_path.write_text(json.dumps(trace_data))
    
    res = _load_trace(str(file_path), MagicMock())
    assert res == {"trace": trace_data}

def test_load_trace_dict(tmp_path):
    trace_data = {"trace": [{"node": "a"}], "query": "q"}
    file_path = tmp_path / "trace.json"
    file_path.write_text(json.dumps(trace_data))
    
    res = _load_trace(str(file_path), MagicMock())
    assert res == trace_data

@patch("sys.exit", side_effect=SystemExit)
def test_load_trace_invalid(mock_exit, tmp_path):
    file_path = tmp_path / "trace.json"
    file_path.write_text("invalid json")
    
    with pytest.raises(SystemExit):
        _load_trace(str(file_path), MagicMock())
    mock_exit.assert_called_with(1)

@patch("sys.exit", side_effect=SystemExit)
def test_load_trace_not_found(mock_exit):
    with pytest.raises(SystemExit):
        _load_trace("does_not_exist.json", MagicMock())
    mock_exit.assert_called_with(1)

@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test", "GOOGLE_API_KEY": ""})
@patch("counterfact.cli._make_anthropic_caller", return_value="anthropic_caller")
def test_resolve_llm_fn_anthropic(mock_anthropic):
    res = _resolve_llm_fn("anthropic", MagicMock())
    assert res == "anthropic_caller"

@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "GOOGLE_API_KEY": "test"})
@patch("counterfact.cli._make_google_caller", return_value="google_caller")
def test_resolve_llm_fn_google(mock_google):
    res = _resolve_llm_fn("google", MagicMock())
    assert res == "google_caller"

@patch("sys.exit", side_effect=SystemExit)
def test_resolve_llm_fn_unknown(mock_exit):
    with pytest.raises(SystemExit):
        _resolve_llm_fn("unknown", MagicMock())
    mock_exit.assert_called_with(1)

@patch("argparse.ArgumentParser.print_help")
@patch("counterfact.cli._load_trace")
def test_main_eval_no_args(mock_load, mock_print_help):
    with patch("sys.argv", ["counterfact"]):
        main()
        mock_print_help.assert_called()

@patch("counterfact.cli.Console")
def test_make_anthropic_caller(mock_console):
    import sys
    from unittest.mock import MagicMock
    mock_anthropic = MagicMock()
    
    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        caller = counterfact.cli._make_anthropic_caller("key", mock_console())
        res = caller("test prompt", 0.5)
        assert res is not None

@patch("counterfact.cli.Console")
def test_make_google_caller(mock_console):
    import sys
    from unittest.mock import MagicMock
    mock_genai = MagicMock()
    mock_google = MagicMock()
    mock_google.genai = mock_genai
    
    with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
        caller = counterfact.cli._make_google_caller("key", mock_console())
        res = caller("test prompt", 0.5)
        assert res is not None

@patch("counterfact.cli.run_eval")
@patch("counterfact.cli.run_discover")
def test_main_commands(mock_discover, mock_eval):
    with patch("sys.argv", ["counterfact", "eval", "trace.json"]):
        main()
        mock_eval.assert_called()
        
    with patch("sys.argv", ["counterfact", "discover", "logs.txt"]):
        main()
        mock_discover.assert_called()

@patch("counterfact.cli._load_trace")
@patch("counterfact.cli.Console")
@patch("counterfact.evals.run_eval_suite")
def test_run_eval(mock_eval_suite, mock_console, mock_load):
    mock_load.return_value = {"trace": [{"node": "agent1", "output": "x"}], "output": "y"}
    from unittest.mock import MagicMock
    mock_args = MagicMock()
    mock_args.trace_file = "test.json"
    mock_args.provider = None
    
    mock_suite = MagicMock()
    mock_result = MagicMock()
    mock_result.passed = True
    mock_result.check_name = "test_check"
    mock_result.severity = "high"
    mock_result.message = "msg"
    mock_suite.results = [mock_result]
    mock_eval_suite.return_value = mock_suite
    
    counterfact.cli.run_eval(mock_args)
    mock_eval_suite.assert_called()

@patch("counterfact.cli._load_trace")
@patch("counterfact.cli.Console")
@patch("counterfact.discovery.discover_pipeline")
def test_run_discover(mock_discover, mock_console, mock_load):
    mock_load.return_value = {"trace": [{"node": "agent1", "output": "x"}], "output": "y"}
    from unittest.mock import MagicMock
    mock_args = MagicMock()
    mock_args.log_file = "logs.json"
    
    mock_plan = MagicMock()
    mock_plan.pipeline_description = "desc"
    mock_plan.domain = "rag"
    mock_agent = MagicMock()
    mock_agent.name = "agent1"
    mock_agent.inferred_role = "role"
    mock_agent.description = "desc"
    mock_plan.agent_profiles = [mock_agent]
    mock_discover.return_value = mock_plan
    
    with patch("builtins.open", mock_open(read_data="logs content")):
        counterfact.cli.run_discover(mock_args)
    mock_discover.assert_called()
