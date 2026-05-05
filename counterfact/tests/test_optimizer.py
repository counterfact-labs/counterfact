import random
from unittest.mock import patch
from counterfact.optimizer import SearchSpace, optimize_pipeline

def test_search_space():
    space = SearchSpace()
    space.add_categorical("model", ["a", "b"])
    space.add_float("temp", 0.0, 1.0)
    space.add_int("k", 1, 10)
    
    rng = random.Random(42)
    config = space.sample_random(rng)
    
    assert config["model"] in ["a", "b"]
    assert 0.0 <= config["temp"] <= 1.0
    assert 1 <= config["k"] <= 10

@patch("counterfact.optimizer._try_vizier", return_value=None)
def test_optimize_pipeline_random(mock_try_vizier):
    space = SearchSpace()
    space.add_categorical("model", ["a"])
    
    def evaluate(config):
        return 0.8
        
    res = optimize_pipeline(evaluate, space, num_trials=5, seed=42)
    
    assert res.search_method == "random_search"
    assert res.num_trials == 5
    assert res.best_quality == 0.8
    assert res.best_config == {"model": "a"}

@patch("counterfact.optimizer._try_vizier")
def test_optimize_pipeline_vizier(mock_try_vizier):
    space = SearchSpace()
    space.add_categorical("model", ["a", "b"])
    space.add_float("temp", 0.0, 1.0)
    space.add_int("k", 1, 10)
    
    from counterfact.optimizer import TrialResult
    mock_try_vizier.return_value = [
        TrialResult(trial_id=0, config={"model": "a", "temp": 0.5, "k": 5}, quality=0.9)
    ]
    
    def evaluate(config):
        return 0.9
        
    res = optimize_pipeline(evaluate, space, num_trials=2)
    
    assert res.search_method == "vizier"
    assert res.best_quality == 0.9

@patch("counterfact.optimizer._try_vizier", return_value=None)
def test_optimize_pipeline_vizier_fallback(mock_vizier):
    space = SearchSpace()
    space.add_categorical("model", ["a"])
    
    def evaluate(config):
        return 0.7
        
    def progress(step, total, config):
        pass
        
    res = optimize_pipeline(evaluate, space, num_trials=2, progress_callback=progress)
    assert res.search_method == "random_search"
    
    # Test methods
    assert len(space) == 1
    assert len(res.all_trials[0].to_dict()) > 0
    assert len(res.to_dict()) > 0
