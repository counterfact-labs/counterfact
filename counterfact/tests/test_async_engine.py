import pytest
from counterfact.async_engine import run_monte_carlo_async, run_full_diagnostic_async

@pytest.mark.asyncio
async def test_run_monte_carlo_async():
    # We need a mock llm_fn_async and a mock registry
    class MockRegistry:
        def run_all(self, *args, **kwargs):
            from counterfact.types import ClassifierResult
            return [ClassifierResult(name="factuality", score=0.8, reasoning="")]
            
        @staticmethod
        def aggregate_quality(results):
            return 0.8
            
    async def mock_llm_async(prompt, temp):
        return "mocked async output"
        
    trace = [
        {"node": "retriever", "output": "doc"},
        {"node": "synthesizer", "output": "text"}
    ]
    
    # Run async monte carlo
    results = await run_monte_carlo_async(
        trace=trace,
        query="test query",
        output_text="test output",
        sources="test sources",
        num_simulations=5,  # 3 baseline + 2 async ablate
        registry=MockRegistry(),
        llm_fn_async=mock_llm_async,
        seed=42,
        max_concurrent_sims=2
    )
    
    assert len(results) == 5
    # The first 3 should be baseline
    assert sum(1 for r in results if r.is_baseline) == 3
    # The next 2 should be perturbations
    assert sum(1 for r in results if not r.is_baseline) == 2

@pytest.mark.asyncio
async def test_run_full_diagnostic_async():
    class MockRegistry:
        def run_all(self, *args, **kwargs):
            from counterfact.types import ClassifierResult
            return [ClassifierResult(name="factuality", score=0.5, reasoning="")]
            
        @staticmethod
        def aggregate_quality(results):
            return 0.5
            
    async def mock_llm_async(prompt, temp):
        return "mocked async response"
        
    def mock_llm_sync(prompt, temp):
        return '[{"title": "fix", "description": "desc", "intervention_type": "add_agent", "target_agent": null, "estimated_failure_reduction": 0.5, "complexity": "low"}]'

    trace = [
        {"node": "retriever", "output": "doc"},
        {"node": "synthesizer", "output": "text"}
    ]
    
    report = await run_full_diagnostic_async(
        trace=trace,
        query="test",
        output_text="out",
        num_simulations=5,
        registry=MockRegistry(),
        llm_fn_async=mock_llm_async,
        max_concurrent_sims=2,
        seed=42
    )
    
    assert report.num_simulations == 5
    assert round(report.baseline_quality, 2) == 0.5
    assert report.classification is not None

@pytest.mark.asyncio
async def test_run_full_diagnostic_async_quality_gate():
    class MockRegistry:
        def run_all(self, *args, **kwargs):
            from counterfact.types import ClassifierResult
            return [ClassifierResult(name="factuality", score=0.9, reasoning="")]
            
        @staticmethod
        def aggregate_quality(results):
            return 0.9
            
    async def mock_llm_async(prompt, temp):
        return "mocked"
        
    trace = [{"node": "a", "output": "doc"}]
    
    report = await run_full_diagnostic_async(
        trace=trace,
        query="test",
        output_text="out",
        num_simulations=2,
        registry=MockRegistry(),
        llm_fn_async=mock_llm_async,
        quality_gate=0.8,
        seed=42
    )
    
    assert report.classification.failure_type == "no_failure"
