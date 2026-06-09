# `DiagnosticReport` schema + how to write a custom classifier

## The JSON `cf_diagnose.py` writes (`report.to_dict()`)

```jsonc
{
  "query": "the input rendered as a query string",
  "domain": "rag",
  "baseline_quality": 0.248,                 // quality of the pipeline as-is, 0..1
  "shapley_values": { "summarizer": 0.784, "retriever": -0.112, "fact_checker": 0.104 },
  "shapley_cis": {                           // 95% bootstrap CI per agent — trust gate
    "summarizer": { "ci_low": 0.535, "ci_high": 1.032, "n_samples": 6 }
  },
  "baseline_quality_ci": { "ci_low": 0.2, "ci_high": 0.29, "n_samples": 3 },
  "attribution_method": "shapley",           // "loo" | "shapley" | "quality_gate"
  "per_classifier_shapley": {                // attribution split by quality dimension
    "source_coverage": { "retriever": -0.5, "summarizer": 0.464, "fact_checker": 0.036 }
  },
  "classification": {
    "failure_type": "local",                 // see failure-taxonomy.md
    "confidence": 0.95,
    "description": "...",
    "evidence": ["Dominant Shapley value: summarizer = 0.784", "..."],
    "dominant_agent": "summarizer",
    "failing_classifiers": ["answer_relevance", "source_coverage"],
    "damping_ratio": null,                   // set only for feedback_amplification
    "confidence_explanation": "Based on N simulations..."
  },
  "recommendations": [
    {
      "description": "...",
      "intervention_type": "restructure",    // add_agent | modify_agent | restructure | ...
      "target_agent": "retriever",
      "measurement_confidence": "measured"    // "measured" or "estimated"
    }
  ],
  "evaluations": [ /* whether the proposed fixes would help, if evaluated */ ],
  "num_simulations": 30,
  "simulation_details": [ /* per-run: ablated agents, output, classifier scores */ ],
  "simulation_results_summary": {
    "total_simulations": 9, "baseline_runs": 3, "perturbation_runs": 6,
    "agents_analyzed": ["retriever", "summarizer", "fact_checker"],
    "classifiers_used": ["completeness", "answer_relevance", "source_coverage"]
  },
  "eval_suite": { /* structural + consistency check results, if run_evals */ },
  "seed": 42
}
```

Read order for triage: `classification.failure_type` → `shapley_values` + `shapley_cis`
(apply the guardrails in `reading-attribution.md`) → `per_classifier_shapley` →
`recommendations` → `simulation_details` only if you need coalition-level evidence.

## Writing a custom classifier (the high-value move on non-RAG pipelines)

A classifier scores one quality dimension of the output. Signature:

```python
from counterfact.types import ClassifierResult

def answers_the_number(query: str, output: str, sources: str) -> ClassifierResult:
    """Did the output contain the specific figure the query asked for?"""
    import re
    has_number = bool(re.search(r"\$?\d[\d,]*(\.\d+)?", output))
    return ClassifierResult(
        name="answers_the_number",
        score=1.0 if has_number else 0.1,   # 0..1; higher = better
        reasoning="contains a numeric value" if has_number else "no number found",
        weight=1.5,                          # optional; up-weight the dimension that matters
    )
```

LLM-backed classifiers (grounding, relevance) take the same signature and call the LLM
inside — see `examples/quickstart.py` (`make_llm_classifiers`) for grounding/relevance
templates that return JSON `{"score", "reasoning"}`.

Register them and pass via `--registry`:

```python
# myapp/cf_classifiers.py
from counterfact.classifiers import ClassifierRegistry

def build_registry() -> ClassifierRegistry:
    reg = ClassifierRegistry()
    reg.register(answers_the_number, domain="finance")
    reg.register(grounding_classifier, domain="finance")
    return reg
```
```bash
python cf_diagnose.py --factory myapp.pipeline:build --inputs cases.json \
    --registry myapp.cf_classifiers:build_registry --domain finance
```

Why bother: attribution is only as sharp as the quality metric. A classifier that actually
measures the failure the user cares about turns vague Shapley values into a decisive culprit.

## The factory contract (recap)

```python
# myapp/pipeline.py
from counterfact import StateGraph, END

def build():
    g = StateGraph(MyState)
    g.add_node("retriever", retriever)
    g.add_node("synthesizer", synthesizer)
    g.add_edge("retriever", "synthesizer")
    g.add_edge("synthesizer", END)
    g.set_entry_point("retriever")
    return g.compile()        # <- MUST return the COMPILED graph
```
`build()` takes no required args and returns the compiled graph. If your real builder needs
config, wrap it: `def build(): return build_pipeline(load_default_config())`.
