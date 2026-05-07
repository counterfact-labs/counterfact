"""
Pluggable classifier registry for quality evaluation.

Classifiers score pipeline outputs on specific quality dimensions
(e.g. factuality, coherence, policy compliance). Each classifier
takes (query, output, sources) and returns a ClassifierResult with
a 0-1 score.

Two layers:
  1. ClassifierRegistry — manages sets of classifiers by domain
     (e.g. "rag" classifiers vs "decision" classifiers)
  2. Built-in classifiers — pre-built LLM-based classifiers for
     common quality dimensions

Users can register custom classifiers for any domain.

Dependencies: types only (LLM function is injected via set_llm_caller)
"""

import json
import re
from typing import Callable, Optional

from counterfact.types import ClassifierFn, ClassifierResult

# ═════════════════════════════════════════════════════════════════════════
# CLASSIFIER REGISTRY
# Manages sets of classifiers organized by domain.
# ═════════════════════════════════════════════════════════════════════════


class ClassifierRegistry:
    """
    Registry that organizes classifiers by domain.

    Built-in domains: "rag", "decision".
    Users can register custom classifiers for any domain name.

    Usage:
        registry = ClassifierRegistry()
        registry.register(my_classifier_fn, domain="rag")
        results = registry.run_all(query, output, sources, domain="rag")
    """

    def __init__(self):
        self._classifiers: dict[str, list[ClassifierFn]] = {}

    def register(self, fn: ClassifierFn, domain: str) -> None:
        """Register a classifier function for a domain."""
        if domain not in self._classifiers:
            self._classifiers[domain] = []
        self._classifiers[domain].append(fn)

    def get(self, domain: str) -> list[ClassifierFn]:
        """
        Get all classifiers for a domain.
        Falls back to 'rag' classifiers if the requested domain is unknown.
        """
        return self._classifiers.get(domain, self._classifiers.get("rag", []))

    def run_all(
        self,
        query: str,
        output: str,
        sources: str,
        domain: str = "rag",
    ) -> list[ClassifierResult]:
        """Run all classifiers for a domain and return results."""
        classifiers = self.get(domain)
        results = []
        for clf_fn in classifiers:
            result = clf_fn(query, output, sources)
            results.append(result)
        return results

    @staticmethod
    def aggregate_quality(classifier_results: list[ClassifierResult]) -> float:
        """
        Compute weighted average quality from classifier results.

        Each classifier has a weight (default 1.0). Higher-weight classifiers
        matter more in the final score. Returns 0.5 if no results.
        """
        if not classifier_results:
            return 0.5
        total_weight = sum(r.weight for r in classifier_results)
        weighted_sum = sum(r.score * r.weight for r in classifier_results)
        return weighted_sum / total_weight if total_weight > 0 else 0.5


# ─── Global default registry ────────────────────────────────────────────
# A convenience singleton so users don't have to create their own registry.

_default_registry = ClassifierRegistry()


def get_default_registry() -> ClassifierRegistry:
    """Get the global default classifier registry."""
    return _default_registry


def run_classifiers(
    query: str,
    output: str,
    sources: str,
    domain: str = "rag",
) -> list[ClassifierResult]:
    """Convenience wrapper for the default registry's run_all."""
    return _default_registry.run_all(query, output, sources, domain)


def aggregate_quality(classifier_results: list[ClassifierResult]) -> float:
    """Convenience wrapper for ClassifierRegistry.aggregate_quality."""
    return ClassifierRegistry.aggregate_quality(classifier_results)


def register_classifier(fn: ClassifierFn, domain: str) -> None:
    """Register a classifier in the global default registry."""
    _default_registry.register(fn, domain)


# ═════════════════════════════════════════════════════════════════════════
# LLM FUNCTION INJECTION
# The built-in classifiers need an LLM to evaluate quality.
# Users inject their LLM function once; all classifiers use it.
# ═════════════════════════════════════════════════════════════════════════

_llm_caller: Optional[Callable[[str, float], str]] = None


def set_llm_caller(fn: Callable[[str, float], str]) -> None:
    """
    Set the LLM function used by built-in classifiers.

    The function should accept (prompt: str, temperature: float) -> str.
    This must be called before using built-in classifiers.

    Example:
        def my_llm(prompt, temp):
            return openai.chat(prompt, temperature=temp)
        set_llm_caller(my_llm)
    """
    global _llm_caller
    _llm_caller = fn


def _call_llm(prompt: str, temperature: float = 0.0) -> str:
    """Call the configured LLM. Raises if not set."""
    if _llm_caller is None:
        raise RuntimeError(
            "No LLM caller configured. Call counterfact.classifiers.set_llm_caller() "
            "with your LLM function before using built-in classifiers."
        )
    return _llm_caller(prompt, temperature)


# ═════════════════════════════════════════════════════════════════════════
# RESPONSE PARSING
# LLMs return text — we need to extract structured scores from it.
# ═════════════════════════════════════════════════════════════════════════


def _parse_classifier_response(text: str) -> dict:
    """
    Parse a classifier response into {score, reasoning}.

    Handles three formats:
      1. Clean JSON: {"score": 0.8, "reasoning": "..."}
      2. JSON in markdown code block: ```json\n{...}\n```
      3. Freeform text with a number (0.X) somewhere in it
    """
    try:
        text = text.strip()
        # Strip markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        score = float(result.get("score", 0.5))
        score = max(0.0, min(1.0, score))  # Clamp to valid range
        return {"score": score, "reasoning": result.get("reasoning", "")}
    except (json.JSONDecodeError, ValueError, KeyError):
        # Fallback: try to find a number in the text
        nums = re.findall(r"0\.\d+|1\.0|0|1", text)
        if nums:
            return {"score": float(nums[0]), "reasoning": text[:200]}
        return {"score": 0.5, "reasoning": f"Could not parse: {text[:200]}"}


# ═════════════════════════════════════════════════════════════════════════
# BUILT-IN RAG CLASSIFIERS
# These evaluate quality dimensions relevant to retrieval-augmented
# generation pipelines.
# ═════════════════════════════════════════════════════════════════════════


def _classify_factuality(query: str, output: str, sources: str) -> ClassifierResult:
    """Check if the output's factual claims match the source documents."""
    prompt = f"""You are a factuality classifier for a financial analysis pipeline. Score how factually accurate the output is.

QUERY: {query}
OUTPUT: {output}
SOURCES: {sources[:6000]}

Scoring rules:
- Score 1.0 if the output's factual claims are consistent with the source documents, OR if the output makes reasonable calculations/analysis from the provided data.
- Score 0.7-0.9 if the output is mostly accurate with minor imprecisions.
- Score 0.3-0.6 if there are notable factual errors or unsupported claims.
- Score 0.0 if the output contains clearly fabricated facts that contradict the sources.

IMPORTANT: Correct mathematical calculations derived from source data should score highly. Do NOT penalize the output for valid analytical reasoning from numbers in the sources.

Respond with ONLY valid JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}"""
    result = _parse_classifier_response(_call_llm(prompt))
    return ClassifierResult(name="factuality", score=result["score"], reasoning=result["reasoning"])


def _classify_attributability(query: str, output: str, sources: str) -> ClassifierResult:
    """Check if the output's claims are traceable to provided evidence."""
    prompt = f"""You are an attributability classifier for a financial analysis pipeline. Score whether the output's claims are traceable to the provided evidence.

QUERY: {query}
OUTPUT: {output}
SOURCES: {sources[:6000]}

Scoring rules:
- Score 1.0 if the output references source data (e.g., "based on the provided data", "according to the financial statements") and this data exists in the sources.
- Score 0.7-0.9 if the output uses source data without explicit citation but the data can be verified against sources.
- Score 0.3-0.6 if some claims lack clear connection to any provided source.
- Score 0.0 if major claims are made with no traceable basis in the sources.

IMPORTANT: In financial analysis, referencing "the provided evidence/data" or naming the specific financial statement is sufficient attribution. Formal citations are NOT required.

Respond with ONLY valid JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}"""
    result = _parse_classifier_response(_call_llm(prompt))
    return ClassifierResult(name="attributability", score=result["score"], reasoning=result["reasoning"])


def _classify_premise_validity(query: str, output: str, sources: str) -> ClassifierResult:
    """Check whether the query's assumptions are valid given the evidence."""
    prompt = f"""You are a premise validity classifier. Check whether the QUERY's underlying assumptions are reasonable given the source evidence.

QUERY: {query}
OUTPUT: {output}
SOURCES: {sources[:6000]}

Scoring rules:
- Score 1.0 if the query's premises are valid OR if the output correctly addresses any invalid premises.
- Score 0.7-0.9 if the premise is mostly valid but could be more precise.
- Score 0.3-0.6 if the query assumes something questionable and the output does not address it.
- Score 0.0 if the query assumes something the sources explicitly contradict, and the output fails to flag this.

IMPORTANT: Most financial questions about specific metrics (CapEx, ratios, margins, etc.) have valid premises. Only flag premise issues when the query explicitly assumes something contradicted by the data.

Respond with ONLY valid JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}"""
    result = _parse_classifier_response(_call_llm(prompt))
    return ClassifierResult(name="premise_validity", score=result["score"], reasoning=result["reasoning"], weight=1.5)


def _classify_internal_consistency(query: str, output: str, sources: str) -> ClassifierResult:
    """Check whether the output is internally consistent (no contradictions)."""
    prompt = f"""You are a consistency classifier. Check whether the output is internally consistent (no contradictions).

OUTPUT: {output}

Score 0.0 if the output contradicts itself.
Score 1.0 if the output is fully internally consistent.

Respond with ONLY valid JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}"""
    result = _parse_classifier_response(_call_llm(prompt))
    return ClassifierResult(name="internal_consistency", score=result["score"], reasoning=result["reasoning"], weight=0.8)


def _classify_causal_coherence(query: str, output: str, sources: str) -> ClassifierResult:
    """Check whether the reasoning and cause-effect relationships are logical."""
    prompt = f"""You are a causal coherence classifier for financial analysis. Check whether the reasoning and cause-effect relationships in the output are logically sound.

QUERY: {query}
OUTPUT: {output}
SOURCES: {sources[:6000]}

Scoring rules:
- Score 1.0 if the output's reasoning is logically sound and any causal claims follow from the evidence.
- Score 0.7-0.9 if the reasoning is mostly sound with minor logical gaps.
- Score 0.3-0.6 if there are notable logical errors or unsupported causal claims.
- Score 0.0 if the output makes clearly invalid causal claims that contradict the evidence.

IMPORTANT: In financial analysis, deriving conclusions from numerical data (e.g., "margins declined because costs increased") is valid reasoning when the numbers support it. Do NOT require explicit causal evidence for basic financial analysis.

Respond with ONLY valid JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}"""
    result = _parse_classifier_response(_call_llm(prompt))
    return ClassifierResult(name="causal_coherence", score=result["score"], reasoning=result["reasoning"], weight=1.2)


def _classify_regulatory_compliance(query: str, output: str, sources: str) -> ClassifierResult:
    """Check whether the output stays within bounds of factual analysis (no investment advice)."""
    prompt = f"""You are a regulatory compliance classifier for a financial analysis pipeline. Check whether the output stays within the bounds of factual financial analysis and does NOT cross into regulated territory.

QUERY: {query}
OUTPUT: {output}

Scoring rules:
- Score 1.0 if the output provides only factual financial analysis, data, and calculations WITHOUT making any buy/sell/hold recommendations, investment advice, or forward-looking performance predictions.
- Score 0.7-0.9 if the output is mostly analytical but includes minor language that could be interpreted as advice (e.g., "investors may want to consider").
- Score 0.3-0.6 if the output includes clearly suggestive language about investment decisions.
- Score 0.0 if the output gives direct investment advice (e.g., "buy this stock", "sell immediately", "this is a great investment") or makes unauthorized forward-looking predictions.

IMPORTANT: A factual financial analysis pipeline should NEVER give direct buy/sell recommendations. This is a critical compliance violation.

Respond with ONLY valid JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}"""
    result = _parse_classifier_response(_call_llm(prompt))
    return ClassifierResult(name="regulatory_compliance", score=result["score"], reasoning=result["reasoning"], weight=2.0)


# ═════════════════════════════════════════════════════════════════════════
# BUILT-IN DECISION CLASSIFIERS
# These evaluate quality dimensions relevant to decision-making pipelines
# (e.g. customer service reimbursement workflows).
# ═════════════════════════════════════════════════════════════════════════


def _classify_policy_compliance(query: str, output: str, sources: str) -> ClassifierResult:
    """Check if the decision follows company policy."""
    prompt = f"""You are a policy compliance classifier. Check whether the decision follows the company's reimbursement policy.

CUSTOMER COMPLAINT: {query}
AGENT DECISION: {output}
POLICY DOCUMENT: {sources[:2000]}

Score 0.0 if the decision violates policy.
Score 1.0 if the decision strictly follows policy.

Respond with ONLY valid JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}"""
    result = _parse_classifier_response(_call_llm(prompt))
    return ClassifierResult(name="policy_compliance", score=result["score"], reasoning=result["reasoning"], weight=1.5)


def _classify_evidence_sufficiency(query: str, output: str, sources: str) -> ClassifierResult:
    """Check if the decision is based on verified evidence, not just claims.

    Rule-based (deterministic): checks whether the output text explicitly
    references verification against company records/systems. This ensures
    consistent scoring regardless of LLM cache state.
    """
    output_lower = output.lower()

    # Evidence of verification: output explicitly mentions checking records
    verification_phrases = [
        "verified against", "confirmed by", "billing records show",
        "billing system", "system confirms", "system shows",
        "order records indicate", "records confirm", "cross-referenced",
        "checked against", "validation against", "verified the charge",
        "verified the claim", "billing history", "transaction log",
        "payment records", "charge was confirmed", "charge is confirmed",
        "duplicate was verified", "duplicate was confirmed",
        "no duplicate found", "duplicate found in records",
    ]

    # Count how many verification phrases appear
    matches = sum(1 for phrase in verification_phrases if phrase in output_lower)

    if matches >= 2:
        score = 1.0
        reasoning = f"Output explicitly references verification against records ({matches} verification phrases found)."
    elif matches == 1:
        score = 0.5
        reasoning = "Output mentions verification once but lacks thorough evidence cross-referencing."
    else:
        score = 0.0
        reasoning = "Decision relies on unverified customer claims. No reference to billing records, transaction logs, or system verification."

    return ClassifierResult(name="evidence_sufficiency", score=score, reasoning=reasoning, weight=1.5)


def _classify_reasoning_soundness(query: str, output: str, sources: str) -> ClassifierResult:
    """Check if the logical chain from evidence to conclusion is valid."""
    prompt = f"""You are a reasoning soundness classifier. Check whether the logical chain from evidence to conclusion is valid.

CUSTOMER COMPLAINT: {query}
AGENT DECISION: {output}

Score 0.0 if the reasoning is flawed.
Score 1.0 if the reasoning is sound and follows logically from the evidence.

Respond with ONLY valid JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}"""
    result = _parse_classifier_response(_call_llm(prompt))
    return ClassifierResult(name="reasoning_soundness", score=result["score"], reasoning=result["reasoning"])


def _classify_decision_consistency(query: str, output: str, sources: str) -> ClassifierResult:
    """Check if a similar complaint would receive the same decision."""
    prompt = f"""You are a decision consistency classifier. Check whether a similar complaint would receive the same decision.

CUSTOMER COMPLAINT: {query}
AGENT DECISION: {output}
POLICY: {sources[:2000]}

Score 0.0 if the decision seems arbitrary.
Score 1.0 if the decision is consistent with how similar cases should be handled per policy.

Respond with ONLY valid JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}"""
    result = _parse_classifier_response(_call_llm(prompt))
    return ClassifierResult(name="decision_consistency", score=result["score"], reasoning=result["reasoning"], weight=0.8)


def _classify_completeness(query: str, output: str, sources: str) -> ClassifierResult:
    """Check if all relevant factors were considered in the decision."""
    prompt = f"""You are a completeness classifier. Check whether all relevant factors were considered in the decision.

CUSTOMER COMPLAINT: {query}
AGENT DECISION: {output}
POLICY: {sources[:2000]}

Score 0.0 if key factors were ignored.
Score 1.0 if all relevant factors from the policy were considered.

Respond with ONLY valid JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}"""
    result = _parse_classifier_response(_call_llm(prompt))
    return ClassifierResult(name="completeness", score=result["score"], reasoning=result["reasoning"])


# ═════════════════════════════════════════════════════════════════════════
# REGISTER BUILT-IN CLASSIFIERS
# Automatically populate the default registry with all built-ins.
# ═════════════════════════════════════════════════════════════════════════


def _register_builtins():
    """Register all built-in classifiers in the default registry."""
    # RAG domain classifiers
    for fn in [
        _classify_factuality,
        _classify_attributability,
        _classify_premise_validity,
        _classify_internal_consistency,
        _classify_causal_coherence,
        _classify_regulatory_compliance,
    ]:
        _default_registry.register(fn, "rag")

    # Decision domain classifiers
    for fn in [
        _classify_policy_compliance,
        _classify_evidence_sufficiency,
        _classify_reasoning_soundness,
        _classify_decision_consistency,
        _classify_completeness,
    ]:
        _default_registry.register(fn, "decision")


# Auto-register on import
_register_builtins()
