"""Graded perturbation (degradation) analysis.

Plain ablation — replacing a node with a no-op — answers "is this agent
load-bearing?" For some modules that question is uninformative. Ablate a
retriever and the pipeline gets no context at all, so it structurally collapses;
you learn the retriever is necessary, not whether *retrieval quality* is what's
hurting your answers. The interesting question for a retriever, reranker, or
parser is the dose-response one: as the module's output gets worse, how much
does final quality fall?

This module answers that by **graded degradation**. Ablation is treated as one
end of a spectrum: a degrader takes a node's output and a magnitude in [0, 1] and
returns a progressively worse version, where ``magnitude=1.0`` is "as bad as
ablation" (drop everything) and ``0.0`` is unchanged. For each node we sweep
several magnitudes, re-run the real pipeline, score each run, and read the curve:

  * **quality_driver** — quality falls smoothly as the module degrades. The
    module's output *quality* drives the answer; improving it should help.
  * **structural** — quality is flat under partial degradation but collapses only
    at full removal (or the pipeline errors). The module is required to run, but
    ablation is the blunt, uninformative signal — look elsewhere for quality.
  * **harmful** — degrading or removing the module *improves* quality. It is
    actively hurting the answer.
  * **robust** — quality barely moves even at full degradation. Low impact.

So the developer (or the skill) never has to choose "ablate vs degrade" up
front: the sweep subsumes ablation, and the classification reads the right
interpretation off the curve. A small library of degraders, auto-selected by an
inferred module type, makes this work out of the box; a per-node override hook
lets you supply domain-specific degraders (e.g. inject hard distractors).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from counterfact.graph import CounterfactualGraph

# A degrader maps (value, magnitude, rng) -> a degraded value. magnitude is in
# [0, 1]; 0 returns the value unchanged, 1 is maximally degraded (~ablation).
Degrader = Callable[[Any, float, random.Random], Any]

_EPS = 1e-9


# ═════════════════════════════════════════════════════════════════════════
# DEGRADER LIBRARY
# Each factory returns a Degrader(value, magnitude, rng) -> degraded value.
# ═════════════════════════════════════════════════════════════════════════


def drop_items() -> Degrader:
    """Keep only a ``(1 - magnitude)`` fraction of a list (e.g. retrieved docs).

    magnitude=1 drops everything (ablation-equivalent); magnitude=0 keeps all.
    Items are assumed ranked best-first, so we keep the prefix.
    """

    def _degrade(value: Any, magnitude: float, rng: random.Random) -> Any:
        if not isinstance(value, (list, tuple)):
            return value
        keep = max(0, round(len(value) * (1.0 - magnitude)))
        return type(value)(value[:keep])

    _degrade.__name__ = "drop_items"
    return _degrade


def shuffle_relevance() -> Degrader:
    """Decay ranking quality: move good items toward the back (reranker failure).

    At magnitude=1 the list is fully reversed (worst-first); partial magnitudes
    swap a proportional number of elements from the front toward the back.
    """

    def _degrade(value: Any, magnitude: float, rng: random.Random) -> Any:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return value
        items = list(value)
        # Swap a proportional number of front/back pairs; at magnitude 1 this is
        # a full reversal (best items pushed to the back).
        pairs = len(items) // 2
        k = round(pairs * magnitude)
        for i in range(k):
            items[i], items[len(items) - 1 - i] = items[len(items) - 1 - i], items[i]
        return type(value)(items)

    _degrade.__name__ = "shuffle_relevance"
    return _degrade


def inject_distractors(distractor: str = "Irrelevant filler passage unrelated to the question.") -> Degrader:
    """Replace a ``magnitude`` fraction of list items with a distractor string.

    Models a retriever pulling in off-topic context without changing the count
    (so the failure is quality, not quantity).
    """

    def _degrade(value: Any, magnitude: float, rng: random.Random) -> Any:
        if not isinstance(value, (list, tuple)) or not value:
            return value
        items = list(value)
        n = round(len(items) * magnitude)
        idxs = rng.sample(range(len(items)), min(n, len(items)))
        for i in idxs:
            items[i] = distractor
        return type(value)(items)

    _degrade.__name__ = "inject_distractors"
    return _degrade


def truncate_text() -> Degrader:
    """Keep only the leading ``(1 - magnitude)`` fraction of a string."""

    def _degrade(value: Any, magnitude: float, rng: random.Random) -> Any:
        if not isinstance(value, str):
            return value
        keep = max(0, round(len(value) * (1.0 - magnitude)))
        return value[:keep]

    _degrade.__name__ = "truncate_text"
    return _degrade


def drop_sentences() -> Degrader:
    """Drop a ``magnitude`` fraction of sentences from a string."""

    def _degrade(value: Any, magnitude: float, rng: random.Random) -> Any:
        if not isinstance(value, str) or not value.strip():
            return value
        import re

        parts = re.split(r"(?<=[.!?])\s+", value)
        keep = max(0, round(len(parts) * (1.0 - magnitude)))
        return " ".join(parts[:keep])

    _degrade.__name__ = "drop_sentences"
    return _degrade


def drop_fields() -> Degrader:
    """Drop a ``magnitude`` fraction of a dict's keys (parser/extractor failure)."""

    def _degrade(value: Any, magnitude: float, rng: random.Random) -> Any:
        if not isinstance(value, dict) or not value:
            return value
        keys = list(value.keys())
        n_drop = round(len(keys) * magnitude)
        drop = set(rng.sample(keys, min(n_drop, len(keys))))
        return {k: v for k, v in value.items() if k not in drop}

    _degrade.__name__ = "drop_fields"
    return _degrade


# ═════════════════════════════════════════════════════════════════════════
# MODULE-TYPE INFERENCE + DEGRADER SELECTION
# ═════════════════════════════════════════════════════════════════════════

_RETRIEVER_HINTS = ("retriev", "search", "fetch", "lookup", "recall", "context", "doc")
_RERANKER_HINTS = ("rerank", "rank", "order", "sort", "score")
_PARSER_HINTS = ("pars", "extract", "structur", "classif", "route", "triage")


def infer_module_type(node_name: str, sample_value: Any) -> str:
    """Best-effort guess of a node's role from its name and a sample output.

    Returns one of ``retriever``, ``reranker``, ``parser``, ``generator``.
    Name hints win; otherwise the output shape decides (list -> retriever,
    dict -> parser, string -> generator).
    """
    name = node_name.lower()
    if any(h in name for h in _RERANKER_HINTS):
        return "reranker"
    if any(h in name for h in _RETRIEVER_HINTS):
        return "retriever"
    if any(h in name for h in _PARSER_HINTS):
        return "parser"
    if isinstance(sample_value, (list, tuple)):
        return "retriever"
    if isinstance(sample_value, dict):
        return "parser"
    return "generator"


def default_degrader_for(module_type: str, sample_value: Any) -> Degrader:
    """Pick a sensible built-in degrader for an inferred module type/value."""
    if module_type == "reranker" and isinstance(sample_value, (list, tuple)):
        return shuffle_relevance()
    if module_type == "retriever":
        if isinstance(sample_value, (list, tuple)):
            return drop_items()
        return truncate_text()
    if module_type == "parser" and isinstance(sample_value, dict):
        return drop_fields()
    # generator / fallback: degrade prose by dropping content
    if isinstance(sample_value, str):
        return drop_sentences()
    if isinstance(sample_value, (list, tuple)):
        return drop_items()
    if isinstance(sample_value, dict):
        return drop_fields()
    return truncate_text()


# ═════════════════════════════════════════════════════════════════════════
# RESULT TYPES
# ═════════════════════════════════════════════════════════════════════════


@dataclass
class NodeSensitivity:
    """Dose-response result for one node."""

    node: str
    module_type: str
    degrader: str
    target_key: Optional[str]
    baseline_quality: float
    curve: list = field(default_factory=list)        # [(magnitude, quality), ...]
    sensitivity: float = 0.0                          # signed: baseline - quality@max magnitude
    partial_sensitivity: float = 0.0                  # max quality drop before full magnitude
    classification: str = "robust"
    recommendation: str = ""
    errored_at_full: bool = False

    def to_dict(self) -> dict:
        return {
            "node": self.node,
            "module_type": self.module_type,
            "degrader": self.degrader,
            "target_key": self.target_key,
            "baseline_quality": round(self.baseline_quality, 3),
            "curve": [[round(m, 3), round(q, 3)] for m, q in self.curve],
            "sensitivity": round(self.sensitivity, 4),
            "partial_sensitivity": round(self.partial_sensitivity, 4),
            "classification": self.classification,
            "recommendation": self.recommendation,
            "errored_at_full": self.errored_at_full,
        }


@dataclass
class SensitivityReport:
    """Per-node graded-degradation results for a pipeline."""

    baseline_quality: float
    magnitudes: list
    nodes: list = field(default_factory=list)  # list[NodeSensitivity]
    seed: Optional[int] = None

    def ranked(self) -> list:
        """Nodes by absolute impact on quality, most impactful first."""
        return sorted(self.nodes, key=lambda n: abs(n.sensitivity), reverse=True)

    def most_sensitive(self) -> Optional[NodeSensitivity]:
        ranked = self.ranked()
        return ranked[0] if ranked else None

    def to_dict(self) -> dict:
        top = self.most_sensitive()
        return {
            "baseline_quality": round(self.baseline_quality, 3),
            "magnitudes": list(self.magnitudes),
            "seed": self.seed,
            "nodes": [n.to_dict() for n in self.ranked()],
            "most_sensitive": top.node if top else None,
        }

    def to_markdown(self, path: Optional[str] = None) -> str:
        lines = [
            "# Sensitivity analysis (graded degradation)",
            "",
            f"Baseline quality: **{self.baseline_quality:.3f}** · magnitudes swept: "
            f"{', '.join(str(m) for m in self.magnitudes)}",
            "",
            "| node | type | class | sensitivity | partial | degrader |",
            "|---|---|---|---|---|---|",
        ]
        for n in self.ranked():
            lines.append(
                f"| `{n.node}` | {n.module_type} | **{n.classification}** | "
                f"{n.sensitivity:+.3f} | {n.partial_sensitivity:+.3f} | `{n.degrader}` |"
            )
        lines.append("")
        for n in self.ranked():
            curve = "  ".join(f"{m:.2f}->{q:.2f}" for m, q in n.curve)
            lines.append(f"- `{n.node}` ({n.classification}): {n.recommendation}")
            lines.append(f"    curve: q@0={self.baseline_quality:.2f}  {curve}")
        md = "\n".join(lines)
        if path:
            with open(path, "w") as f:
                f.write(md)
        return md


# ═════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════

_RECO = {
    "quality_driver": (
        "Final quality is sensitive to this module's output quality even under "
        "partial degradation. Investing in this module (better retrieval/ranking/"
        "extraction/prompt) should move the needle."
    ),
    "structural": (
        "The pipeline needs this module to run, but quality is insensitive to "
        "partial degradation — only full removal hurts. Ablation is the blunt, "
        "uninformative signal here; look to other modules for quality wins."
    ),
    "harmful": (
        "Degrading or removing this module IMPROVES quality — it is actively "
        "hurting the answer. Consider fixing its instruction or removing it."
    ),
    "robust": (
        "Quality barely moves even at full degradation. Low impact on this metric."
    ),
}


def _classify(baseline_q: float, curve: list, errored_at_full: bool, structural_eps: float) -> str:
    """Classify a node from its dose-response curve."""
    if not curve:
        return "robust"
    # Signed drop relative to baseline at each magnitude.
    drops = [(m, baseline_q - q) for m, q in curve]
    max_full = max(m for m, _ in curve)
    drop_at_full = next((d for m, d in drops if abs(m - max_full) < _EPS), 0.0)
    partial = [d for m, d in drops if m < max_full - _EPS]
    partial_drop = max(partial) if partial else 0.0
    best_gain = -min((d for _, d in drops), default=0.0)  # quality increase from degrading

    if best_gain > structural_eps:
        return "harmful"
    if partial_drop >= structural_eps:
        return "quality_driver"
    if drop_at_full >= structural_eps or errored_at_full:
        return "structural"
    return "robust"


# ═════════════════════════════════════════════════════════════════════════
# NODE WRAPPING
# ═════════════════════════════════════════════════════════════════════════


def _primary_output_key(output: dict, input_state: dict, prefer: Optional[str]) -> Optional[str]:
    """The key a node 'produces': among keys it changed, the largest payload.

    Prefers an explicitly requested key; otherwise picks the changed key with the
    biggest list/string value (that is what degrading should target).
    """
    if prefer and prefer in output:
        return prefer
    changed = {k: v for k, v in output.items() if input_state.get(k) != v}
    pool = changed or output

    def _size(v: Any) -> int:
        if isinstance(v, (list, tuple, dict)):
            return len(v) * 1000
        if isinstance(v, str):
            return len(v)
        return 0

    best = max(pool.items(), key=lambda kv: _size(kv[1]), default=(None, None))
    return best[0]


def _degraded_node(original_fn: Callable, degrader: Degrader, magnitude: float,
                   rng: random.Random, target_key: Optional[str]) -> Callable:
    """Wrap a node so it runs normally, then degrades its primary output."""

    def _node(state: dict) -> dict:
        # Magnitude 1.0 is true ablation: the node contributes nothing, so its
        # output reverts to whatever upstream produced. This makes ablation the
        # literal endpoint of the degradation spectrum, and is what reveals a
        # *harmful* node (removing it lets the correct upstream value survive).
        if magnitude >= 1.0 - _EPS:
            return dict(state)
        out = original_fn(state)
        if not isinstance(out, dict):
            return out
        key = _primary_output_key(out, state, target_key)
        if key is not None and key in out:
            return {**out, key: degrader(out[key], magnitude, rng)}
        return out

    _node.__name__ = f"degraded_{getattr(original_fn, '__name__', 'node')}"
    return _node


# ═════════════════════════════════════════════════════════════════════════
# MAIN ENTRY
# ═════════════════════════════════════════════════════════════════════════


def run_degradation_analysis(
    graph: "CounterfactualGraph",
    input_state: dict,
    *,
    degraders: Optional[dict] = None,
    target_keys: Optional[dict] = None,
    magnitudes: tuple = (0.25, 0.5, 1.0),
    nodes: Optional[list] = None,
    quality_fn: Optional[Callable[[str, dict], float]] = None,
    registry=None,
    llm_fn: Optional[Callable] = None,
    domain: str = "rag",
    sources: str = "",
    seed: Optional[int] = None,
    structural_eps: float = 0.05,
    progress_callback: Optional[Callable] = None,
) -> SensitivityReport:
    """Sweep graded degradation for each node and classify its dose-response.

    For every node, the original function is run and its primary output is
    progressively degraded (``magnitudes`` from mild to ``1.0`` = ablation-
    equivalent). The real pipeline is re-executed at each magnitude and scored;
    the resulting curve is classified as quality_driver / structural / harmful /
    robust (see module docstring).

    Args:
        graph: A compiled ``CounterfactualGraph`` (must carry a build recipe).
        input_state: Input dict to invoke the pipeline with.
        degraders: Optional ``{node_name: Degrader}`` overrides. Nodes without an
            override get a built-in degrader chosen by inferred module type.
        target_keys: Optional ``{node_name: state_key}`` to force which output key
            is degraded (otherwise the node's largest changed output is used).
        magnitudes: Degradation levels to sweep, ascending; ``1.0`` is ablation-
            equivalent (drop everything).
        nodes: Restrict the analysis to these node names (default: all).
        quality_fn: ``(output_text, state) -> float`` quality metric. If omitted,
            the classifier registry aggregate is used (like ``diagnose``).
        registry / llm_fn / domain / sources: Classifier configuration, used only
            when ``quality_fn`` is not supplied.
        seed: Seed for deterministic degraders (shuffles, distractor placement).
        structural_eps: Minimum quality change considered meaningful.
        progress_callback: Optional ``callback(current, total, status)``.

    Returns:
        A :class:`SensitivityReport`.

    Raises:
        ValueError: If the graph has no build recipe (cannot clone/re-run).
    """
    recipe = getattr(graph, "_recipe", None)
    if recipe is None:
        raise ValueError(
            "run_degradation_analysis needs a build recipe. Use counterfact.StateGraph "
            "or build_graph_from_spec (raw LangGraph graphs cannot be re-run with degradation)."
        )

    from counterfact.classifiers import ClassifierRegistry, get_default_registry
    from counterfact.perturbation import _extract_final_output, _run_pipeline_safe

    reg = registry if registry is not None else get_default_registry()
    query = input_state.get("query", str(input_state)[:200])
    degraders = degraders or {}
    target_keys = target_keys or {}

    def _score(result: dict, output_text: str) -> float:
        if quality_fn is not None:
            return float(quality_fn(output_text, result if isinstance(result, dict) else {}))
        clf = reg.run_all(query, output_text, sources, domain)
        return ClassifierRegistry.aggregate_quality(clf)

    target_nodes = nodes or graph.get_node_names()

    # ── Probe run ─────────────────────────────────────────────────────
    # Capture each node's REAL input/output (the execution trace summarizes list
    # and dict payloads to strings like "list[5]", which would defeat both the
    # module-type inference and the degrader selection). We wrap each node with a
    # pass-through that records its true (input, output), then run once.
    captured: dict[str, tuple] = {}

    def _make_capture(orig: Callable, name: str) -> Callable:
        def _cap(state: dict) -> dict:
            out = orig(state)
            captured[name] = (dict(state), out if isinstance(out, dict) else {})
            return out

        _cap.__name__ = f"capture_{name}"
        return _cap

    probe = graph
    for name in target_nodes:
        orig = recipe.nodes.get(name)
        if orig is not None:
            probe = probe.clone_with_replacement(name, _make_capture(orig, name))
    base_result, _ = _run_pipeline_safe(probe, input_state)
    base_output = _extract_final_output(base_result)
    baseline_q = _score(base_result, base_output)

    sample_out = {n: captured.get(n, ({}, {}))[1] for n in target_nodes}
    sample_in = {n: captured.get(n, ({}, {}))[0] for n in target_nodes}

    mags = sorted(magnitudes)
    results: list[NodeSensitivity] = []
    total = len(target_nodes) * len(mags)
    done = 0

    for node in target_nodes:
        out_sample = sample_out.get(node, {})
        in_sample = sample_in.get(node, {})
        target_key = target_keys.get(node) or _primary_output_key(out_sample, in_sample, None)
        sample_value = out_sample.get(target_key) if target_key else None
        mtype = infer_module_type(node, sample_value)
        degrader = degraders.get(node) or default_degrader_for(mtype, sample_value)
        degrader_name = getattr(degrader, "__name__", None) or getattr(
            degrader, "__qualname__", "custom"
        )

        original_fn = recipe.nodes.get(node)
        curve: list = []
        errored_at_full = False
        if original_fn is not None:
            for m in mags:
                rng = random.Random(seed if seed is not None else 0)
                replacement = _degraded_node(original_fn, degrader, m, rng, target_key)
                perturbed = graph.clone_with_replacement(node, replacement)
                result, _ = _run_pipeline_safe(perturbed, input_state)
                output_text = _extract_final_output(result)
                if abs(m - mags[-1]) < _EPS and isinstance(result, dict) and "_error" in result:
                    errored_at_full = True
                curve.append((m, _score(result, output_text)))
                done += 1
                if progress_callback:
                    progress_callback(done, total, f"degrade {node} @ {m}")

        sensitivity = baseline_q - (curve[-1][1] if curve else baseline_q)
        partial = [baseline_q - q for m, q in curve if m < mags[-1] - _EPS]
        partial_sensitivity = max(partial) if partial else 0.0
        classification = _classify(baseline_q, curve, errored_at_full, structural_eps)
        results.append(
            NodeSensitivity(
                node=node,
                module_type=mtype,
                degrader=str(degrader_name),
                target_key=target_key,
                baseline_quality=baseline_q,
                curve=curve,
                sensitivity=sensitivity,
                partial_sensitivity=partial_sensitivity,
                classification=classification,
                recommendation=_RECO[classification],
                errored_at_full=errored_at_full,
            )
        )

    return SensitivityReport(
        baseline_quality=baseline_q, magnitudes=list(mags), nodes=results, seed=seed
    )
