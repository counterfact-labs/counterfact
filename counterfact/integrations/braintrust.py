"""Adapter: drive counterfact attribution and eval sets from Braintrust.

Two integration points, both additive and backward-compatible (the core API and
the existing LangGraph examples are untouched):

1. **Scoring** — :func:`quality_fn_from_scorer` adapts a Braintrust / ``autoevals``
   scorer into counterfact's ``quality_fn(output_text, state) -> float``. That is
   the metric ``CounterfactualGraph.diagnose(..., quality_fn=...)`` uses to drive
   Shapley/LOO attribution, so your Shapley scores reflect the *same* scorer your
   Braintrust evals use. Any callable shaped like an autoevals scorer works —
   ``scorer(output=..., expected=..., input=...)`` returning a ``Score`` (with a
   ``.score`` in ``[0, 1]``) or a bare float.

2. **Datasets** — :func:`cases_from_dataset` converts Braintrust dataset records
   (``{"input": ..., "expected": ...}``) into counterfact case dicts
   (``{"input": ..., "gold": ..., "id": ...}``) for ``diagnose_dataset`` /
   ``EvalSet``. :func:`load_braintrust_dataset` is a thin lazy wrapper over
   ``braintrust.init_dataset`` for callers who want to pull straight from a
   project.

The ``braintrust`` / ``autoevals`` packages are only imported when you actually
call :func:`load_braintrust_dataset` — passing your own scorer or an iterable of
records keeps everything dependency-free and offline-testable.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

QualityFn = Callable[[str, dict], float]


def quality_fn_from_scorer(
    scorer: Callable[..., Any],
    *,
    expected_key: str = "expected",
    input_key: Optional[str] = None,
    pass_input: bool = False,
    on_missing_expected: str = "error",
) -> QualityFn:
    """Adapt a Braintrust/autoevals scorer to a counterfact ``quality_fn``.

    The returned function has the signature counterfact expects —
    ``(output_text: str, state: dict) -> float`` — and pulls the reference label
    from ``state[expected_key]`` (the per-case gold carried alongside the input).

    Args:
        scorer: A callable invoked as ``scorer(output=..., expected=...)`` (plus
            ``input=...`` if ``pass_input``). May return an autoevals ``Score``
            (its ``.score`` is used) or a plain number. A ``None`` score maps to
            ``0.0``.
        expected_key: State key holding the reference/gold answer.
        input_key: State key holding the original question (used as the scorer's
            ``input=`` when ``pass_input`` is True).
        pass_input: Whether to pass ``input=`` to the scorer (some scorers, e.g.
            ``Factuality``, use it; heuristic scorers ignore it).
        on_missing_expected: ``"error"`` (default) raises if the gold is missing;
            ``"zero"`` scores such a case ``0.0`` instead.

    Returns:
        ``quality_fn(output_text, state) -> float`` in ``[0, 1]``.
    """
    if on_missing_expected not in ("error", "zero"):
        raise ValueError("on_missing_expected must be 'error' or 'zero'")

    def _quality(output_text: str, state: dict) -> float:
        if expected_key not in state or state.get(expected_key) is None:
            if on_missing_expected == "zero":
                return 0.0
            raise KeyError(
                f"quality_fn_from_scorer: state has no reference under '{expected_key}'. "
                f"Carry the gold label in the input state, or pass on_missing_expected='zero'."
            )
        kwargs: dict[str, Any] = {"output": output_text, "expected": state[expected_key]}
        if pass_input:
            kwargs["input"] = state.get(input_key) if input_key else None
        result = scorer(**kwargs)
        score = getattr(result, "score", result)
        if score is None:
            return 0.0
        return float(score)

    return _quality


def _record_get(record: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict-like or attribute-style dataset record."""
    if isinstance(record, dict):
        return record.get(key, default)
    if hasattr(record, "get"):
        try:
            return record.get(key, default)
        except TypeError:  # pragma: no cover - defensive
            pass
    return getattr(record, key, default)


def cases_from_dataset(
    dataset: Iterable[Any],
    *,
    input_key: str = "input",
    expected_key: str = "expected",
    id_key: str = "id",
    embed_expected_as: Optional[str] = "expected",
) -> list[dict]:
    """Convert Braintrust dataset records into counterfact case dicts.

    Each Braintrust record looks like ``{"input": ..., "expected": ..., "id": ...}``.
    counterfact's ``diagnose_dataset`` / ``EvalSet`` want
    ``{"input": <dict>, "gold": <expected>, "id": ...}``.

    Crucially, the gold is *also* embedded inside the per-case ``input`` state
    (under ``embed_expected_as``) so that a ``quality_fn`` built with
    :func:`quality_fn_from_scorer` can read the reference during ablation re-runs.

    Args:
        dataset: Any iterable of records (a ``braintrust`` dataset handle is
            iterable; a list of dicts works for offline use).
        input_key / expected_key / id_key: Field names on each record.
        embed_expected_as: If set, copy the expected value into the input state
            under this key (default ``"expected"``, matching
            ``quality_fn_from_scorer``'s default). Pass ``None`` to skip.

    Returns:
        A list of ``{"input", "gold", "id"}`` case dicts in dataset order.
    """
    cases: list[dict] = []
    for i, record in enumerate(dataset):
        raw_input = _record_get(record, input_key, {})
        expected = _record_get(record, expected_key)
        case_id = _record_get(record, id_key, i)

        # Normalize the input into a state dict.
        if isinstance(raw_input, dict):
            state = dict(raw_input)
        else:
            state = {input_key: raw_input}

        if embed_expected_as and expected is not None:
            state.setdefault(embed_expected_as, expected)

        cases.append({"input": state, "gold": expected, "id": case_id})
    return cases


def load_braintrust_dataset(project: str, name: str, **init_kwargs: Any) -> Any:
    """Lazily open a Braintrust dataset handle via ``braintrust.init_dataset``.

    This is the only function here that imports ``braintrust``. The returned
    handle is iterable, so pass it straight to :func:`cases_from_dataset`.

    Args:
        project: Braintrust project name.
        name: Dataset name within the project.
        **init_kwargs: Forwarded to ``braintrust.init_dataset``.

    Returns:
        A Braintrust dataset handle (iterable of records).
    """
    try:
        import braintrust
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "The Braintrust SDK is not installed. Install it with "
            "`pip install 'counterfact[braintrust]'`, or build cases yourself and "
            "pass them to cases_from_dataset()."
        ) from exc
    return braintrust.init_dataset(project=project, name=name, **init_kwargs)
