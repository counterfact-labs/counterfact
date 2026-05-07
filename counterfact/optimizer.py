"""
Single-objective pipeline optimizer.

Finds the best pipeline configuration (model assignments, temperature,
top-K, loop depth, etc.) to maximize a single quality metric.

Uses Google OSS Vizier for Bayesian optimization when available, with
a random-search fallback. The optimizer is completely generic: the user
provides an evaluation function and a search space definition.

Usage:
    from counterfact.optimizer import optimize_pipeline, SearchSpace

    space = SearchSpace()
    space.add_categorical("synth_model", ["gemini-2.5-pro", "gemini-2.5-flash"])
    space.add_float("temperature", 0.0, 1.0)
    space.add_int("top_k", 1, 10)

    def evaluate(config: dict) -> float:
        result = my_pipeline.invoke(config)
        return score_quality(result)

    result = optimize_pipeline(evaluate, space, num_trials=30)
    print(result.best_config)     # {"synth_model": "gemini-2.5-pro", "temperature": 0.3, ...}
    print(result.best_quality)    # 0.92

Dependencies: numpy (types only for dataclasses)
"""

import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

# ═════════════════════════════════════════════════════════════════════════
# SEARCH SPACE DEFINITION
# ═════════════════════════════════════════════════════════════════════════


@dataclass
class ParamSpec:
    """Specification for a single search parameter."""
    name: str
    param_type: str   # "categorical", "float", "int"
    values: Optional[list] = None       # for categorical
    min_value: Optional[float] = None   # for float/int
    max_value: Optional[float] = None   # for float/int


class SearchSpace:
    """
    Define the parameter search space for pipeline optimization.

    Supports three parameter types:
      - categorical: discrete choices (e.g., model names)
      - float: continuous range (e.g., temperature)
      - int: integer range (e.g., top-K, loop depth)

    Example:
        space = SearchSpace()
        space.add_categorical("model", ["pro", "flash", "lite"])
        space.add_float("temperature", 0.0, 1.0)
        space.add_int("max_loops", 1, 5)
    """

    def __init__(self):
        self.params: list[ParamSpec] = []

    def add_categorical(self, name: str, values: list[str]) -> "SearchSpace":
        """Add a categorical parameter (e.g., model selection)."""
        self.params.append(ParamSpec(
            name=name, param_type="categorical", values=values,
        ))
        return self

    def add_float(self, name: str, min_value: float, max_value: float) -> "SearchSpace":
        """Add a continuous float parameter (e.g., temperature)."""
        self.params.append(ParamSpec(
            name=name, param_type="float",
            min_value=min_value, max_value=max_value,
        ))
        return self

    def add_int(self, name: str, min_value: int, max_value: int) -> "SearchSpace":
        """Add an integer parameter (e.g., top-K, loop depth)."""
        self.params.append(ParamSpec(
            name=name, param_type="int",
            min_value=min_value, max_value=max_value,
        ))
        return self

    def sample_random(self, rng: random.Random) -> dict:
        """Sample a random configuration from the search space."""
        config: dict[str, Any] = {}
        for p in self.params:
            if p.param_type == "categorical":
                config[p.name] = rng.choice(p.values or [])
            elif p.param_type == "float":
                config[p.name] = rng.uniform(p.min_value or 0.0, p.max_value or 1.0)
            elif p.param_type == "int":
                config[p.name] = rng.randint(int(p.min_value or 0), int(p.max_value or 100))
        return config

    def __len__(self) -> int:
        return len(self.params)


# ═════════════════════════════════════════════════════════════════════════
# RESULT TYPES
# ═════════════════════════════════════════════════════════════════════════


@dataclass
class TrialResult:
    """Result from evaluating a single configuration."""
    trial_id: int
    config: dict[str, Any]
    quality: float

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "config": self.config,
            "quality": round(self.quality, 4),
        }


@dataclass
class OptimizationResult:
    """
    Complete result of a pipeline optimization run.

    Attributes:
        best_config: The configuration that achieved the highest quality.
        best_quality: The quality score of the best configuration.
        all_trials: All evaluated configurations, sorted by quality descending.
        improvement: Absolute quality improvement of best over worst trial.
        search_method: "vizier" or "random_search".
    """
    best_config: dict[str, Any]
    best_quality: float
    all_trials: list[TrialResult]
    improvement: float
    search_method: str
    num_trials: int

    def to_dict(self) -> dict:
        return {
            "best_config": self.best_config,
            "best_quality": round(self.best_quality, 4),
            "all_trials": [t.to_dict() for t in self.all_trials],
            "improvement": round(self.improvement, 4),
            "search_method": self.search_method,
            "num_trials": self.num_trials,
            "top_5": [t.to_dict() for t in self.all_trials[:5]],
        }


# ═════════════════════════════════════════════════════════════════════════
# VIZIER OPTIMIZATION (preferred)
# ═════════════════════════════════════════════════════════════════════════


def _try_vizier(
    evaluate_fn: Callable[[dict], float],
    space: SearchSpace,
    num_trials: int,
    progress_callback: Optional[Callable] = None,
) -> Optional[list[TrialResult]]:  # pragma: no cover
    """
    Run Bayesian optimization via Google OSS Vizier.

    Vizier uses a Gaussian Process surrogate model to efficiently explore
    the search space — it learns which regions are promising and focuses
    trials there, rather than sampling randomly.

    Returns None if Vizier is not installed.
    """
    try:
        from vizier.service import clients  # type: ignore
        from vizier.service import pyvizier as vz
    except ImportError:
        return None

    problem = vz.ProblemStatement()

    for p in space.params:
        if p.param_type == "categorical":
            problem.search_space.root.add_categorical_param(
                p.name, feasible_values=[str(v) for v in (p.values or [])],
            )
        elif p.param_type == "float":
            problem.search_space.root.add_float_param(
                p.name, min_value=p.min_value or 0.0, max_value=p.max_value or 1.0,
            )
        elif p.param_type == "int":
            problem.search_space.root.add_int_param(
                p.name, min_value=int(p.min_value or 0), max_value=int(p.max_value or 100),
            )

    # Single objective: maximize quality
    problem.metric_information.append(
        vz.MetricInformation(name="quality", goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    )

    study_config = vz.StudyConfig.from_problem(problem)
    study_config.algorithm = "DEFAULT"

    study_client = clients.Study.from_study_config(
        study_config,
        owner="counterfact",
        study_id=f"quality_opt_{random.randint(0, 99999)}",
    )

    trials = []
    for i in range(num_trials):
        suggestions = study_client.suggest(count=1)
        for suggestion in suggestions:
            config: dict[str, Any] = {}
            for p in space.params:
                raw = suggestion.parameters[p.name]
                if p.param_type == "int":
                    config[p.name] = int(raw)
                elif p.param_type == "float":
                    config[p.name] = float(raw)
                else:
                    config[p.name] = raw

            quality = evaluate_fn(config)
            trials.append(TrialResult(trial_id=i, config=config, quality=quality))

            suggestion.complete(vz.Measurement({"quality": quality}))

        if progress_callback:
            progress_callback(i + 1, num_trials, config)

    return trials


# ═════════════════════════════════════════════════════════════════════════
# RANDOM SEARCH FALLBACK
# ═════════════════════════════════════════════════════════════════════════


def _random_search(
    evaluate_fn: Callable[[dict], float],
    space: SearchSpace,
    num_trials: int,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
) -> list[TrialResult]:
    """
    Random search fallback when Vizier is not available.

    Samples random configurations and evaluates each one. Simple but effective
    for small-to-medium search spaces (< 100 trials).
    """
    rng = random.Random(seed)
    trials = []

    for i in range(num_trials):
        config = space.sample_random(rng)
        quality = evaluate_fn(config)
        trials.append(TrialResult(trial_id=i, config=config, quality=quality))

        if progress_callback:
            progress_callback(i + 1, num_trials, config)

    return trials


# ═════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════


def optimize_pipeline(
    evaluate_fn: Callable[[dict], float],
    search_space: SearchSpace,
    num_trials: int = 30,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
) -> OptimizationResult:
    """
    Find the pipeline configuration that maximizes output quality.

    This is a single-objective optimizer: it searches over the parameter
    space defined by `search_space` and finds the configuration that
    produces the highest quality score as measured by `evaluate_fn`.

    Args:
        evaluate_fn: Function that takes a config dict and returns a
            quality score (float, higher is better). This should run
            the pipeline with the given parameters and score the output.
        search_space: SearchSpace defining the parameters to optimize
            (model choices, temperature, top-K, loop depth, etc.).
        num_trials: Number of configurations to evaluate. More trials
            = better results but longer runtime.
        seed: Random seed for reproducibility (random search only).
        progress_callback: Optional callback(current, total, config).

    Returns:
        OptimizationResult with the best configuration, all trial
        results, and improvement metrics.

    Example:
        space = SearchSpace()
        space.add_categorical("synth_model", ["pro", "flash", "lite"])
        space.add_float("temperature", 0.0, 1.0)
        space.add_int("max_loops", 1, 5)

        def evaluate(config):
            result = pipeline.invoke({"model": config["synth_model"], ...})
            return score_quality(result)

        result = optimize_pipeline(evaluate, space, num_trials=30)
        print(result.best_config)   # {"synth_model": "pro", "temperature": 0.31, ...}
        print(result.best_quality)  # 0.92
    """
    # Try Vizier first (Bayesian optimization)
    trials = _try_vizier(evaluate_fn, search_space, num_trials, progress_callback)
    search_method = "vizier"

    if trials is None:
        trials = _random_search(
            evaluate_fn, search_space, num_trials, seed, progress_callback,
        )
        search_method = "random_search"

    # Sort by quality descending
    trials.sort(key=lambda t: t.quality, reverse=True)

    best = trials[0] if trials else TrialResult(0, {}, 0.0)
    worst = trials[-1] if trials else TrialResult(0, {}, 0.0)

    return OptimizationResult(
        best_config=best.config,
        best_quality=best.quality,
        all_trials=trials,
        improvement=best.quality - worst.quality,
        search_method=search_method,
        num_trials=len(trials),
    )
