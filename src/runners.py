"""
High-level evaluation runner.

This module exposes the :func:`run_evaluation` helper that the README's
*Python API* example calls into. It mirrors the CLI ``evaluate`` command but
is import-friendly and accepts pre-loaded objects, returning structured
results instead of writing JSON to disk.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from deterministic_horizon.tasks import load_task
from deterministic_horizon.tasks.base import TaskInstance

log = logging.getLogger(__name__)


def run_evaluation(
    model: str,
    instances: Sequence[TaskInstance] | Sequence[Mapping[str, Any]],
    conditions: Sequence[str] = ("C1", "C3"),
    *,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    task: str | None = None,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """
    Evaluate ``model`` on ``instances`` across the given ``conditions``.

    Parameters
    ----------
    model : str
        Identifier passed to :func:`deterministic_horizon.models.load_model`.
    instances : sequence of :class:`TaskInstance` or dicts
        Instances to evaluate. Dicts are converted via ``TaskInstance.from_dict``.
    conditions : sequence of str
        Subset of ``{"C1", "C2", "C3", "C4", "C5"}``.
    temperature : float
    max_tokens : int
    task : str, optional
        Override task auto-detection. Otherwise the task name is taken from
        the first instance.
    progress : bool
        Show a progress bar (``tqdm`` if available).

    Returns
    -------
    list of result dicts (one per (instance, condition) pair).
    """
    from deterministic_horizon.models import load_model

    inst_list: list[TaskInstance] = [
        i if isinstance(i, TaskInstance) else TaskInstance.from_dict(i) for i in instances
    ]
    if not inst_list:
        return []

    task_name = task or inst_list[0].task_name
    task_obj = load_task(task_name)
    model_obj = load_model(model, temperature=temperature, max_tokens=max_tokens)

    pairs: list[tuple[TaskInstance, str]] = [(i, c) for c in conditions for i in inst_list]
    iterator: Iterable[tuple[TaskInstance, str]] = pairs
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(pairs, desc=f"Evaluating {model}")
        except ImportError:  # pragma: no cover
            pass

    results: list[dict[str, Any]] = []
    for instance, condition in iterator:
        try:
            prompt, system_prompt = task_obj.format_prompt(
                instance.initial_state, instance.target_state, condition
            )
            if condition == "C3":
                response = model_obj.generate_with_tools(
                    prompt, task_obj.get_tool_definitions(), system_prompt
                )
            else:
                response = model_obj.generate(prompt, system_prompt)

            result = task_obj.evaluate(instance, response.content)
            result.condition = condition
            result.model = model
            result.total_tokens = response.total_tokens
            result.latency_ms = response.latency_ms
            result.tool_calls = response.tool_calls

            payload = result.to_dict()
            payload["optimal_depth"] = instance.optimal_depth
            results.append(payload)
        except Exception as exc:  # log + skip; never abort the whole run
            log.warning("instance %s failed: %s", instance.instance_id, exc)
            results.append(
                {
                    "instance_id": instance.instance_id,
                    "condition": condition,
                    "model": model,
                    "correct": False,
                    "optimal_depth": instance.optimal_depth,
                    "error": str(exc),
                }
            )

    return results


__all__ = ["run_evaluation"]
