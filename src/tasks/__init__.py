"""Task implementations for state-space search problems."""

from deterministic_horizon.tasks.arithmetic import ArithmeticTask
from deterministic_horizon.tasks.base import BaseTask, TaskInstance, TaskResult
from deterministic_horizon.tasks.fsa import FSATask
from deterministic_horizon.tasks.permutation import PermutationTask

# Task registry
TASK_REGISTRY: dict[str, type[BaseTask]] = {
    "permutation": PermutationTask,
    "fsa": FSATask,
    "arithmetic": ArithmeticTask,
}


def generate_instances(
    task: str,
    n_instances: int = 100,
    depth_range: tuple[int, int] = (4, 28),
    depth_step: int = 4,
    seed: int = 42,
    **task_kwargs,
) -> list[TaskInstance]:
    """
    Generate task instances.

    Args:
        task: Task name (e.g., 'permutation', 'fsa', 'arithmetic')
        n_instances: Number of instances to generate
        depth_range: (min_depth, max_depth) tuple. For PermutationProbe the max
            BFS-optimal depth is the diameter C(n, 2) (28 for the default n=8).
        depth_step: Step between successive depths.
        seed: Random seed for deterministic generation
        **task_kwargs: Additional task-specific arguments

    Returns:
        List of TaskInstance objects
    """
    task_lower = task.lower()

    if task_lower not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(f"Unknown task: {task}. Available: {available}")

    task_class = TASK_REGISTRY[task_lower]
    task_obj = task_class(seed=seed, **task_kwargs)

    return task_obj.generate_instances(
        n_instances=n_instances,
        min_depth=depth_range[0],
        max_depth=depth_range[1],
        depth_step=depth_step,
    )


def load_task(task_name: str, **kwargs) -> BaseTask:
    """Load a task by name."""
    task_lower = task_name.lower()
    if task_lower not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(f"Unknown task: {task_name}. Available: {available}")
    return TASK_REGISTRY[task_lower](**kwargs)


__all__ = [
    "BaseTask",
    "TaskInstance",
    "TaskResult",
    "PermutationTask",
    "FSATask",
    "ArithmeticTask",
    "generate_instances",
    "load_task",
    "TASK_REGISTRY",
]
