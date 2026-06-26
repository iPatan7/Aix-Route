"""Unit tests for tasks."""

from __future__ import annotations

import pytest
from deterministic_horizon import (
    ArithmeticTask,
    FSATask,
    PermutationTask,
    generate_instances,
)


def test_permutation_apply_operator_swap():
    task = PermutationTask(n_elements=4, seed=0)
    state = [0, 1, 2, 3]
    assert task.apply_operator(state, "swap_01") == [1, 0, 2, 3]
    assert task.apply_operator(state, "rotate_left") == [1, 2, 3, 0]
    assert task.apply_operator(state, "rotate_right") == [3, 0, 1, 2]


def test_permutation_bfs_returns_solution():
    task = PermutationTask(n_elements=4, seed=0)
    sol = task.bfs_solve([0, 1, 2, 3], [1, 0, 2, 3], max_depth=5)
    assert sol is not None
    ops, _ = sol
    assert ops == ["swap_01"]


def test_generate_instance_round_trip():
    task = PermutationTask(n_elements=6, seed=42)
    inst = task.generate_instance(target_depth=8)
    # Replaying the operator sequence on the initial state must yield the target.
    state = list(inst.initial_state)
    for op in inst.optimal_solution:
        state = task.apply_operator(state, op)
    assert task.state_equal(state, inst.target_state)
    assert inst.optimal_depth == 8


def test_generate_instances_helper():
    instances = generate_instances("permutation", n_instances=20, depth_range=(4, 12), seed=7)
    assert len(instances) > 0
    depths = {i.optimal_depth for i in instances}
    assert depths <= {4, 5, 6, 7, 8, 9, 10, 11, 12}


def test_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        generate_instances("does-not-exist", n_instances=2)


# --- Construction + prompt-formatting regression coverage for every task ---
# default_operators() runs inside BaseTask.__init__, so constructing each task
# guards against attribute-ordering regressions; format_prompt is exercised for
# all five experimental conditions (C1-C5).

_TASK_FACTORIES = [
    pytest.param(lambda: PermutationTask(n_elements=8, seed=1), id="permutation"),
    pytest.param(lambda: FSATask(seed=1), id="fsa"),
    pytest.param(lambda: ArithmeticTask(seed=1), id="arithmetic"),
]


@pytest.mark.parametrize("make_task", _TASK_FACTORIES)
def test_task_constructs_with_defaults(make_task):
    task = make_task()
    assert task.default_operators()


@pytest.mark.parametrize("make_task", _TASK_FACTORIES)
@pytest.mark.parametrize("condition", ["C1", "C2", "C3", "C4", "C5"])
def test_format_prompt_returns_nonempty_pair(make_task, condition):
    task = make_task()
    state = task.initial_state()
    user_prompt, system_prompt = task.format_prompt(state, state, condition)
    assert isinstance(user_prompt, str) and user_prompt.strip()
    assert isinstance(system_prompt, str) and system_prompt.strip()


@pytest.mark.parametrize("make_task", _TASK_FACTORIES)
def test_condition_semantics_match_paper(make_task):
    # C2 = depth-limited CoT (oracle optimal length); C4 = explicit length
    # encouragement. Lock these in so they cannot silently revert.
    task = make_task()
    state = task.initial_state()
    c2 = " ".join(task.format_prompt(state, state, "C2")).lower()
    c4 = " ".join(task.format_prompt(state, state, "C4")).lower()
    assert "minimum" in c2 or "optimal" in c2
    assert "as many steps" in c4
