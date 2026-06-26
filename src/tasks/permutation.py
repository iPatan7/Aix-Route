"""Permutation puzzle task implementation."""

import re
from collections import deque
from typing import Any

from deterministic_horizon.tasks.base import BaseTask, TaskInstance


class PermutationTask(BaseTask):
    """
    PermutationProbe task (paper §5).

    States are permutations of ``n`` elements (the symmetric group ``S_n``) and
    the canonical operators are the ``n-1`` **adjacent transpositions**
    ``{(i, i+1)}``, matching the paper's task definition (state space ``S_n``,
    diameter ``C(n, 2)``). Under adjacent transpositions the BFS-optimal solution
    length from the identity to a target equals the target's inversion count, so
    instances can be generated at an *exactly* controlled BFS-optimal depth
    (paper Appendix, "PermutationProbe Instance Generation").

    Rotations/reversals remain available through :meth:`apply_operator` for
    experimentation, but they are intentionally excluded from the default
    operator set so that ``optimal_depth`` is a true BFS-optimal depth.
    """

    def __init__(
        self,
        seed: int = 42,
        n_elements: int = 8,
        operators: list[str] | None = None,
    ) -> None:
        super().__init__(seed=seed, n_elements=n_elements, operators=operators)

        # Build operator functions
        self._operator_funcs = {
            "swap_01": lambda p: self._swap(p, 0, 1),
            "swap_12": lambda p: self._swap(p, 1, 2),
            "swap_23": lambda p: self._swap(p, 2, 3),
            "swap_34": lambda p: self._swap(p, 3, 4),
            "swap_45": lambda p: self._swap(p, 4, 5),
            "swap_56": lambda p: self._swap(p, 5, 6),
            "swap_67": lambda p: self._swap(p, 6, 7),
            "rotate_left": lambda p: p[1:] + [p[0]],
            "rotate_right": lambda p: [p[-1]] + p[:-1],
            "reverse": lambda p: p[::-1],
            "reverse_first_half": lambda p: p[: len(p) // 2][::-1] + p[len(p) // 2 :],
            "reverse_second_half": lambda p: p[: len(p) // 2] + p[len(p) // 2 :][::-1],
        }

    def default_operators(self) -> list[str]:
        """Return the canonical operator set: adjacent transpositions only.

        This matches the paper's PermutationProbe definition (operators =
        adjacent transpositions ``{(i, i+1)}``), under which BFS-optimal depth
        equals the inversion count.
        """
        return [f"swap_{i}{i + 1}" for i in range(self.n_elements - 1)]

    def initial_state(self) -> list[int]:
        """Generate identity permutation as initial state."""
        return list(range(self.n_elements))

    def apply_operator(self, state: list[int], operator: str) -> list[int]:
        """Apply operator to permutation state."""
        state = list(state)  # Copy to avoid mutation

        if operator in self._operator_funcs:
            return self._operator_funcs[operator](state)

        # Try to parse swap operator
        swap_match = re.match(r"swap_(\d)(\d)", operator)
        if swap_match:
            i, j = int(swap_match.group(1)), int(swap_match.group(2))
            return self._swap(state, i, j)

        raise ValueError(f"Unknown operator: {operator}")

    def _swap(self, perm: list[int], i: int, j: int) -> list[int]:
        """Swap elements at positions i and j."""
        perm = list(perm)
        if i < len(perm) and j < len(perm):
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    # ------------------------------------------------------------------
    # BFS-optimal instance generation (paper Appendix, Algorithm 6).
    #
    # Under adjacent transpositions the BFS-optimal depth from the identity to
    # a target equals the target's inversion count. We therefore sample a random
    # target with *exactly* ``target_depth`` inversions and reconstruct the
    # optimal solution analytically, guaranteeing ``optimal_depth`` is a genuine
    # BFS-optimal depth rather than a (possibly non-minimal) random-walk length.
    # ------------------------------------------------------------------
    def max_depth(self) -> int:
        """Graph diameter: C(n, 2) for adjacent transpositions on ``S_n``."""
        n = self.n_elements
        return n * (n - 1) // 2

    def _random_target_with_inversions(self, k: int) -> list[int]:
        """Sample a uniform-ish permutation of ``[0, n)`` with exactly ``k`` inversions."""
        n = self.n_elements
        if not 0 <= k <= self.max_depth():
            raise ValueError(
                f"target_depth={k} out of range for n={n} "
                f"(adjacent-transposition diameter is {self.max_depth()})"
            )
        remaining = list(range(n))
        result: list[int] = []
        left = k
        for _ in range(n):
            max_here = len(remaining) - 1
            # Inversions the *later* positions can still absorb: C(len-1, 2).
            capacity_after = (len(remaining) - 1) * (len(remaining) - 2) // 2
            lo = max(0, left - capacity_after)
            hi = min(max_here, left)
            take = self._rng.randint(lo, hi)
            result.append(remaining.pop(take))
            left -= take
        return result

    def _optimal_path(self, target: list[int]) -> tuple[list[str], list[list[int]]]:
        """Analytic BFS-optimal adjacent-transposition path identity → target."""
        n = len(target)
        work = list(range(n))
        ops: list[str] = []
        states: list[list[int]] = [list(work)]
        for i in range(n):
            j = work.index(target[i], i)
            for pos in range(j, i, -1):
                work[pos - 1], work[pos] = work[pos], work[pos - 1]
                ops.append(f"swap_{pos - 1}{pos}")
                states.append(list(work))
        return ops, states

    def generate_instance(self, target_depth: int) -> TaskInstance:
        """Generate a PermutationProbe instance with BFS-optimal depth ``target_depth``."""
        import hashlib
        import json

        initial = self.initial_state()
        target = self._random_target_with_inversions(target_depth)
        ops, states = self._optimal_path(target)

        # Defensive verification: the analytic path must be optimal and reach
        # the target. (Inversion count == minimal adjacent transpositions.)
        if len(ops) != target_depth or not self.state_equal(states[-1], target):
            sol = self.bfs_solve(initial, target, max_depth=self.max_depth())
            if sol is None or len(sol[0]) != target_depth:
                raise RuntimeError(
                    f"failed to construct depth-{target_depth} instance for n={self.n_elements}"
                )
            ops, states = sol

        instance_data = {
            "initial": self.state_to_string(initial),
            "target": self.state_to_string(target),
            "depth": target_depth,
        }
        instance_id = hashlib.md5(json.dumps(instance_data, sort_keys=True).encode()).hexdigest()[
            :12
        ]

        prompt, system_prompt = self.format_prompt(initial, target, "C1")

        return TaskInstance(
            instance_id=instance_id,
            task_name="permutation",
            initial_state=initial,
            target_state=target,
            optimal_depth=target_depth,
            optimal_solution=ops,
            intermediate_states=states,
            prompt=prompt,
            system_prompt=system_prompt,
            metadata={"n_elements": self.n_elements, "operators": self.operators},
        )

    def state_equal(self, state1: list[int], state2: list[int]) -> bool:
        """Check if two permutations are equal."""
        return list(state1) == list(state2)

    def state_to_string(self, state: list[int]) -> str:
        """Convert permutation to string."""
        return "[" + ", ".join(map(str, state)) + "]"

    def parse_state(self, text: str) -> list[int] | None:
        """Parse permutation from text."""
        # Try to find array pattern
        match = re.search(r"\[[\s,\d]+\]", text)
        if match:
            try:
                array_str = match.group(0)
                # Extract numbers
                numbers = re.findall(r"\d+", array_str)
                return [int(n) for n in numbers]
            except (ValueError, IndexError):
                pass

        # Try to find sequence of numbers
        numbers = re.findall(r"\b(\d)\b", text)
        if len(numbers) == self.n_elements:
            return [int(n) for n in numbers]

        return None

    def format_prompt(
        self,
        initial_state: list[int],
        target_state: list[int],
        condition: str,
    ) -> tuple[str, str]:
        """Format task prompt for given condition."""
        state_str = self.state_to_string(initial_state)
        target_str = self.state_to_string(target_state)

        ops_str = ", ".join(self.operators)

        if condition == "C1":
            # Neural Chain-of-Thought
            system_prompt = """You are solving a permutation puzzle. Think through each step carefully, showing the state after each operation.

Available operations:
- swap_XY: Swap the elements at adjacent positions X and Y (Y = X+1)

Show your work step by step, writing the state after each operation."""

            user_prompt = f"""Transform the permutation from initial state to target state.

Initial state: {state_str}
Target state: {target_str}

Available operations: {ops_str}

Think step by step. For each step, write:
1. The operation you're applying
2. The resulting state

Continue until you reach the target state."""

        elif condition == "C2":
            # Depth-limited CoT (oracle optimal length)
            system_prompt = """You are solving a permutation puzzle. Reason step by step, showing the state after each operation, but use exactly the optimal (minimum) number of operations and do not add any extra steps."""

            user_prompt = f"""Transform the permutation from initial state to target state in the minimum number of operations.

Initial state: {state_str}
Target state: {target_str}

Available operations: {ops_str}

Think step by step, writing the operation and the resulting state at each step, and stop as soon as you reach the target. Use no more than the optimal number of steps."""

        elif condition == "C3":
            # Tool-integrated
            system_prompt = """You are solving a permutation puzzle with access to a verification tool.
Use the tools to verify your state after each operation.
The verify_state tool will tell you if your current state is correct."""

            user_prompt = f"""Transform the permutation from initial state to target state.

Initial state: {state_str}
Target state: {target_str}

Use the apply_operation tool to apply operations and verify_state to check your progress.
Operations: {ops_str}"""

        elif condition == "C4":
            # Explicit length encouragement
            system_prompt = """You are solving a permutation puzzle. Think through each step carefully, showing the state after each operation. Take as many steps as you need: there is no limit on the number of operations, so reason for as long as necessary to be certain."""

            user_prompt = f"""Transform the permutation from initial state to target state.

Initial state: {state_str}
Target state: {target_str}

Available operations: {ops_str}

Take as many reasoning steps as you need. For each step, write the operation you're applying and the resulting state, and continue until you reach the target state."""

        elif condition == "C5":
            # Fine-tuned format (detailed trace)
            system_prompt = """Solve the permutation puzzle step by step. 
Always show the complete state after each operation in the format: State: [x, y, z, ...]"""

            user_prompt = f"""Initial: {state_str}
Target: {target_str}
Operations: {ops_str}

Solve step by step, showing each state change."""

        else:
            raise ValueError(f"Unknown condition: {condition}")

        return user_prompt, system_prompt

    def bfs_solve(
        self,
        initial: list[int],
        target: list[int],
        max_depth: int = 50,
    ) -> tuple[list[str], list[list[int]]] | None:
        """
        Solve using BFS to find optimal solution.

        Returns:
            (operations, states) tuple or None if no solution within max_depth
        """
        initial = tuple(initial)
        target = tuple(target)

        if initial == target:
            return [], [list(initial)]

        # BFS
        queue = deque([(initial, [], [list(initial)])])
        visited = {initial}

        while queue:
            state, ops, states = queue.popleft()

            if len(ops) >= max_depth:
                continue

            for op in self.operators:
                new_state = tuple(self.apply_operator(list(state), op))

                if new_state == target:
                    return ops + [op], states + [list(new_state)]

                if new_state not in visited:
                    visited.add(new_state)
                    queue.append(
                        (
                            new_state,
                            ops + [op],
                            states + [list(new_state)],
                        )
                    )

        return None

    def make_tool_session(
        self,
        initial_state: list[int],
        target_state: list[int],
    ) -> "_PermutationToolSession":
        """
        Create a stateful tool session.

        Each LLM evaluation under the C3 (tool-integrated) condition gets a
        fresh session. The session keeps track of the *true* current state
        as operations are applied, so verify/get-state calls return correct
        information rather than canned responses.
        """
        return _PermutationToolSession(self, initial_state, target_state)

    def get_tool_definitions(
        self,
        session: "_PermutationToolSession | None" = None,
    ) -> list[dict[str, Any]]:
        """
        Return tool definitions wired to a session (created on demand).

        The returned ``executor`` callables maintain real puzzle state,
        making C3 a faithful implementation of tool-integrated reasoning.
        """
        if session is None:
            session = self.make_tool_session(self.initial_state(), self.initial_state())

        return [
            {
                "function": {
                    "name": "apply_operation",
                    "description": (
                        "Apply a permutation operation to the current state. "
                        f"Valid operations: {', '.join(self.operators)}."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": list(self.operators),
                                "description": "Operation name",
                            },
                        },
                        "required": ["operation"],
                    },
                },
                "executor": session.apply_operation,
            },
            {
                "function": {
                    "name": "verify_state",
                    "description": "Check whether the current state matches an expected permutation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expected": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Expected permutation state",
                            },
                        },
                        "required": ["expected"],
                    },
                },
                "executor": session.verify_state,
            },
            {
                "function": {
                    "name": "get_current_state",
                    "description": "Return the current permutation state.",
                    "parameters": {"type": "object", "properties": {}},
                },
                "executor": session.get_current_state,
            },
            {
                "function": {
                    "name": "solve_bfs",
                    "description": (
                        "Return an optimal sequence of operations from the current "
                        "state to the target. This is the canonical 'tool' for C3."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
                "executor": session.solve_bfs,
            },
        ]


class _PermutationToolSession:
    """Stateful session for the C3 tool-integrated condition."""

    def __init__(
        self,
        task: "PermutationTask",
        initial_state: list[int],
        target_state: list[int],
    ) -> None:
        self._task = task
        self._state: list[int] = list(initial_state)
        self._target: list[int] = list(target_state)
        self._history: list[str] = []

    def apply_operation(self, operation: str) -> dict[str, Any]:
        try:
            self._state = self._task.apply_operator(self._state, operation)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
        self._history.append(operation)
        return {
            "status": "success",
            "operation": operation,
            "state": list(self._state),
            "matches_target": self._task.state_equal(self._state, self._target),
        }

    def verify_state(self, expected: list[int]) -> dict[str, Any]:
        return {
            "status": "success",
            "matches": list(expected) == list(self._state),
            "actual": list(self._state),
        }

    def get_current_state(self) -> dict[str, Any]:
        return {"status": "success", "state": list(self._state)}

    def solve_bfs(self) -> dict[str, Any]:
        sol = self._task.bfs_solve(self._state, self._target, max_depth=60)
        if sol is None:
            return {"status": "error", "message": "no solution within max_depth"}
        ops, states = sol
        return {"status": "success", "operations": ops, "states": [list(s) for s in states]}
