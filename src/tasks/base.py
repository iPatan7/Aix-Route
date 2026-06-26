"""Abstract base class for state-space search tasks."""

import hashlib
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskInstance:
    """Container for a single task instance."""

    # Instance identification
    instance_id: str
    task_name: str

    # Problem specification
    initial_state: Any
    target_state: Any
    optimal_depth: int

    # Optimal solution
    optimal_solution: list[str]
    intermediate_states: list[Any]

    # Prompt
    prompt: str
    system_prompt: str = ""

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "instance_id": self.instance_id,
            "task_name": self.task_name,
            "initial_state": self._serialize_state(self.initial_state),
            "target_state": self._serialize_state(self.target_state),
            "optimal_depth": self.optimal_depth,
            "optimal_solution": self.optimal_solution,
            "intermediate_states": [self._serialize_state(s) for s in self.intermediate_states],
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskInstance":
        """Create from dictionary."""
        return cls(
            instance_id=data["instance_id"],
            task_name=data["task_name"],
            initial_state=data["initial_state"],
            target_state=data["target_state"],
            optimal_depth=data["optimal_depth"],
            optimal_solution=data["optimal_solution"],
            intermediate_states=data["intermediate_states"],
            prompt=data["prompt"],
            system_prompt=data.get("system_prompt", ""),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def _serialize_state(state: Any) -> Any:
        """Serialize state for JSON storage."""
        if isinstance(state, (list, tuple)):
            return list(state)
        elif isinstance(state, dict):
            return state
        elif isinstance(state, set):
            return list(state)
        return state


@dataclass
class TaskResult:
    """Container for task evaluation result."""

    instance_id: str
    condition: str
    model: str

    # Core result
    correct: bool
    model_answer: str

    # Trace analysis
    model_trace: list[tuple[Any, str]]  # (claimed_state, operation)
    step_to_first_error: int | None

    # Metrics
    ssj_score: float | None = None
    precision: float | None = None
    recall: float | None = None

    # Response metadata
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0

    # Tool usage (for C3)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    # Raw response
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instance_id": self.instance_id,
            "condition": self.condition,
            "model": self.model,
            "correct": self.correct,
            "model_answer": self.model_answer,
            "model_trace": [(TaskInstance._serialize_state(s), op) for s, op in self.model_trace],
            "step_to_first_error": self.step_to_first_error,
            "ssj_score": self.ssj_score,
            "precision": self.precision,
            "recall": self.recall,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "tool_calls": self.tool_calls,
            "raw_response": self.raw_response,
        }


class BaseTask(ABC):
    """Abstract base class for state-space search tasks."""

    def __init__(
        self,
        seed: int = 42,
        n_elements: int = 8,
        operators: list[str] | None = None,
    ) -> None:
        """
        Initialize task.

        Args:
            seed: Random seed for deterministic generation
            n_elements: Number of elements in state space
            operators: List of operator names
        """
        self.seed = seed
        self.n_elements = n_elements
        self.operators = operators or self.default_operators()
        self._rng = random.Random(seed)

    @abstractmethod
    def default_operators(self) -> list[str]:
        """Return default operators for this task."""
        pass

    @abstractmethod
    def initial_state(self) -> Any:
        """Generate a random initial state."""
        pass

    @abstractmethod
    def apply_operator(self, state: Any, operator: str) -> Any:
        """Apply an operator to a state."""
        pass

    @abstractmethod
    def state_equal(self, state1: Any, state2: Any) -> bool:
        """Check if two states are equal."""
        pass

    @abstractmethod
    def state_to_string(self, state: Any) -> str:
        """Convert state to string representation."""
        pass

    @abstractmethod
    def parse_state(self, text: str) -> Any | None:
        """Parse state from text. Returns None if parsing fails."""
        pass

    @abstractmethod
    def format_prompt(
        self,
        initial_state: Any,
        target_state: Any,
        condition: str,
    ) -> tuple[str, str]:
        """
        Format task prompt.

        Returns:
            (user_prompt, system_prompt) tuple
        """
        pass

    def random_operator(self) -> str:
        """Select a random operator."""
        return self._rng.choice(self.operators)

    def generate_instance(self, target_depth: int) -> TaskInstance:
        """
        Generate a single task instance.

        Args:
            target_depth: Target solution depth (BFS optimal)

        Returns:
            TaskInstance object
        """
        # Generate initial state
        state = self.initial_state()
        states = [state]
        operations = []

        # Apply random operations
        for _ in range(target_depth):
            op = self.random_operator()
            state = self.apply_operator(state, op)
            operations.append(op)
            states.append(state)

        target = state

        # Create instance ID
        instance_data = {
            "initial": self.state_to_string(states[0]),
            "target": self.state_to_string(target),
            "depth": target_depth,
        }
        instance_id = hashlib.md5(json.dumps(instance_data, sort_keys=True).encode()).hexdigest()[
            :12
        ]

        # Format prompt (using C1 as default for generation)
        prompt, system_prompt = self.format_prompt(states[0], target, "C1")

        return TaskInstance(
            instance_id=instance_id,
            task_name=self.__class__.__name__.lower().replace("task", ""),
            initial_state=states[0],
            target_state=target,
            optimal_depth=target_depth,
            optimal_solution=operations,
            intermediate_states=states,
            prompt=prompt,
            system_prompt=system_prompt,
            metadata={
                "n_elements": self.n_elements,
                "operators": self.operators,
            },
        )

    def generate_instances(
        self,
        n_instances: int = 100,
        min_depth: int = 5,
        max_depth: int = 30,
        depth_step: int = 5,
    ) -> list[TaskInstance]:
        """
        Generate multiple task instances across depth range.

        Args:
            n_instances: Total number of instances
            min_depth: Minimum solution depth
            max_depth: Maximum solution depth
            depth_step: Step between depth values

        Returns:
            List of TaskInstance objects
        """
        depths = list(range(min_depth, max_depth + 1, depth_step))
        instances_per_depth = n_instances // len(depths)

        instances = []
        for depth in depths:
            for _ in range(instances_per_depth):
                try:
                    instances.append(self.generate_instance(depth))
                except ValueError:
                    # Depth exceeds this task's reachable diameter (e.g. a
                    # permutation depth above C(n, 2)); skip it rather than abort.
                    break

        # Shuffle to avoid depth-based ordering artifacts
        self._rng.shuffle(instances)

        return instances

    def evaluate(
        self,
        instance: TaskInstance,
        model_response: str,
    ) -> TaskResult:
        """
        Evaluate model response against instance.

        Args:
            instance: Task instance
            model_response: Model's response text

        Returns:
            TaskResult with evaluation metrics
        """
        # Parse model trace
        model_trace = self.parse_trace(model_response)

        # Extract final answer
        final_state = model_trace[-1][0] if model_trace else None

        # Check correctness
        correct = final_state is not None and self.state_equal(final_state, instance.target_state)

        # Compute step to first error
        sfe = self.compute_sfe(instance, model_trace)

        # Compute SSJ metrics
        ssj, precision, recall = self.compute_ssj(instance, model_trace)

        return TaskResult(
            instance_id=instance.instance_id,
            condition="",  # Set by caller
            model="",  # Set by caller
            correct=correct,
            model_answer=self.state_to_string(final_state) if final_state else "",
            model_trace=model_trace,
            step_to_first_error=sfe,
            ssj_score=ssj,
            precision=precision,
            recall=recall,
            raw_response=model_response,
        )

    def parse_trace(self, response: str) -> list[tuple[Any, str]]:
        """
        Parse reasoning trace from model response.

        Returns list of (state, operation) tuples.
        """
        trace = []
        lines = response.split("\n")

        current_state = None
        current_op = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to parse state
            parsed_state = self.parse_state(line)
            if parsed_state is not None:
                if current_state is not None:
                    trace.append((current_state, current_op or ""))
                current_state = parsed_state
                current_op = None

            # Try to identify operation
            for op in self.operators:
                if op.lower() in line.lower():
                    current_op = op
                    break

        # Add final state
        if current_state is not None:
            trace.append((current_state, current_op or ""))

        return trace

    def compute_sfe(
        self,
        instance: TaskInstance,
        model_trace: list[tuple[Any, str]],
    ) -> int | None:
        """Compute step-to-first-error."""
        if not model_trace:
            return 0

        # Compare model trace to ground truth
        for i, (model_state, _) in enumerate(model_trace):
            if i >= len(instance.intermediate_states):
                return i
            if not self.state_equal(model_state, instance.intermediate_states[i]):
                return i

        return None  # No error found

    def compute_ssj(
        self,
        instance: TaskInstance,
        model_trace: list[tuple[Any, str]],
    ) -> tuple[float, float, float]:
        """
        Compute State-Space Jaccard with precision/recall decomposition.

        Returns:
            (ssj_score, precision, recall) tuple
        """
        if not model_trace:
            return 0.0, 0.0, 0.0

        # Get state sets
        true_states = {self.state_to_string(s) for s in instance.intermediate_states}
        model_states = {self.state_to_string(s) for s, _ in model_trace}

        # Compute metrics
        intersection = true_states & model_states
        union = true_states | model_states

        if not union:
            return 0.0, 0.0, 0.0

        ssj = len(intersection) / len(union)
        precision = len(intersection) / len(model_states) if model_states else 0.0
        recall = len(intersection) / len(true_states) if true_states else 0.0

        return ssj, precision, recall

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Get tool definitions for C3 condition.

        Override in subclasses for task-specific tools.
        """
        return [
            {
                "function": {
                    "name": "apply_operation",
                    "description": f"Apply an operation to the current state. Valid operations: {', '.join(self.operators)}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": self.operators,
                                "description": "The operation to apply",
                            },
                        },
                        "required": ["operation"],
                    },
                },
            },
            {
                "function": {
                    "name": "verify_state",
                    "description": "Verify the current state matches expected state",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expected_state": {
                                "type": "string",
                                "description": "The expected state to verify",
                            },
                        },
                        "required": ["expected_state"],
                    },
                },
            },
        ]
