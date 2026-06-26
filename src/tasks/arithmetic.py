"""Multi-step arithmetic task implementation."""

import re
from fractions import Fraction
from typing import Any

from deterministic_horizon.tasks.base import BaseTask


class ArithmeticTask(BaseTask):
    """
    Multi-step arithmetic task.

    States are numerical values. Operations are arithmetic operations
    applied sequentially. Goal is to transform initial value to target.
    """

    def __init__(
        self,
        seed: int = 42,
        n_elements: int = 100,  # Max value range
        use_fractions: bool = False,
        **kwargs,
    ) -> None:
        self.max_value = n_elements
        self.use_fractions = use_fractions
        # Operation parameters
        self._operands = [2, 3, 5, 7, 10]
        super().__init__(seed=seed, n_elements=n_elements, **kwargs)

    def default_operators(self) -> list[str]:
        """Return default arithmetic operators."""
        ops = []
        for n in self._operands:
            ops.extend([f"add_{n}", f"sub_{n}", f"mul_{n}"])
            if n != 0:
                ops.append(f"div_{n}")
        return ops

    def initial_state(self) -> Fraction | int:
        """Generate random initial value."""
        value = self._rng.randint(1, self.max_value)
        return Fraction(value) if self.use_fractions else value

    def apply_operator(self, state: Fraction | int, operator: str) -> Fraction | int:
        """Apply arithmetic operation."""
        # Parse operator
        match = re.match(r"(add|sub|mul|div)_(\d+)", operator)
        if not match:
            raise ValueError(f"Unknown operator: {operator}")

        op_type, operand = match.groups()
        operand = int(operand)

        if self.use_fractions:
            state = Fraction(state)

        if op_type == "add":
            result = state + operand
        elif op_type == "sub":
            result = state - operand
        elif op_type == "mul":
            result = state * operand
        elif op_type == "div":
            if operand == 0:
                return state
            result = state / operand if self.use_fractions else state // operand
        else:
            raise ValueError(f"Unknown operation type: {op_type}")

        return result

    def state_equal(self, state1: Any, state2: Any) -> bool:
        """Check value equality."""
        if self.use_fractions:
            return Fraction(state1) == Fraction(state2)
        return int(state1) == int(state2)

    def state_to_string(self, state: Any) -> str:
        """Convert value to string."""
        if self.use_fractions and isinstance(state, Fraction):
            if state.denominator == 1:
                return str(state.numerator)
            return f"{state.numerator}/{state.denominator}"
        return str(int(state))

    def parse_state(self, text: str) -> int | Fraction | None:
        """Parse numerical value from text."""
        # Try fraction format
        frac_match = re.search(r"(-?\d+)/(\d+)", text)
        if frac_match:
            num, denom = int(frac_match.group(1)), int(frac_match.group(2))
            return Fraction(num, denom) if self.use_fractions else num // denom

        # Try integer
        int_match = re.search(r"(?<![/\d])(-?\d+)(?![/\d])", text)
        if int_match:
            value = int(int_match.group(1))
            return Fraction(value) if self.use_fractions else value

        return None

    def format_prompt(
        self,
        initial_state: Any,
        target_state: Any,
        condition: str,
    ) -> tuple[str, str]:
        """Format arithmetic task prompt."""
        init_str = self.state_to_string(initial_state)
        target_str = self.state_to_string(target_state)

        ops_desc = []
        for op in self.operators:
            match = re.match(r"(add|sub|mul|div)_(\d+)", op)
            if match:
                op_type, operand = match.groups()
                symbol = {"add": "+", "sub": "-", "mul": "×", "div": "÷"}[op_type]
                ops_desc.append(f"{op}: {symbol}{operand}")
        ops_str = ", ".join(ops_desc)

        if condition == "C1":
            system_prompt = """You are solving a multi-step arithmetic problem.
Show each step clearly with the operation and resulting value.
Think carefully about each calculation."""

            user_prompt = f"""Transform the number from {init_str} to {target_str}.

Available operations: {ops_str}

Show your work step by step. For each step:
1. State the current value
2. State the operation you're applying
3. State the new value

Continue until you reach {target_str}."""

        elif condition == "C2":
            system_prompt = """You are solving a multi-step arithmetic problem. Show each step with the operation and resulting value, but use exactly the optimal (minimum) number of operations and do not add any extra steps."""

            user_prompt = f"""Transform the number from {init_str} to {target_str} in the minimum number of operations.

Available operations: {ops_str}

Show your work step by step (current value, operation, new value) and stop as soon as you reach {target_str}, using no more than the optimal number of steps."""

        elif condition == "C3":
            system_prompt = """Use the calculator tools to transform the number.
Verify your result after each operation."""

            user_prompt = f"""Transform {init_str} to {target_str}.
Operations: {ops_str}
Use tools to apply operations and verify results."""

        elif condition == "C4":
            system_prompt = """You are solving a multi-step arithmetic problem. Show each step with the operation and resulting value. Take as many steps as you need: there is no limit, so reason for as long as necessary to reach the target."""

            user_prompt = f"""Transform the number from {init_str} to {target_str}.

Available operations: {ops_str}

Take as many steps as you need. For each step, state the current value, the operation, and the new value, continuing until you reach {target_str}."""

        else:
            return self.format_prompt(initial_state, target_state, "C1")

        return user_prompt, system_prompt

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get calculator tool definitions."""
        return [
            {
                "function": {
                    "name": "calculate",
                    "description": "Perform an arithmetic operation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": self.operators,
                                "description": "Operation to perform",
                            },
                        },
                        "required": ["operation"],
                    },
                },
            },
            {
                "function": {
                    "name": "get_current_value",
                    "description": "Get the current numerical value",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "function": {
                    "name": "verify_value",
                    "description": "Verify the current value matches expected",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expected": {
                                "type": "number",
                                "description": "Expected value",
                            },
                        },
                        "required": ["expected"],
                    },
                },
            },
        ]
