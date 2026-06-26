"""Finite State Automaton simulation task."""

import re
from typing import Any

from deterministic_horizon.tasks.base import BaseTask


class FSATask(BaseTask):
    """
    Finite State Automaton (FSA) simulation task.

    The task involves simulating state transitions in a deterministic
    finite automaton given an input string.
    """

    def __init__(
        self,
        seed: int = 42,
        n_states: int = 8,
        n_symbols: int = 4,
        **kwargs,
    ) -> None:
        self.n_states = n_states
        self.n_symbols = n_symbols

        # Generate random transition table
        super().__init__(seed=seed, n_elements=n_states, **kwargs)

        self._generate_transition_table()

    def _generate_transition_table(self) -> None:
        """Generate random transition table."""
        self.states = [f"S{i}" for i in range(self.n_states)]
        self.symbols = [chr(ord("a") + i) for i in range(self.n_symbols)]

        # Random transitions
        self.transitions = {}
        for state in self.states:
            for symbol in self.symbols:
                next_state = self._rng.choice(self.states)
                self.transitions[(state, symbol)] = next_state

    def default_operators(self) -> list[str]:
        """Return symbols as operators."""
        return [chr(ord("a") + i) for i in range(self.n_symbols)]

    def initial_state(self) -> str:
        """Return initial state."""
        return "S0"

    def apply_operator(self, state: str, operator: str) -> str:
        """Apply symbol transition."""
        return self.transitions.get((state, operator), state)

    def state_equal(self, state1: str, state2: str) -> bool:
        """Check state equality."""
        return state1 == state2

    def state_to_string(self, state: str) -> str:
        """Convert state to string."""
        return state

    def parse_state(self, text: str) -> str | None:
        """Parse state from text."""
        match = re.search(r"S\d+", text)
        if match:
            state = match.group(0)
            if state in self.states:
                return state
        return None

    def format_prompt(
        self,
        initial_state: str,
        target_state: str,
        condition: str,
    ) -> tuple[str, str]:
        """Format prompt for FSA task."""
        # Format transition table
        table_lines = ["Transition Table:"]
        for (state, symbol), next_state in sorted(self.transitions.items()):
            table_lines.append(f"  {state} --{symbol}--> {next_state}")
        table_str = "\n".join(table_lines)

        if condition == "C1":
            system_prompt = """You are simulating a finite state automaton (FSA).
Given the transition table and input sequence, trace through each state transition.
Show the state after each symbol is processed."""

            user_prompt = f"""{table_str}

Starting state: {initial_state}

You need to reach state: {target_state}

Trace through the automaton step by step. For each step, show:
1. Current state
2. Input symbol
3. Next state

Find an input sequence that reaches the target state."""

        elif condition == "C2":
            system_prompt = """You are simulating a finite state automaton (FSA). Trace each transition step by step, but use exactly the optimal (minimum) number of transitions and do not add any extra steps."""

            user_prompt = f"""{table_str}

Starting state: {initial_state}

You need to reach state: {target_state}

Trace through the automaton step by step, showing the current state, input symbol, and next state at each step, using the minimum number of transitions needed to reach the target."""

        elif condition == "C3":
            system_prompt = """Simulate the FSA using the verification tools.
Use step_fsa to process each symbol and verify_state to check your position."""

            user_prompt = f"""{table_str}

Starting state: {initial_state}
Target state: {target_state}

Use tools to navigate to the target state."""

        elif condition == "C4":
            system_prompt = """You are simulating a finite state automaton (FSA). Trace each transition step by step, showing the state after each symbol. Take as many steps as you need: there is no limit, so reason for as long as necessary to reach the target."""

            user_prompt = f"""{table_str}

Starting state: {initial_state}

You need to reach state: {target_state}

Take as many steps as you need. For each step, show the current state, the input symbol, and the next state, continuing until you reach the target state."""

        else:
            return self.format_prompt(initial_state, target_state, "C1")

        return user_prompt, system_prompt

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions for C3."""
        return [
            {
                "function": {
                    "name": "step_fsa",
                    "description": "Process one input symbol in the FSA",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "enum": self.symbols,
                                "description": "Input symbol to process",
                            },
                        },
                        "required": ["symbol"],
                    },
                },
            },
            {
                "function": {
                    "name": "get_current_state",
                    "description": "Get the current FSA state",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "function": {
                    "name": "check_target_reached",
                    "description": "Check if current state is the target",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
