"""CLI entry point: ``python -m benchmarks.quick --model gpt-4o --task permutation --n 50``."""

from __future__ import annotations

import argparse
import json
import sys

from benchmarks.quick.runner import run_quick_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.quick",
        description=(
            "Quick validation of a model's Deterministic Horizon d*. "
            "Simulated (free) by default; pass --real to hit the model API."
        ),
    )
    parser.add_argument("--model", default="gpt-4o", help="Model identifier (default: gpt-4o)")
    parser.add_argument(
        "--task",
        default="permutation",
        help="Task name: permutation | fsa | arithmetic (default: permutation)",
    )
    parser.add_argument("--n", type=int, default=50, help="Number of instances (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Accuracy threshold defining d* (default: 0.5)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=3.0,
        help="Allowed |recovered d* - paper d*| to count as a match (default: 3.0)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Evaluate the real model via its API (requires keys, costs money).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the result as a single JSON object instead of a text report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        result = run_quick_benchmark(
            model=args.model,
            task=args.task,
            n=args.n,
            seed=args.seed,
            threshold=args.threshold,
            tolerance=args.tolerance,
            real=args.real,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary())

    # Exit non-zero on mismatch so CI / scripts can gate on it.
    return 0 if result.matches_paper else 1


if __name__ == "__main__":
    raise SystemExit(main())
