#!/usr/bin/env python3
"""Measure accuracy-vs-depth on Groq models and fit the horizon."""

from __future__ import annotations

import json
import sys

from deterministic_horizon.calibrate import fit_horizon
from deterministic_horizon.models import load_model
from deterministic_horizon.tasks.permutation import PermutationTask

MODEL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
DEPTHS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [3, 5, 8, 10, 12, 15, 18]

m = load_model(MODEL, max_tokens=2048, temperature=0.0)
task = PermutationTask(n_elements=8, seed=42)

depths, accs = [], []
print(f"# model={MODEL} n={N}")
for depth in DEPTHS:
    if depth > task.max_depth():
        continue
    insts = [task.generate_instance(depth) for _ in range(N)]
    correct = 0
    for inst in insts:
        resp = m.generate(inst.prompt, system_prompt=inst.system_prompt)
        correct += int(task.evaluate(inst, resp.content).correct)
    acc = correct / len(insts)
    depths.append(depth)
    accs.append(acc)
    print(f"depth {depth:>2}: acc {acc:.2f} ({correct}/{len(insts)})", flush=True)

fit = fit_horizon(depths, accs)
out = {"model": MODEL, "depths": depths, "accuracies": accs, "fit": fit}
print("RESULT_JSON " + json.dumps(out))
if fit is None:
    print(f"FIT: None (out of regime — flag {MODEL})")
else:
    print(f"FIT: eps0={fit['eps0']:.4f} d*={fit['d_star']:.2f} "
          f"L_eff={fit['L_eff']:.2f} R2={fit['r_squared']:.3f} "
          f"constraint_ok={fit['constraint_ok']} n_pts={fit['n_points']}")
