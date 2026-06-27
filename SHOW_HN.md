# Show HN: We measured where GPT-4o breaks — and built the kill switch

**Live demo:** https://agent-seal.xyz/depth
**Code (MIT):** https://github.com/iPatan7/Aix-Route

---

AI agents can now reason for 100+ steps. Here's the problem: after about 20, they
hallucinate confidently. The accuracy doesn't trail off gently — it falls off a
cliff, and the model keeps producing fluent, wrong answers the whole way down.

We wanted to know *exactly where* that cliff is for a given model, on a given
provider, so an agent could delegate to a tool **before** it crosses it instead
of after. So we measured it.

## The experiment ($0)

No grant, no cluster. Groq's free tier and a local Ollama box.

We use a permutation-sorting probe (`S_n`) where every instance has a known
BFS-optimal solution depth — so "reason to depth d" is a real, gradeable thing,
not a vibe. We run a model at temperature 0 across a sweep of depths, score exact
final-state match, and fit the accuracy-vs-depth curve.

The whole calibration for one model is a handful of API calls. You can reproduce
any number below for free.

## The math

Accuracy decay isn't linear and isn't a plain exponential. It's
*super-exponential* — a quadratic term in the exponent:

```
P(correct at depth d) ≈ exp( −d·ε₀ − γ·d(d+1) / (2·L_eff) )
```

- `ε₀`    — baseline per-step error
- `γ`     — shared attention-decay constant (0.15)
- `L_eff` — effective decoherence length (~10² steps, *not* the context window)

That super-exponential form fits the data at **R² = 0.96**, versus 0.83 for a
plain exponential and 0.71 for linear. The Deterministic Horizon `d*` is just the
depth where this curve crosses 50% accuracy — solvable in closed form.

## What we found

Two numbers that matter, and the gap between them is the whole point:

**Paper calibration (local / HF serving):**

| Model            | d\*  |
|------------------|------|
| o3-mini          | 31   |
| deepseek-r1      | 29   |
| llama-3.3-70b    | 28   |
| qwen-2.5-72b     | 28   |
| claude opus      | 27   |
| **gpt-4o**       | **22** |
| qwen-2.5-7b      | 19   |

**The same models, served on Groq's free tier:**

| Model                       | d\*   |
|-----------------------------|-------|
| gpt-oss-20b                 | 15.5  |
| gpt-oss-120b                | 15    |
| **llama-3.3-70b-versatile** | **8.2** |
| llama-3.1-8b-instant        | 3.0   |

Read those two tables together. The *same* 70B Llama that holds reasoning to
depth ~28 on a local HF setup collapses to **d\* = 8.2** when you call it through
Groq's free tier (ε₀=0.070, R²=0.75). Quantization, serving config, and prompt
path move the cliff — a lot. The horizon is a property of *how you deploy the
model*, not just the weights.

Which is exactly why hard-coding "GPT-4o is good for ~22 steps" and trusting it
is the bug. You have to measure the model *as you actually serve it*.

## The product: Aix-Route

Open-source routing layer. You give it an estimated reasoning depth for the next
agent step; it decides whether to keep reasoning or delegate to a tool, based on
the calibrated horizon for *your* model.

```python
from deterministic_horizon import should_delegate, should_delegate_batch

should_delegate(estimated_depth=35, model="gpt-4o")          # → True (past 22)
should_delegate_batch([5, 8, 35], model="gpt-4o")            # → [False, False, True]
```

TypeScript port too:

```js
const ar = require('aix-route');
ar.shouldDelegate(35, 'gpt-4o');   // → true
```

It ships with a `calibrate` CLI so you don't inherit our numbers — you measure
your own and paste the `(ε₀, d*)` tuple back in. When the fit is out of regime,
it refuses and tells you to fall back to an empirical crossing depth. It never
invents a horizon.

```bash
export GROQ_API_KEY=gsk-...
aix-route calibrate --model llama-3.3-70b-versatile --depths 3,5,8,10,12,15,18 --n 10
```

## Why we think this is real and not training-specific

Across models from six different orgs, failures correlate at r = 0.81–0.91 on the
*same instances* — they break on the same problems, which says architectural, not
a quirk of one lab's data. Fine-tuning recovers only ~3% of the lost accuracy.
And tool-integrated runs hit 86–94% on the identical tasks where neural CoT sits
at 24–42%, at 4.2–4.7× lower cost per correct solution. The tool isn't a crutch
past the horizon — it's the only thing that works.

## Try it

- **Demo:** https://agent-seal.xyz/depth — see the curve and the horizon for each model
- **Repo:** https://github.com/iPatan7/Aix-Route — star it, read the math
- **Calibrate your own model** — it's free, it's a CLI flag, and you might be
  surprised how low your real horizon is once you measure how you actually serve.

Happy to answer questions on the probe design, the decoherence fit, or the
routing policy.
