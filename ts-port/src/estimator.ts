// Aix-Route — Deterministic Horizon routing (TypeScript port)
// https://github.com/iPatan7/Aix-Route — License: MIT
//
// ============================================================
//  DEPTH ESTIMATOR — task text -> reasoning depth (heuristic)
//
//  A small, explainable, regex-based estimator that maps a free-
//  text task description to an integer reasoning depth, to feed
//  shouldDelegate(). Intentionally NOT machine-learned: a rough,
//  deterministic, auditable heuristic is the right tool for routing
//  a single task to native-vs-tool. Pass an explicit depth when you
//  have a better estimate.
// ============================================================

export const DEPTH_MIN = 1;
export const DEPTH_MAX = 40;

function clampDepth(d: number): number {
  return Math.max(DEPTH_MIN, Math.min(DEPTH_MAX, Math.round(d)));
}

/**
 * Estimate the deterministic reasoning depth of a task description.
 *
 *   "Sort [5,2,8,1,3] using adjacent swaps"  -> 10  (n*(n-1)/2, n=5)
 *   "Sum the list [3,1,4,1,5]"               ->  5  (linear, n=5)
 *   "Plan a trip, then book it, then pack"   ->  derived from chain
 *   "Solve this in 7 steps"                  ->  7  (explicit)
 */
export function estimateDepth(task: string): number {
  if (!task || typeof task !== "string") return DEPTH_MIN;
  const raw = task.trim();
  const t = raw.toLowerCase();

  // 1. Explicit step count wins.
  const stepMatch = t.match(/(\d+)\s*-?\s*step/);
  if (stepMatch) return clampDepth(parseInt(stepMatch[1], 10));

  // 1b. "<n> <step-word>" — e.g. "35 swaps", "20 lines", "12 moves". The
  //     number is an explicit operation count, so its magnitude IS the depth.
  //     (Mirrors Aix-Route's count-word rule; without this, "apply 35 swaps"
  //     under-counts to ~2 and mis-routes an obviously deep task to native.)
  const countMatch = t.match(
    /(\d+)\s+(?:\w+\s+){0,2}?(?:swap|step|move|line|operation|iteration|transposition|hop|transition)s?\b/,
  );
  if (countMatch) return clampDepth(parseInt(countMatch[1], 10));

  // 2. Bracketed list → element count drives algorithmic depth.
  const listMatch = raw.match(/\[([^\]]*)\]/);
  const items = listMatch
    ? listMatch[1].split(",").map((s) => s.trim()).filter(Boolean)
    : [];
  const n = items.length;

  // Quadratic work over a list (sort / adjacent swaps / bubble / all pairs).
  if (n >= 2 && /\b(sort|swap|adjacent|bubble|pairwise|all pairs|compare each)\b/.test(t)) {
    return clampDepth((n * (n - 1)) / 2);
  }
  // Linear pass over a list.
  if (n >= 2 && /\b(sum|count|filter|map|each|every|iterate|loop|reverse|find|scan)\b/.test(t)) {
    return clampDepth(n);
  }

  // 3. General reasoning chain: connectives + imperative verbs + clauses.
  const connectives = (t.match(/\b(then|after|next|finally|followed by|once|subsequently|thereafter)\b/g) || []).length;
  const verbs = (t.match(/\b(compute|calculate|derive|prove|solve|plan|search|lookup|verify|check|compare|combine|merge|transform|translate|summari[sz]e|extract|generate|build|design|analy[sz]e|evaluate|determine|infer|deduce|simulate|optimi[sz]e)\b/g) || []).length;
  const clauses = (t.match(/[.;]/g) || []).length;
  const numbers = (t.match(/\b\d+\b/g) || []).length;
  const words = t.split(/\s+/).filter(Boolean).length;

  const depth =
    1 +
    connectives * 2 +
    verbs +
    Math.min(clauses, 4) +
    Math.min(numbers, 3) +
    Math.floor(words / 25);

  return clampDepth(depth);
}
