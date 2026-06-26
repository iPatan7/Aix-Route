"""Statistical utilities for Deterministic Horizon analysis."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def accuracy_by_depth(
    results: list[dict[str, Any]],
    depth_key: str = "optimal_depth",
    correct_key: str = "correct",
) -> dict[int, dict[str, float]]:
    """
    Compute accuracy statistics grouped by reasoning depth.

    Args:
        results: List of result dictionaries
        depth_key: Key for depth value
        correct_key: Key for correctness boolean

    Returns:
        Dictionary mapping depth to accuracy statistics
    """
    # Group by depth
    by_depth = defaultdict(list)
    for result in results:
        depth = result.get(depth_key, 0)
        correct = result.get(correct_key, False)
        by_depth[depth].append(1 if correct else 0)

    # Compute stats
    stats_by_depth = {}
    for depth, values in sorted(by_depth.items()):
        arr = np.array(values)
        n = len(arr)
        mean = np.mean(arr)

        # Wilson score interval for binomial proportion
        if n > 0:
            z = 1.96  # 95% CI
            denominator = 1 + z**2 / n
            center = (mean + z**2 / (2 * n)) / denominator
            spread = z * np.sqrt((mean * (1 - mean) + z**2 / (4 * n)) / n) / denominator
            ci_low = max(0, center - spread)
            ci_high = min(1, center + spread)
        else:
            ci_low = ci_high = 0.0

        stats_by_depth[depth] = {
            "accuracy": mean,
            "std": np.std(arr),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": n,
        }

    return stats_by_depth


def estimate_horizon(
    results: list[dict[str, Any]],
    threshold: float = 0.5,
    depth_key: str = "optimal_depth",
    correct_key: str = "correct",
) -> dict[str, Any]:
    """
    Estimate the Deterministic Horizon d* where accuracy crosses threshold.

    Uses the super-exponential fit from Theorem 4.2 of the paper. We fit the
    reduced form ``P(correct) = exp(-ε₀·d - g·d(d+1)/2)`` where the fitted slope
    ``g = γ / L_eff`` already absorbs the effective decoherence length; the
    closed-form horizon below is then Theorem 4.5 with L_eff folded into g:
    ``d* = (-ε₀ + sqrt(ε₀² + 2g·ln(1/α))) / g``.

    Args:
        results: List of result dictionaries
        threshold: Accuracy threshold for horizon (default 0.5)
        depth_key: Key for depth value
        correct_key: Key for correctness boolean

    Returns:
        Dictionary with horizon estimate and confidence interval
    """
    acc_by_depth = accuracy_by_depth(results, depth_key, correct_key)

    depths = np.array(sorted(acc_by_depth.keys()))
    accuracies = np.array([acc_by_depth[d]["accuracy"] for d in depths])

    # Fit super-exponential model: acc = exp(-a*d - b*d²)
    def decoherence_model(d, eps0, gamma):
        # Simplified model assuming L >> d
        return np.exp(-eps0 * d - gamma * d * (d + 1) / 2)

    try:
        # Fit model
        popt, pcov = curve_fit(
            decoherence_model,
            depths,
            accuracies,
            p0=[0.02, 0.001],
            bounds=([0, 0], [1, 0.1]),
            maxfev=10000,
        )
        eps0, gamma = popt

        # Compute fitted curve
        fitted = decoherence_model(depths, eps0, gamma)
        r_squared = 1 - np.sum((accuracies - fitted) ** 2) / np.sum(
            (accuracies - np.mean(accuracies)) ** 2
        )

        # Find horizon analytically
        # P = threshold => -eps0*d - gamma*d²/2 = log(threshold)
        # Solving: d* = (-eps0 + sqrt(eps0² - 2*gamma*log(threshold))) / gamma
        log_thresh = np.log(threshold)
        discriminant = eps0**2 - 2 * gamma * log_thresh

        if discriminant >= 0 and gamma > 0:
            d_star = (-eps0 + np.sqrt(discriminant)) / gamma
        else:
            # Fallback: find by interpolation
            d_star = float(np.interp(threshold, accuracies[::-1], depths[::-1]))

        # Bootstrap CI for d_star
        d_star_bootstrap = []
        n_bootstrap = 1000
        rng = np.random.RandomState(42)

        for _ in range(n_bootstrap):
            # Resample results
            indices = rng.choice(len(results), size=len(results), replace=True)
            resampled = [results[i] for i in indices]
            resampled_acc = accuracy_by_depth(resampled, depth_key, correct_key)

            resamp_depths = np.array(sorted(resampled_acc.keys()))
            resamp_accs = np.array([resampled_acc[d]["accuracy"] for d in resamp_depths])

            try:
                popt_boot, _ = curve_fit(
                    decoherence_model,
                    resamp_depths,
                    resamp_accs,
                    p0=[eps0, gamma],
                    bounds=([0, 0], [1, 0.1]),
                    maxfev=5000,
                )
                eps0_b, gamma_b = popt_boot
                disc_b = eps0_b**2 - 2 * gamma_b * log_thresh
                if disc_b >= 0 and gamma_b > 0:
                    d_star_b = (-eps0_b + np.sqrt(disc_b)) / gamma_b
                    d_star_bootstrap.append(d_star_b)
            except (RuntimeError, ValueError):
                continue

        if d_star_bootstrap:
            ci_low = np.percentile(d_star_bootstrap, 2.5)
            ci_high = np.percentile(d_star_bootstrap, 97.5)
        else:
            ci_low = ci_high = d_star

        return {
            "d_star": d_star,
            "d_star_ci_low": ci_low,
            "d_star_ci_high": ci_high,
            "eps0": eps0,
            "gamma": gamma,
            "r_squared": r_squared,
            "threshold": threshold,
        }

    except (RuntimeError, ValueError):
        # Fallback: simple interpolation
        for i, (d, acc) in enumerate(zip(depths, accuracies, strict=False)):
            if acc < threshold:
                if i > 0:
                    # Linear interpolation
                    d_prev, acc_prev = depths[i - 1], accuracies[i - 1]
                    d_star = d_prev + (threshold - acc_prev) * (d - d_prev) / (acc - acc_prev)
                else:
                    d_star = d
                return {
                    "d_star": d_star,
                    "d_star_ci_low": d_star - 2,
                    "d_star_ci_high": d_star + 2,
                    "method": "interpolation",
                    "threshold": threshold,
                }

        return {
            "d_star": float(depths[-1]),
            "method": "lower_bound",
            "threshold": threshold,
        }


def cross_model_correlation(
    results_by_model: dict[str, list[dict[str, Any]]],
    instance_key: str = "instance_id",
    correct_key: str = "correct",
) -> dict[str, Any]:
    """
    Compute cross-model correlation to assess architectural vs training causation.

    High correlation (r > 0.8) suggests architectural cause.
    Low correlation suggests training-specific cause.

    Args:
        results_by_model: Dictionary mapping model name to results
        instance_key: Key for instance identifier
        correct_key: Key for correctness boolean

    Returns:
        Correlation matrix and summary statistics
    """
    models = list(results_by_model.keys())
    n_models = len(models)

    # Build instance-to-correctness mapping for each model
    model_results = {}
    all_instances = set()

    for model, results in results_by_model.items():
        model_results[model] = {r[instance_key]: r[correct_key] for r in results}
        all_instances.update(model_results[model].keys())

    # Find common instances
    common_instances = all_instances.copy()
    for results in model_results.values():
        common_instances &= set(results.keys())

    common_instances = sorted(common_instances)

    # Build correctness vectors
    vectors = {}
    for model in models:
        vectors[model] = np.array(
            [1 if model_results[model].get(inst, False) else 0 for inst in common_instances]
        )

    # Compute correlation matrix
    corr_matrix = np.zeros((n_models, n_models))

    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                r, p = stats.pearsonr(vectors[model_i], vectors[model_j])
                corr_matrix[i, j] = r

    # Summary statistics
    upper_triangle = corr_matrix[np.triu_indices(n_models, k=1)]

    return {
        "correlation_matrix": corr_matrix.tolist(),
        "models": models,
        "mean_correlation": float(np.mean(upper_triangle)),
        "std_correlation": float(np.std(upper_triangle)),
        "min_correlation": float(np.min(upper_triangle)),
        "max_correlation": float(np.max(upper_triangle)),
        "n_common_instances": len(common_instances),
    }


def bootstrap_ci(
    data: Sequence[float],
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: Data values
        statistic: Statistic function (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        seed: Random seed

    Returns:
        (point_estimate, ci_low, ci_high)
    """
    data = np.array(data)
    rng = np.random.RandomState(seed)

    # Point estimate
    point = statistic(data)

    # Bootstrap
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile CI
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return point, ci_low, ci_high


def compute_effect_size(
    group1: Sequence[float],
    group2: Sequence[float],
    method: str = "cohens_d",
) -> dict[str, float]:
    """
    Compute effect size between two groups.

    Args:
        group1: First group values
        group2: Second group values
        method: Effect size method ('cohens_d', 'hedges_g', 'glass_delta')

    Returns:
        Dictionary with effect size and interpretation
    """
    g1 = np.array(group1)
    g2 = np.array(group2)

    n1, n2 = len(g1), len(g2)
    mean1, mean2 = np.mean(g1), np.mean(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)

    if method == "cohens_d":
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    elif method == "hedges_g":
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        # Hedges correction
        correction = 1 - 3 / (4 * (n1 + n2) - 9)
        d = d * correction

    elif method == "glass_delta":
        d = (mean1 - mean2) / np.sqrt(var2) if var2 > 0 else 0

    else:
        raise ValueError(f"Unknown method: {method}")

    # Interpret effect size
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        "effect_size": d,
        "method": method,
        "interpretation": interpretation,
        "n1": n1,
        "n2": n2,
    }


def fit_decoherence_model(
    depths: Sequence[int],
    accuracies: Sequence[float],
    l_eff: float = 150.0,
) -> dict[str, Any]:
    """
    Fit the decoherence model from Theorem 4.2.

    Model: P(correct) = exp(-m*ε₀ - γ*m*(m+1)/(2*L_eff))

    Args:
        depths: Reasoning depths
        accuracies: Accuracy values at each depth
        l_eff: *Effective* decoherence length L_eff (O(10²) steps), the number
            of reasoning steps over which attention keeps usable state
            resolution — NOT the raw context window L=O(10⁵). The paper's
            GPT-4o fit uses L_eff = 150.

    Returns:
        Fitted parameters and goodness of fit
    """
    depths = np.array(depths)
    accuracies = np.array(accuracies)
    L = l_eff

    def model(m, eps0, gamma):
        return np.exp(-m * eps0 - gamma * m * (m + 1) / (2 * L))

    try:
        popt, pcov = curve_fit(
            model, depths, accuracies, p0=[0.02, 0.001], bounds=([0, 0], [1, 1]), maxfev=10000
        )
        eps0, gamma = popt

        # Goodness of fit
        fitted = model(depths, eps0, gamma)
        ss_res = np.sum((accuracies - fitted) ** 2)
        ss_tot = np.sum((accuracies - np.mean(accuracies)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Parameter CIs from covariance
        perr = np.sqrt(np.diag(pcov))

        return {
            "eps0": eps0,
            "eps0_std": perr[0],
            "gamma": gamma,
            "gamma_std": perr[1],
            "r_squared": r_squared,
            "l_eff": L,
            "fitted_values": fitted.tolist(),
        }

    except (RuntimeError, ValueError) as e:
        return {
            "error": str(e),
            "eps0": None,
            "gamma": None,
            "r_squared": None,
        }
