"""Shared semantic noise model (stage 2), used by every backend so all policies see the same random numbers.

The label of one instance in one distance bin is drawn once from np.random.default_rng([seed, key, bin]):
correct with probability p_bins[bin], otherwise uniform over the other categories. Repeating an observation in
the same bin therefore returns the same label, and AgentMap fuses each (instance, bin) at most once.
p_bins defaults to p(d) = clip(0.95 - 0.1 d, 0.5, 0.95) evaluated at each bin's far edge (1.5, 3.0, 4.0 m), the
worst case inside the bin: one close look (0.8) clears tau = 0.7, one mid or far look (0.65, 0.55) does not.
"""
from __future__ import annotations

import functools

import numpy as np

BIN_EDGES = (1.5, 3.0)
BIN_FAR = (1.5, 3.0, 4.0)


def p_of_d(d: float) -> float:
    return float(np.clip(0.95 - 0.1 * d, 0.5, 0.95))


DEFAULT_P_BINS = tuple(round(p_of_d(d), 4) for d in BIN_FAR)  # (0.8, 0.65, 0.55)


def dist_bin(d: float) -> int:
    return 0 if d < BIN_EDGES[0] else 1 if d < BIN_EDGES[1] else 2


def noisy_label(seed: int, key: int, bin_: int, true_cat: int, n_cat: int, p_bins=DEFAULT_P_BINS, kernel_cdf=None) -> int:
    """kernel_cdf None: a wrong label is uniform over the other classes. Otherwise it is drawn from row true_cat of the
    wrong-label kernel (cumulative), with the second number of the same seeded stream."""
    rng = np.random.default_rng([seed, key, bin_])
    if rng.random() < p_bins[bin_]:
        return true_cat
    if kernel_cdf is not None:
        row = kernel_cdf[true_cat]
        return int(min(np.searchsorted(row, rng.random() * row[-1], side="right"), n_cat - 1))
    j = int(rng.integers(n_cat - 1))
    return j if j < true_cat else j + 1


def frequency_preserving_kernel(prior, cap=0.45, tol=1e-12, iters=100000) -> np.ndarray:
    """Wrong-label kernel K (row: true class, column: reported label) with K[c, c] = 0, rows summing to 1 and pi K = pi.
    With the agent's prior equal to pi, one look in bin b then leaves the reported class at posterior exactly p_b, for
    every class: a mid or far look never clears tau on its own, a near look always does. Common classes absorb most
    wrong labels, rare ones (most task classes) few. Built as the symmetric maximum-entropy coupling F[c, y] = v_c v_y
    (c != y) with row sums pi_c (symmetric Sinkhorn), K = F / pi. It exists only if max(pi) < 1/2: a larger class is
    capped at `cap` for the construction (pi K = pi then holds only approximately; see single_look_posteriors)."""
    pi = np.asarray(prior, dtype=np.float64)
    if pi.ndim != 1 or len(pi) < 3 or not np.all(np.isfinite(pi)) or np.any(pi <= 0):
        raise ValueError("prior must be a finite, strictly positive vector over >= 3 classes")
    pi = pi / pi.sum()
    q = pi.copy()
    i = int(np.argmax(q))
    if q[i] >= 0.5:
        q[np.arange(len(q)) != i] *= (1.0 - cap) / (1.0 - q[i])
        q[i] = cap
        if q.max() >= 0.5:   # capping the largest class pushed another past 1/2: no such kernel
            raise ValueError("prior too concentrated for a frequency-preserving kernel")
    v = np.sqrt(q)
    for _ in range(iters):
        row = v * (v.sum() - v)
        if np.max(np.abs(row / q - 1.0)) < tol:
            break
        v = np.sqrt(v * q / (v.sum() - v))
    else:
        raise ValueError("frequency-preserving kernel did not converge")
    F = np.outer(v, v)
    np.fill_diagonal(F, 0.0)
    K = F / F.sum(axis=1, keepdims=True)
    if not (np.all(np.isfinite(K)) and np.all(K >= 0) and np.allclose(q @ K, q, rtol=0, atol=1e-9)):
        raise ValueError("frequency-preserving kernel failed validation")
    return K


def single_look_posteriors(prior, kernel, p_bins) -> np.ndarray:
    """(n_bins, n_cat): posterior of the reported class after one look from the prior, for each reported class."""
    pi = np.asarray(prior, dtype=np.float64) / np.sum(prior)
    out = []
    for p in p_bins:
        L = p * np.eye(len(pi)) + (1.0 - p) * kernel     # L[c, y] = P(report y | true c)
        out.append(pi * p / (pi @ L))
    return np.array(out)


@functools.lru_cache(maxsize=8)
def _prior_file(path: str) -> tuple:
    import json
    return tuple(json.load(open(path))["prior"])


def noise_model(sensor_cfg: dict):
    """(prior, kernel) for sensor.confusion == "prior" (prior read from sensor.prior_file, train split only), else
    (None, None) for the uniform model. Generator and agent take the same pair."""
    mode = sensor_cfg.get("confusion", "uniform")
    if mode == "uniform":
        return None, None
    if mode != "prior":
        raise ValueError(f"sensor.confusion must be 'uniform' or 'prior', got {mode!r}")
    prior = np.array(_prior_file(sensor_cfg["prior_file"]))
    from .contracts import CATEGORIES
    if len(prior) != len(CATEGORIES):
        raise ValueError(f"{sensor_cfg['prior_file']} has {len(prior)} classes, CATEGORIES has {len(CATEGORIES)}: recompute the prior")
    return prior / prior.sum(), frequency_preserving_kernel(prior)
