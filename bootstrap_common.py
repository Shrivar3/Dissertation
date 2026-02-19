# ============================================================
# src/bootstrap_common.py
#
# Shared utilities for NS bootstrap methods:
# - log-sum-exp, log-diff-exp
# - deterministic NS weights (static n_live)
# - compute logZ and ESS
# - candidate pool builder (dead + phantom)
# ============================================================

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple


def _logsumexp_vec(logv: np.ndarray) -> float:
    logv = np.asarray(logv, dtype=float).ravel()
    if logv.size == 0:
        return -np.inf
    m = float(np.max(logv))
    if not np.isfinite(m):  # all -inf
        return -np.inf
    return float(m + np.log(np.sum(np.exp(logv - m))))


def _logdiffexp(a: float, b: float) -> float:
    """log(exp(a) - exp(b)) assuming a>b, else -inf."""
    if b >= a:
        return -np.inf
    return float(a + np.log1p(-np.exp(b - a)))


def deterministic_log_weights_static_ns(n_live: int, N: int) -> Tuple[np.ndarray, float]:
    """
    Deterministic shrinkage (static NS):
      X_i = exp(-i/n_live), i=0..N
      w_i = X_{i-1} - X_i, tail = X_N
    Returns:
      log_w_dead (N,), log_w_tail (scalar)
    """
    n_live = int(n_live)
    N = int(N)
    if n_live <= 0:
        raise ValueError("n_live must be >= 1")
    if N < 0:
        raise ValueError("N must be >= 0")
    X_det = np.exp(-np.arange(N + 1) / float(n_live))
    w_dead = X_det[:-1] - X_det[1:]
    w_tail = X_det[-1]
    log_w_dead = np.log(w_dead)
    log_w_tail = float(np.log(w_tail))
    return log_w_dead, log_w_tail


def compute_logZ_and_ESS_from_logw_and_ell(
    log_w_dead: np.ndarray,
    ell_dead: np.ndarray,
    log_w_tail: float,
    ell_tail: float,
) -> Tuple[float, float, float]:
    """
    Returns:
      logZ_total, ESS_dead, ESS_dead_tail
    """
    log_w_dead = np.asarray(log_w_dead, dtype=float).ravel()
    ell_dead = np.asarray(ell_dead, dtype=float).ravel()
    log_contrib_dead = log_w_dead + ell_dead

    logZ_dead = _logsumexp_vec(log_contrib_dead)
    p_dead = np.exp(log_contrib_dead - logZ_dead)
    ess_dead = 1.0 / np.sum(p_dead ** 2)

    log_contrib_tail = float(log_w_tail) + float(ell_tail)
    log_contrib_all = np.concatenate([log_contrib_dead, [log_contrib_tail]])
    logZ_total = _logsumexp_vec(log_contrib_all)

    p_all = np.exp(log_contrib_all - logZ_total)
    ess_dead_tail = 1.0 / np.sum(p_all ** 2)

    return float(logZ_total), float(ess_dead), float(ess_dead_tail)


def compute_base_det_reference(ns_out: Dict[str, object], n_live: int) -> Dict[str, float | np.ndarray]:
    """
    Deterministic reference for the given ns_out (dead logLs and tail logL).
    Useful as a baseline comparator.
    """
    dead = np.asarray(ns_out["dead_logLs"], dtype=float).ravel()
    N = dead.size
    ell_tail = float(ns_out["tail_logL"])

    log_w_dead, log_w_tail = deterministic_log_weights_static_ns(n_live=n_live, N=N)
    logZ, ess_d, ess_dt = compute_logZ_and_ESS_from_logw_and_ell(
        log_w_dead=log_w_dead,
        ell_dead=dead,
        log_w_tail=log_w_tail,
        ell_tail=ell_tail,
    )
    return {
        "dead_base": dead,
        "ell_tail": ell_tail,
        "logZ_det": float(logZ),
        "ESS_dead_det": float(ess_d),
        "ESS_dead_tail_det": float(ess_dt),
    }


def build_global_candidate_pool_logL_all(ns_out: dict, pool_decimals=None) -> np.ndarray:
    dead = np.asarray(ns_out["dead_logLs"], dtype=float).ravel()
    ph_bins = ns_out.get("phantom_bins_logL", [])
    parts = []
    for a in ph_bins:
        if a is None:
            continue
        arr = np.asarray(a).ravel()
        if arr.size == 0:
            continue
        parts.append(arr.astype(float, copy=False))
    ph = np.concatenate(parts) if parts else np.array([], dtype=float)

    cand = np.concatenate([dead, ph], axis=0)
    cand = cand[np.isfinite(cand)]

    # OPTIONAL: rounding WITHOUT unique (keeps multiplicities)
    if pool_decimals is not None:
        cand = np.round(cand, int(pool_decimals))

    return cand
