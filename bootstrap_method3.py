# ============================================================
# SRC/bootstrap_method3.py
#
# Method (3): Stochastic shrinkage + two-sided "level-chain" ladder bootstrap
# - Builds ladders ell^(b) using two-sided coupling (+ optional SWISH)
# - For EACH ladder b, simulates MULTIPLE stochastic shrinkage paths (n_shrink)
# - Computes logZ + ESS for each (b, s) pair
#
# Requires:
#   ns_out: dict with keys:
#       "dead_logLs": (N,) array
#       "tail_logL":  float
#       optionally "phantom_bins_logL": list of arrays
#   n_live: int
# ============================================================

from __future__ import annotations
import numpy as np
import time
from typing import Dict, Optional, Tuple

from .bootstrap_common import (
    _logdiffexp,
    deterministic_log_weights_static_ns,
    compute_logZ_and_ESS_from_logw_and_ell,
    compute_base_det_reference,
    build_global_candidate_pool_logL_all,
)


# ============================================================
# Stochastic shrinkage (static NS): simulate log_w per replicate
# ============================================================

def simulate_stochastic_log_weights_static_ns(
    n_live: int,
    N: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, float]:
    """
    t_i ~ Beta(n_live, 1), logX_0=0, logX_i = sum_{j<=i} log t_j
    w_i = X_{i-1} - X_i, tail = X_N
    Return log_w_dead (N,), log_w_tail (scalar).
    """
    n_live = int(n_live)
    N = int(N)
    if int(n_live) < 1:
        raise ValueError("n_live must be >= 1.")
    if N < 1:
        raise ValueError("Need at least 1 dead point (N>=1).")

    t = rng.beta(float(n_live), 1.0, size=N)
    t = np.clip(t, 1e-300, 1.0)
    logt = np.log(t)

    logX = np.empty(N + 1, dtype=float)
    logX[0] = 0.0
    logX[1:] = np.cumsum(logt)

    log_w_dead = np.empty(N, dtype=float)
    for i in range(1, N + 1):
        log_w_dead[i - 1] = _logdiffexp(float(logX[i - 1]), float(logX[i]))

    log_w_tail = float(logX[N])
    return log_w_dead, log_w_tail


# ============================================================
# Core uniform draws with TWO-SIDED coupling
# ============================================================

def _rand_uniform_from_cand_in_interval(
    cand: np.ndarray,
    low: float,
    high: float,
    rng: np.random.RandomState,
    allow_equal_low: bool = True,
    allow_equal_high: bool = True,
) -> Optional[float]:
    cand = np.asarray(cand, dtype=float)

    if not np.isfinite(low):
        lo_idx = 0
    else:
        lo_idx = np.searchsorted(cand, low, side="left" if allow_equal_low else "right")

    if not np.isfinite(high):
        hi_idx = cand.size - 1
    else:
        hi_idx = np.searchsorted(cand, high, side="right" if allow_equal_high else "left") - 1

    if hi_idx < lo_idx or lo_idx >= cand.size or hi_idx < 0:
        return None

    j = rng.randint(lo_idx, hi_idx + 1)
    return float(cand[j])


def forward_sweep_upper_coupled(
    cand: np.ndarray,
    prev: np.ndarray,
    rng: np.random.RandomState,
    allow_equal_lower: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    prev = np.asarray(prev, dtype=float).ravel()
    N = prev.size
    ell = np.empty(N, dtype=float)

    last = -np.inf
    for i in range(N):
        low = last
        high = float(prev[i + 1]) if i < N - 1 else np.inf

        val = _rand_uniform_from_cand_in_interval(
            cand, low, high, rng,
            allow_equal_low=allow_equal_lower,
            allow_equal_high=True
        )
        if val is None:
            val = float(prev[i])
            if allow_equal_lower:
                if val < low:
                    val = low
            else:
                if val <= low:
                    val = low + eps
            if np.isfinite(high) and val > high:
                val = high

        ell[i] = val
        last = val

    return ell


def backward_sweep_lower_coupled(
    cand: np.ndarray,
    prev: np.ndarray,
    ell_init: np.ndarray,
    rng: np.random.RandomState,
    allow_equal_lower: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    prev = np.asarray(prev, dtype=float).ravel()
    ell = np.asarray(ell_init, dtype=float).ravel().copy()
    N = prev.size

    for i in range(N - 1, -1, -1):
        low = float(prev[i - 1]) if i > 0 else -np.inf
        high = float(ell[i + 1]) if i < N - 1 else np.inf

        val = _rand_uniform_from_cand_in_interval(
            cand, low, high, rng,
            allow_equal_low=True,
            allow_equal_high=allow_equal_lower
        )
        if val is None:
            val = float(prev[i])
            if val < low:
                val = low
            if np.isfinite(high) and val > high:
                val = high
            if (not allow_equal_lower) and (i < N - 1) and (val >= high):
                val = min(high - eps, val)

        ell[i] = val

    ell = np.maximum.accumulate(ell)
    return ell


# ============================================================
# FAST "SWISH" block mixing (ladder-space)
# ============================================================

def swish_block_mix(
    cand: np.ndarray,
    prev: np.ndarray,
    ell: np.ndarray,
    rng: np.random.RandomState,
    n_blocks: int = 80,
    mean_block_len: int = 120,
    n_site_sweeps_per_block: int = 2,
    allow_equal_lower: bool = True,
    eps: float = 1e-12,
    p_use_upper_coupling: float = 0.5,
) -> np.ndarray:
    prev = np.asarray(prev, dtype=float).ravel()
    ell = np.asarray(ell, dtype=float).ravel().copy()
    N = ell.size
    ell = np.maximum.accumulate(ell)

    for _ in range(int(n_blocks)):
        L = max(2, int(rng.exponential(scale=max(2.0, float(mean_block_len)))))
        a = int(rng.randint(0, N))
        b = int(min(N - 1, a + L - 1))
        use_upper = (rng.rand() < float(p_use_upper_coupling))

        idxs = np.arange(a, b + 1)
        for _ in range(int(n_site_sweeps_per_block)):
            rng.shuffle(idxs)
            for i in idxs:
                left = float(ell[i - 1]) if i > 0 else -np.inf
                right = float(ell[i + 1]) if i < N - 1 else np.inf

                if use_upper:
                    ub = float(prev[i + 1]) if i < N - 1 else np.inf
                    low, high = left, min(right, ub)
                else:
                    lb = float(prev[i - 1]) if i > 0 else -np.inf
                    low, high = max(left, lb), right

                val = _rand_uniform_from_cand_in_interval(
                    cand, low, high, rng,
                    allow_equal_low=True,
                    allow_equal_high=allow_equal_lower
                )
                if val is None:
                    continue

                ell[i] = val
                if i > 0 and ell[i] < ell[i - 1]:
                    ell[i] = ell[i - 1]
                if i < N - 1 and ell[i] > ell[i + 1]:
                    ell[i] = ell[i + 1]
                if (not allow_equal_lower) and i > 0 and ell[i] <= ell[i - 1]:
                    ell[i] = ell[i - 1] + eps
                    if i < N - 1:
                        ell[i] = min(ell[i], ell[i + 1])

        ell = np.maximum.accumulate(ell)

    return ell


# ============================================================
# METHOD (3) driver
# ============================================================

def method3_stochshrink_two_sided_chain(
    ns_out: Dict[str, object],
    n_live: int,
    n_boot: int = 2000,
    n_shrink: int = 5,
    seed: Optional[int] = 415,

    # candidate pool
    pool_decimals: Optional[int] = None,

    allow_equal_lower: bool = True,
    chain_prev: bool = True,

    # mixing
    use_swish: bool = True,
    swish_blocks: int = 40,
    swish_mean_block_len: int = 80,
    swish_site_sweeps_per_block: int = 2,
    swish_p_upper: float = 0.5,

    # optional thinning/lagging of the ladder chain
    lag_every: int = 10,

    # prints
    print_every: int = 100,
) -> Dict[str, object]:
    rng = np.random.RandomState(seed)

    dead_base = np.asarray(ns_out["dead_logLs"], dtype=float).ravel()
    ell_tail = float(ns_out["tail_logL"])
    N = int(dead_base.size)

    if int(lag_every) < 1:
        raise ValueError("lag_every must be >= 1.")
    
    decimals_arg = None if pool_decimals is None else int(pool_decimals)
        
    cand = build_global_candidate_pool_logL_all(
        ns_out, pool_decimals=decimals_arg
    )
    cand = cand[np.isfinite(cand)]
    cand = np.sort(cand)   # CRITICAL for searchsorted correctness
    if cand.size == 0:
        raise ValueError("Candidate pool is empty after filtering finite values.")
    base_ref = compute_base_det_reference(ns_out, n_live=int(n_live))

    # Generate n_boot_total ladders but only keep every lag_every-th.
    n_boot_total = int(n_boot) * int(lag_every)

    ladders_kept = np.empty((int(n_boot), N), dtype=float)
    logZ_kept = np.empty((int(n_boot), int(n_shrink)), dtype=float)
    ess_dead_kept = np.empty((int(n_boot), int(n_shrink)), dtype=float)
    ess_dead_tail_kept = np.empty((int(n_boot), int(n_shrink)), dtype=float)

    prev = dead_base.copy()
    kept = 0

    for t in range(int(n_boot_total)):
        t0 = time.perf_counter()

        # --- build ladder (one step of the chain)
        ell = forward_sweep_upper_coupled(
            cand=cand, prev=prev, rng=rng,
            allow_equal_lower=bool(allow_equal_lower)
        )
        ell = backward_sweep_lower_coupled(
            cand=cand, prev=prev, ell_init=ell, rng=rng,
            allow_equal_lower=bool(allow_equal_lower)
        )
        if bool(use_swish):
            ell = swish_block_mix(
                cand=cand, prev=prev, ell=ell, rng=rng,
                n_blocks=int(swish_blocks),
                mean_block_len=int(swish_mean_block_len),
                n_site_sweeps_per_block=int(swish_site_sweeps_per_block),
                allow_equal_lower=bool(allow_equal_lower),
                p_use_upper_coupling=float(swish_p_upper)
            )

        # update chain source regardless of lag
        prev = ell if bool(chain_prev) else dead_base

        # only keep every lag_every-th ladder
        if ((t + 1) % int(lag_every)) != 0:
            continue

        ladders_kept[kept, :] = ell

        # --- MULTIPLE stochastic shrinkage paths for the SAME ladder
        for s in range(int(n_shrink)):
            log_w_dead_s, log_w_tail_s = simulate_stochastic_log_weights_static_ns(
                n_live=int(n_live), N=N, rng=rng
            )
            logZ_kept[kept, s], ess_dead_kept[kept, s], ess_dead_tail_kept[kept, s] = compute_logZ_and_ESS_from_logw_and_ell(
                log_w_dead=log_w_dead_s,
                ell_dead=ell,
                log_w_tail=log_w_tail_s,
                ell_tail=ell_tail
            )

        kept += 1

        t1 = time.perf_counter()
        if int(print_every) and (kept % int(print_every) == 0):
            print(f"[M3 kept {kept:4d}/{int(n_boot)}] "
                  f"logZ(mean over shrink)={logZ_kept[kept-1].mean(): .6f}  "
                  f"ESS(dt, mean)={ess_dead_tail_kept[kept-1].mean(): .1f}  "
                  f"time={t1-t0: .3f}s")

        if kept >= int(n_boot):
            break

    return {
        "logZ": logZ_kept,                        # (n_boot, n_shrink)
        "ESS_dead": ess_dead_kept,                # (n_boot, n_shrink)
        "ESS_dead_tail": ess_dead_tail_kept,      # (n_boot, n_shrink)
        "ell_ladders": ladders_kept,              # (n_boot, N)
        "candidate_pool_size": int(cand.size),
        "base": base_ref,
        "settings": {
            "method": "3 (stochastic shrinkage + two-sided ladder chain)",
            "n_boot": int(n_boot),
            "n_shrink": int(n_shrink),
            "lag_every": int(lag_every),
            "use_swish": bool(use_swish),
            "swish_blocks": int(swish_blocks),
            "swish_mean_block_len": int(swish_mean_block_len),
            "swish_site_sweeps_per_block": int(swish_site_sweeps_per_block),
            "swish_p_upper": float(swish_p_upper),
            "chain_prev": bool(chain_prev),
            "pool_decimals": pool_decimals,
            "seed": seed,
        }
    }
