# ============================================================
# src/bootstrap_method3.py
#
# - Candidate pool from dead + phantom log-likelihood values.
# - Anchor pool: H anchor values, each the minimum of n_live prior log-likelihoods.
# - For each bootstrap replicate:
#     1. draw one fresh anchor;
#     2. keep the same anchor fixed for S anchored two-sweep updates;
#     3. simulate an independent static shrinkage path;
#     4. compute the bootstrap log-evidence.
#
# This version intentionally removes the older SWISH block-mixing layer from the
# canonical submitted-method path.
# ============================================================

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from bootstrap_common import (
    _logdiffexp,
    build_global_candidate_pool_logL_all,
    compute_base_det_reference,
    compute_logZ_and_ESS_from_logw_and_ell,
)


# ============================================================
# Static NS stochastic shrinkage
# ============================================================

def simulate_stochastic_log_weights_static_ns(
    n_live: int,
    N: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, float]:
    """
    Simulate the static Nested Sampling shrinkage process.

    T_i ~ Beta(n_live, 1)
    X_i = prod_{j<=i} T_j
    w_i = X_{i-1} - X_i

    Returns
    -------
    log_w_dead:
        Array of length N containing log quadrature weights for dead points.
    log_w_tail:
        log remaining prior volume after N removals.
    """
    n_live = int(n_live)
    N = int(N)

    if n_live < 1:
        raise ValueError("n_live must be >= 1.")

    if N < 1:
        raise ValueError("Need at least one dead point.")

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
# Candidate-pool interval sampling
# ============================================================

def _rand_uniform_from_cand_in_interval(
    cand: np.ndarray,
    low: float,
    high: float,
    rng: np.random.RandomState,
    allow_equal_low: bool = True,
    allow_equal_high: bool = True,
) -> Optional[float]:
    """
    Draw uniformly from the empirical candidate pool restricted to an interval.

    The candidate pool must be sorted.
    """
    cand = np.asarray(cand, dtype=float).ravel()

    if cand.size == 0:
        return None

    if not np.isfinite(low):
        lo_idx = 0
    else:
        lo_idx = np.searchsorted(
            cand,
            float(low),
            side="left" if allow_equal_low else "right",
        )

    if not np.isfinite(high):
        hi_idx = cand.size - 1
    else:
        hi_idx = (
            np.searchsorted(
                cand,
                float(high),
                side="right" if allow_equal_high else "left",
            )
            - 1
        )

    if hi_idx < lo_idx or lo_idx >= cand.size or hi_idx < 0:
        return None

    j = rng.randint(lo_idx, hi_idx + 1)
    return float(cand[j])


def _draw_anchor(
    anchor_pool_logL: np.ndarray,
    rng: np.random.RandomState,
) -> float:
    anchor_pool = np.asarray(anchor_pool_logL, dtype=float).ravel()
    anchor_pool = anchor_pool[np.isfinite(anchor_pool)]

    if anchor_pool.size == 0:
        raise ValueError("Anchor pool is empty after filtering finite values.")

    return float(anchor_pool[rng.randint(0, anchor_pool.size)])


def _anchor_crossing_index(anchor: float, prev: np.ndarray) -> int:
    """
    Code-index version of the dissertation's crossing index kappa(a, ell^-).

    prev has indices 0, ..., N-1.
    We extend it by prev[N] = +inf.
    We return the first r in {1, ..., N} such that prev_ext[r] >= anchor.
    """
    prev = np.asarray(prev, dtype=float).ravel()
    N = int(prev.size)

    prev_ext = np.concatenate([prev, np.array([np.inf])])

    for r in range(1, N + 1):
        if prev_ext[r] >= float(anchor):
            return int(r)

    return int(N)


def anchored_two_sweep_update(
    cand: np.ndarray,
    prev: np.ndarray,
    anchor: float,
    rng: np.random.RandomState,
    allow_equal_lower: bool = True,
) -> np.ndarray:
    """
    One anchored empirical two-sweep transition.

    This is the code version of the submitted dissertation mechanism:

    - coordinate 0 is fixed to the sampled anchor;
    - the upper sweep updates left-to-right;
    - the lower sweep updates right-to-left;
    - candidate draws are taken from the empirical dead+phantom pool;
    - the anchor is kept fixed throughout the transition.
    """
    cand = np.asarray(cand, dtype=float).ravel()
    cand = cand[np.isfinite(cand)]
    cand = np.sort(cand)

    prev = np.asarray(prev, dtype=float).ravel()
    N = int(prev.size)

    if N < 2:
        raise ValueError("The ladder must contain at least two entries.")

    if cand.size == 0:
        raise ValueError("Candidate pool is empty.")

    anchor = float(anchor)

    prev_ext = np.concatenate([prev, np.array([np.inf])])
    kappa = _anchor_crossing_index(anchor, prev)

    # ------------------------------------------------------------
    # Upper sweep
    # ------------------------------------------------------------
    ell_tilde = np.empty(N, dtype=float)
    ell_tilde[0] = anchor

    for i in range(1, N):
        low = float(ell_tilde[i - 1])
        high_idx = max(i + 1, kappa)
        high = float(prev_ext[high_idx])

        val = _rand_uniform_from_cand_in_interval(
            cand=cand,
            low=low,
            high=high,
            rng=rng,
            allow_equal_low=bool(allow_equal_lower),
            allow_equal_high=True,
        )

        if val is None:
            raise RuntimeError(
                "Empty candidate interval in upper sweep "
                f"at i={i}: low={low}, high={high}, kappa={kappa}."
            )

        ell_tilde[i] = float(val)

    # ------------------------------------------------------------
    # Lower sweep
    # ------------------------------------------------------------
    ell_plus = ell_tilde.copy()
    ell_plus[0] = anchor
    ell_plus[-1] = ell_tilde[-1]

    for i in range(N - 2, 0, -1):
        low = max(anchor, float(prev[i - 1]))
        high = float(ell_plus[i + 1])

        val = _rand_uniform_from_cand_in_interval(
            cand=cand,
            low=low,
            high=high,
            rng=rng,
            allow_equal_low=True,
            allow_equal_high=bool(allow_equal_lower),
        )

        if val is None:
            raise RuntimeError(
                "Empty candidate interval in lower sweep "
                f"at i={i}: low={low}, high={high}, anchor={anchor}."
            )

        ell_plus[i] = float(val)

    ell_plus[0] = anchor

    if np.any(np.diff(ell_plus) < -1e-12):
        raise RuntimeError("Anchored two-sweep update produced a non-monotone ladder.")

    return ell_plus


# ============================================================
# Dissertation-matching Method 3 driver
# ============================================================

def method3_stochshrink_anchored_two_sweep_chain(
    ns_out: Dict[str, object],
    n_live: int,
    anchor_pool_logL: np.ndarray,
    n_boot: int = 200,
    n_shrink: int = 1,
    seed: Optional[int] = 415,
    S: int = 10,
    pool_decimals: Optional[int] = None,
    allow_equal_lower: bool = True,
    chain_prev: bool = True,
    print_every: int = 0,
) -> Dict[str, object]:
    """
    Submitted-dissertation version of Method 3.

    For each stored bootstrap replicate:
    1. draw one fresh anchor from the anchor pool;
    2. keep that anchor fixed;
    3. apply S anchored two-sweep updates;
    4. simulate n_shrink independent static shrinkage paths.

    The dissertation settings are usually:
        n_boot = 200
        n_shrink = 1
        S = 10
        H = len(anchor_pool_logL) = 100
    """
    rng = np.random.RandomState(seed)

    dead_base = np.asarray(ns_out["dead_logLs"], dtype=float).ravel()
    ell_tail = float(ns_out["tail_logL"])

    N = int(dead_base.size)

    if N < 2:
        raise ValueError("Need at least two dead log-likelihood values.")

    decimals_arg = None if pool_decimals is None else int(pool_decimals)

    cand = build_global_candidate_pool_logL_all(
        ns_out,
        pool_decimals=decimals_arg,
    )

    cand = np.asarray(cand, dtype=float).ravel()
    cand = cand[np.isfinite(cand)]
    cand = np.sort(cand)

    if cand.size == 0:
        raise ValueError("Candidate pool is empty after filtering finite values.")

    anchor_pool = np.asarray(anchor_pool_logL, dtype=float).ravel()
    anchor_pool = anchor_pool[np.isfinite(anchor_pool)]

    if anchor_pool.size == 0:
        raise ValueError("Anchor pool is empty after filtering finite values.")

    base_ref = compute_base_det_reference(ns_out, n_live=int(n_live))

    n_boot = int(n_boot)
    n_shrink = int(n_shrink)
    S = int(S)

    ladders_kept = np.empty((n_boot, N), dtype=float)
    anchors_kept = np.empty(n_boot, dtype=float)

    logZ_kept = np.empty((n_boot, n_shrink), dtype=float)
    ess_dead_kept = np.empty((n_boot, n_shrink), dtype=float)
    ess_dead_tail_kept = np.empty((n_boot, n_shrink), dtype=float)

    prev = dead_base.copy()

    for b in range(n_boot):
        anchor = _draw_anchor(anchor_pool, rng)
        anchors_kept[b] = float(anchor)

        ell = prev.copy()

        for _ in range(S):
            ell = anchored_two_sweep_update(
                cand=cand,
                prev=ell,
                anchor=anchor,
                rng=rng,
                allow_equal_lower=bool(allow_equal_lower),
            )

        ladders_kept[b, :] = ell

        if bool(chain_prev):
            prev = ell
        else:
            prev = dead_base.copy()

        for s in range(n_shrink):
            log_w_dead_s, log_w_tail_s = simulate_stochastic_log_weights_static_ns(
                n_live=int(n_live),
                N=N,
                rng=rng,
            )

            (
                logZ_kept[b, s],
                ess_dead_kept[b, s],
                ess_dead_tail_kept[b, s],
            ) = compute_logZ_and_ESS_from_logw_and_ell(
                log_w_dead=log_w_dead_s,
                ell_dead=ell,
                log_w_tail=log_w_tail_s,
                ell_tail=ell_tail,
            )

        if int(print_every) and ((b + 1) % int(print_every) == 0):
            print(
                f"[M3 anchored {b + 1:4d}/{n_boot}] "
                f"logZ={logZ_kept[b].mean(): .6f}"
            )

    return {
        "logZ": logZ_kept,
        "ESS_dead": ess_dead_kept,
        "ESS_dead_tail": ess_dead_tail_kept,
        "ell_ladders": ladders_kept,
        "anchors": anchors_kept,
        "candidate_pool_size": int(cand.size),
        "anchor_pool_size": int(anchor_pool.size),
        "base": base_ref,
        "settings": {
            "method": "3 dissertation anchored two-sweep ladder bootstrap",
            "n_boot": int(n_boot),
            "n_shrink": int(n_shrink),
            "S": int(S),
            "chain_prev": bool(chain_prev),
            "pool_decimals": pool_decimals,
            "allow_equal_lower": bool(allow_equal_lower),
            "seed": seed,
        },
    }


# ============================================================
# Backward-compatible alias
# ============================================================

def method3_stochshrink_two_sided_chain(
    ns_out: Dict[str, object],
    n_live: int,
    anchor_pool_logL: Optional[np.ndarray] = None,
    n_boot: int = 200,
    n_shrink: int = 1,
    seed: Optional[int] = 415,
    S: int = 10,
    pool_decimals: Optional[int] = None,
    allow_equal_lower: bool = True,
    chain_prev: bool = True,
    print_every: int = 0,
    **unused_kwargs,
) -> Dict[str, object]:
    """
    Backward-compatible wrapper.

    The old function name is retained so notebooks do not immediately break,
    but the canonical implementation is now the submitted-dissertation anchored
    two-sweep bootstrap.

    You must pass anchor_pool_logL.
    """
    if anchor_pool_logL is None:
        raise ValueError(
            "anchor_pool_logL must be supplied. "
            "Use method3_stochshrink_anchored_two_sweep_chain(...) directly."
        )

    return method3_stochshrink_anchored_two_sweep_chain(
        ns_out=ns_out,
        n_live=int(n_live),
        anchor_pool_logL=anchor_pool_logL,
        n_boot=int(n_boot),
        n_shrink=int(n_shrink),
        seed=seed,
        S=int(S),
        pool_decimals=pool_decimals,
        allow_equal_lower=bool(allow_equal_lower),
        chain_prev=bool(chain_prev),
        print_every=int(print_every),
    )
