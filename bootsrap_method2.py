# ============================================================
# SRC/bootstrap_method2.py
#
# Method (2): LADDER-ONLY bootstrap (two-sided ladder chain + optional SWISH)
# - Generates ladders exactly like Method (3)
# - Uses DETERMINISTIC NS weights (no stochastic shrinkage)
# - Computes logZ + ESS per ladder
# ============================================================

from __future__ import annotations

import numpy as np
import time
from typing import Dict, Optional

from bootstrap_common import (
    deterministic_log_weights_static_ns,
    compute_logZ_and_ESS_from_logw_and_ell,
    compute_base_det_reference,
    build_global_candidate_pool_logL_unique,
)

# Reuse ladder construction + SWISH from Method (3) module
from bootstrap_method3 import (
    forward_sweep_upper_coupled,
    backward_sweep_lower_coupled,
    swish_block_mix,
)


def method2_detworks_two_sided_chain(
    ns_out: Dict[str, object],
    n_live: int,
    n_boot: int = 2000,
    seed: Optional[int] = 415,

    # candidate pool de-dup
    pool_mode: str = "tol",      # "tol", "round", "exact"
    pool_decimals: int = 14,     # if pool_mode="round"
    pool_tol: float = 1e-12,     # if pool_mode="tol"

    allow_equal_lower: bool = True,
    chain_prev: bool = True,

    # mixing
    use_swish: bool = True,
    swish_blocks: int = 40,
    swish_mean_block_len: int = 80,
    swish_site_sweeps_per_block: int = 2,
    swish_p_upper: float = 0.5,

    # optional thinning/lagging
    lag_every: int = 10,

    # prints
    print_every: int = 100,
) -> Dict[str, object]:
    """
    Inputs:
      ns_out must contain:
        - "dead_logLs": (N,) array
        - "tail_logL":  float
      optional:
        - "phantom_bins_logL": list of arrays

    Output:
      logZ: (n_boot,)
      ESS_dead: (n_boot,)
      ESS_dead_tail: (n_boot,)
      ell_ladders: (n_boot, N)
      candidate_pool_size: int
      base: deterministic baseline
      settings: dict
    """
    rng = np.random.RandomState(seed)

    dead_base = np.asarray(ns_out["dead_logLs"], dtype=float).ravel()
    ell_tail = float(ns_out["tail_logL"])
    N = int(dead_base.size)

    if int(lag_every) < 1:
        raise ValueError("lag_every must be >= 1.")

    cand = build_global_candidate_pool_logL_unique(
        ns_out, mode=str(pool_mode), decimals=int(pool_decimals), tol=float(pool_tol)
    )
    base_ref = compute_base_det_reference(ns_out, n_live=int(n_live))

    # deterministic weights (same for all ladders)
    log_w_dead_det, log_w_tail_det = deterministic_log_weights_static_ns(n_live=int(n_live), N=N)

    # generate n_boot_total steps but keep every lag_every-th ladder
    n_boot_total = int(n_boot) * int(lag_every)

    ladders_kept = np.empty((int(n_boot), N), dtype=float)
    logZ = np.empty(int(n_boot), dtype=float)
    ESS_dead = np.empty(int(n_boot), dtype=float)
    ESS_dead_tail = np.empty(int(n_boot), dtype=float)

    prev = dead_base.copy()
    kept = 0

    for t in range(int(n_boot_total)):
        t0 = time.perf_counter()

        # --- ladder update (same as Method 3)
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

        prev = ell if bool(chain_prev) else dead_base

        if ((t + 1) % int(lag_every)) != 0:
            continue

        ladders_kept[kept, :] = ell

        # deterministic weights only
        logZ[kept], ESS_dead[kept], ESS_dead_tail[kept] = compute_logZ_and_ESS_from_logw_and_ell(
            log_w_dead=log_w_dead_det,
            ell_dead=ell,
            log_w_tail=log_w_tail_det,
            ell_tail=ell_tail,
        )

        kept += 1

        t1 = time.perf_counter()
        if int(print_every) and (kept % int(print_every) == 0):
            print(f"[M2 kept {kept:4d}/{int(n_boot)}] "
                  f"logZ={logZ[kept-1]: .6f}  ESS(dt)={ESS_dead_tail[kept-1]: .1f}  "
                  f"time={t1-t0: .3f}s")

        if kept >= int(n_boot):
            break

    return {
        "logZ": logZ,
        "ESS_dead": ESS_dead,
        "ESS_dead_tail": ESS_dead_tail,
        "ell_ladders": ladders_kept,
        "candidate_pool_size": int(cand.size),
        "base": base_ref,
        "settings": {
            "method": "2 (two-sided ladder chain, deterministic weights only)",
            "n_boot": int(n_boot),
            "lag_every": int(lag_every),
            "use_swish": bool(use_swish),
            "swish_blocks": int(swish_blocks),
            "swish_mean_block_len": int(swish_mean_block_len),
            "swish_site_sweeps_per_block": int(swish_site_sweeps_per_block),
            "swish_p_upper": float(swish_p_upper),
            "chain_prev": bool(chain_prev),
            "pool_mode": str(pool_mode),
            "pool_decimals": int(pool_decimals),
            "pool_tol": float(pool_tol),
            "seed": seed,
            "print_every": int(print_every),
        },
    }
