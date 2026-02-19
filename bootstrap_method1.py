# ============================================================
# SRC/bootstrap_method1.py
#
# Method (1): WEIGHTS-ONLY bootstrap (stochastic shrinkage weights)
# - Ladder is FIXED: ell_dead = ns_out["dead_logLs"] (no resampling)
# - Each bootstrap replicate draws stochastic shrinkage weights:
#       t_i ~ Beta(n_live, 1)
# - Computes logZ + ESS for each replicate
# ============================================================

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

from bootstrap_common import compute_base_det_reference, compute_logZ_and_ESS_from_logw_and_ell, _logdiffexp


def simulate_stochastic_log_weights_static_ns(
    n_live: int,
    N: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, float]:
    """
    Stochastic shrinkage for static NS:
      t_i ~ Beta(n_live, 1)
      logX_0 = 0
      logX_i = sum_{j<=i} log t_j
      w_i = X_{i-1} - X_i, tail = X_N
    Returns:
      log_w_dead (N,), log_w_tail (scalar).
    """
    n_live = int(n_live)
    N = int(N)

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


def method1_stochweights_fixed_ladder(
    ns_out: Dict[str, object],
    n_live: int,
    n_boot: int = 2000,
    seed: Optional[int] = 415,
    print_every: int = 0,
) -> Dict[str, object]:
    """
    Inputs:
      ns_out must contain:
        - "dead_logLs": (N,) array
        - "tail_logL":  float

    Output:
      logZ: (n_boot,)
      ESS_dead: (n_boot,)
      ESS_dead_tail: (n_boot,)
      base: deterministic baseline (for reference)
      settings: dict
    """
    rng = np.random.RandomState(seed)

    ell_dead = np.asarray(ns_out["dead_logLs"], dtype=float).ravel()
    ell_tail = float(ns_out["tail_logL"])
    N = int(ell_dead.size)

    base_ref = compute_base_det_reference(ns_out, n_live=int(n_live))

    logZ = np.empty(int(n_boot), dtype=float)
    ESS_dead = np.empty(int(n_boot), dtype=float)
    ESS_dead_tail = np.empty(int(n_boot), dtype=float)

    for b in range(int(n_boot)):
        log_w_dead, log_w_tail = simulate_stochastic_log_weights_static_ns(
            n_live=int(n_live), N=N, rng=rng
        )
        logZ[b], ESS_dead[b], ESS_dead_tail[b] = compute_logZ_and_ESS_from_logw_and_ell(
            log_w_dead=log_w_dead,
            ell_dead=ell_dead,
            log_w_tail=log_w_tail,
            ell_tail=ell_tail,
        )

        if int(print_every) and ((b + 1) % int(print_every) == 0):
            print(f"[M1 {b+1:5d}/{int(n_boot)}] logZ={logZ[b]: .6f}  ESS(dt)={ESS_dead_tail[b]: .1f}")

    return {
        "logZ": logZ,
        "ESS_dead": ESS_dead,
        "ESS_dead_tail": ESS_dead_tail,
        "base": base_ref,
        "settings": {
            "method": "1 (fixed ladder, stochastic shrinkage weights only)",
            "n_boot": int(n_boot),
            "seed": seed,
            "print_every": int(print_every),
        },
    }
