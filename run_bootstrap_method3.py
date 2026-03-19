from __future__ import annotations


import os

import sys

import time

from pathlib import Path

from typing import List


import numpy as np


# ============================================================

# PATH SETUP

# ============================================================


REPO_ROOT = Path(__file__).resolve().parent

sys.path.append(str(REPO_ROOT / "src"))


from ns_mh_phantom import run_ns_mh_phantom

from bootstrap_common import (

    _logdiffexp,

    compute_logZ_and_ESS_from_logw_and_ell,

    compute_base_det_reference,

    build_global_candidate_pool_logL_all,

)


# ============================================================

# TIMER

# ============================================================


GLOBAL_START = time.perf_counter()



def format_time(seconds: float) -> str:

    seconds = int(seconds)

    d = seconds // 86400

    h = (seconds % 86400) // 3600

    m = (seconds % 3600) // 60

    s = seconds % 60

    if d > 0:

        return f"{d}d {h:02d}:{m:02d}:{s:02d}"

    return f"{h:02d}:{m:02d}:{s:02d}"



def eta_str(elapsed: float, done: int, total: int) -> str:

    if done <= 0:

        return "??:??:??"

    per = elapsed / done

    rem = per * (total - done)

    return format_time(rem)



# ============================================================

# ENV-DRIVEN CONFIG

# ============================================================


LABEL = os.environ.get("LABEL", "nl100_m20_w20")


# Family-level seed bases

BOOT_RUNS = int(os.environ.get("BOOT_RUNS", "35"))  # retained for compatibility / info

DATA_SEED = int(os.environ.get("DATA_SEED", "415"))

NS_BASE_SEED = int(os.environ.get("NS_BASE_SEED", "1000"))

BOOT_BASE_SEED = int(os.environ.get("BOOT_BASE_SEED", "2000"))


# Chunking

CHUNK_ID = int(os.environ.get("CHUNK_ID", "0"))

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", str(BOOT_RUNS)))

RUN_START = CHUNK_ID * CHUNK_SIZE

RUN_STOP = RUN_START + CHUNK_SIZE

RUN_COUNT = RUN_STOP - RUN_START


# Bootstrap controls

N_BOOT_PER_RUN = int(os.environ.get("N_BOOT_PER_RUN", "200"))

N_SHRINK = int(os.environ.get("N_SHRINK", "1"))


# Lowest-pool controls

LOWEST_POOL_N = int(os.environ.get("LOWEST_POOL_N", "1000"))

LOWEST_BLOCK_SIZE = int(os.environ.get("LOWEST_BLOCK_SIZE", "100"))

LOWEST_POOL_SEED = int(os.environ.get("LOWEST_POOL_SEED", "415"))


# Fixed NS seed used only to build the lowest pool for the whole family

LOWEST_POOL_NS_SEED = int(os.environ.get("LOWEST_POOL_NS_SEED", str(NS_BASE_SEED)))


# Data / model config

n = int(os.environ.get("N", "600"))

p = int(os.environ.get("P", "12"))

use_correlated_X = os.environ.get("USE_CORRELATED_X", "False").lower() == "true"

rho = float(os.environ.get("RHO", "1.0"))

sigma_beta = float(os.environ.get("SIGMA_BETA", "1.0"))

sparsity = float(os.environ.get("SPARSITY", "0.0"))

include_intercept = os.environ.get("INCLUDE_INTERCEPT", "False").lower() == "true"


tau_prior = float(os.environ.get("TAU_PRIOR", "1.0"))


# NS config

n_live = int(os.environ.get("N_LIVE", "100"))

ns_mcmc_steps = int(os.environ.get("NS_MCMC_STEPS", "20"))

n_iter_max = int(os.environ.get("N_ITER_MAX", "50000"))

tol_logZ = float(os.environ.get("TOL_LOGZ", "1e-3"))

tol_tail = float(os.environ.get("TOL_TAIL", "1e-2"))

patience = int(os.environ.get("PATIENCE", "40"))

stable_repeats = int(os.environ.get("STABLE_REPEATS", "2"))

verbose = os.environ.get("VERBOSE", "False").lower() == "true"

verbose_interval = int(os.environ.get("VERBOSE_INTERVAL", "500"))


# MH config

mh_step_size = float(os.environ.get("MH_STEP_SIZE", "0.10"))

mh_target_accept = float(os.environ.get("MH_TARGET_ACCEPT", "0.234"))

mh_adapt_rate = float(os.environ.get("MH_ADAPT_RATE", "0.05"))

mh_warmup_steps = int(os.environ.get("MH_WARMUP_STEPS", "20"))

mh_step_size_min = float(os.environ.get("MH_STEP_SIZE_MIN", "1e-6"))

mh_step_size_max = float(os.environ.get("MH_STEP_SIZE_MAX", "10.0"))

mh_store_warmup = os.environ.get("MH_STORE_WARMUP", "False").lower() == "true"


# ============================================================

# PATHS

# ============================================================


RESULTS_ROOT = REPO_ROOT / "results"

BOOT_ROOT = RESULTS_ROOT / "bootstrap_runs" / LABEL

NS_RUNS_DIR = BOOT_ROOT / "ns_runs_out"

BOOT_DIR = BOOT_ROOT / "boot_method3_out"

CHUNK_SUMMARY_DIR = BOOT_ROOT / "chunk_summaries"


BOOT_ROOT.mkdir(parents=True, exist_ok=True)

NS_RUNS_DIR.mkdir(parents=True, exist_ok=True)

BOOT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


MULTI_RUN_REF_PATH = Path(

    os.environ.get(

        "MULTI_RUN_REF_PATH",

        str(RESULTS_ROOT / LABEL / f"combined_{LABEL}.npz"),

    )

)


# ============================================================

# CONFIG TAGS

# ============================================================


dim = int(p + (1 if include_intercept else 0))


TAG_BASE = (

    f"d{dim}_nl{int(n_live)}_m{int(ns_mcmc_steps)}_w{int(mh_warmup_steps)}"

    f"_n{int(n)}_p{int(p)}_ds{int(DATA_SEED)}"

)


TAG_LOW = (

    f"lmin1of{int(LOWEST_BLOCK_SIZE)}"

    f"_lseed{int(LOWEST_POOL_SEED)}"

    f"_lN{int(LOWEST_POOL_N)}"

    f"_lpnsseed{int(LOWEST_POOL_NS_SEED)}"

)


TAG_FULL = f"{TAG_BASE}_{TAG_LOW}"


print(f"[label] {LABEL}")

print(f"[multi-run ref] {MULTI_RUN_REF_PATH}")

print(f"[config tag base] {TAG_BASE}")

print(f"[config tag low ] {TAG_LOW}")

print(f"[config tag full] {TAG_FULL}")

print(f"[chunk] CHUNK_ID={CHUNK_ID} CHUNK_SIZE={CHUNK_SIZE} RUN_START={RUN_START} RUN_STOP={RUN_STOP}")


# ============================================================

# LOAD MULTI-RUN REFERENCE

# ============================================================


if not MULTI_RUN_REF_PATH.exists():

    raise FileNotFoundError(

        f"Multi-run reference file not found:\n{MULTI_RUN_REF_PATH}\n"

        "Please set MULTI_RUN_REF_PATH correctly."

    )


ref = np.load(MULTI_RUN_REF_PATH, allow_pickle=True)

logZs_multi = np.asarray(ref["logZs"], dtype=float)

logZs_multi = logZs_multi[np.isfinite(logZs_multi)]


if logZs_multi.size == 0:

    raise RuntimeError("Multi-run reference file contains no finite logZ values.")


print(f"[multi-run ref] loaded {logZs_multi.size} finite logZ values")

if logZs_multi.size > 1:

    print(f"[multi-run ref] mean={logZs_multi.mean(): .6f}, sd={logZs_multi.std(ddof=1): .6f}")

else:

    print(f"[multi-run ref] mean={logZs_multi.mean(): .6f}, sd=nan")


# ============================================================

# SAVE HELPERS

# ============================================================


def save_ns_out(ns_out: dict, ns_seed: int, data_seed: int, lowest_pool_logL: np.ndarray) -> Path:

    out_path = NS_RUNS_DIR / f"ns_out_{TAG_FULL}_nsseed{int(ns_seed)}.npz"

    np.savez_compressed(

        out_path,

        label=np.array([LABEL], dtype=object),

        tag=np.array([TAG_FULL], dtype=object),

        tag_base=np.array([TAG_BASE], dtype=object),

        tag_low=np.array([TAG_LOW], dtype=object),

        chunk_id=int(CHUNK_ID),

        chunk_size=int(CHUNK_SIZE),

        ns_seed=int(ns_seed),

        data_seed=int(data_seed),

        n=int(n),

        p=int(p),

        dim=int(dim),

        n_live=int(n_live),

        ns_mcmc_steps=int(ns_mcmc_steps),

        mh_warmup_steps=int(mh_warmup_steps),

        lowest_pool_mode=np.array(["block_minima_for_ell0_only"], dtype=object),

        lowest_pool_ns_seed=int(LOWEST_POOL_NS_SEED),

        lowest_pool_block_size=int(LOWEST_BLOCK_SIZE),

        lowest_pool_seed=int(LOWEST_POOL_SEED),

        lowest_pool_n_keep=int(LOWEST_POOL_N),

        lowest_pool_logL=np.asarray(lowest_pool_logL, dtype=float),

        logZ=float(ns_out["logZ"]),

        H=float(ns_out.get("H", np.nan)),

        tail_logL=float(ns_out.get("tail_logL", np.nan)),

        dead_logLs=np.asarray(ns_out["dead_logLs"], dtype=float),

        phantom_bins_logL=np.array(ns_out.get("phantom_bins_logL", []), dtype=object),

        trace_logZ=np.asarray(ns_out.get("trace_logZ", []), dtype=float),

        step_sizes_used=np.asarray(ns_out.get("step_sizes_used", []), dtype=float),

        settings=np.array([ns_out.get("settings", {})], dtype=object),

    )

    return out_path



def save_boot_out(out3: dict, ns_seed: int, boot_seed: int) -> Path:

    out_path = BOOT_DIR / f"boot3_{TAG_FULL}_nsseed{int(ns_seed)}_bootseed{int(boot_seed)}.npz"

    np.savez_compressed(

        out_path,

        label=np.array([LABEL], dtype=object),

        tag=np.array([TAG_FULL], dtype=object),

        tag_base=np.array([TAG_BASE], dtype=object),

        tag_low=np.array([TAG_LOW], dtype=object),

        chunk_id=int(CHUNK_ID),

        chunk_size=int(CHUNK_SIZE),

        ns_seed=int(ns_seed),

        boot_seed=int(boot_seed),

        n=int(n),

        p=int(p),

        dim=int(dim),

        n_live=int(n_live),

        ns_mcmc_steps=int(ns_mcmc_steps),

        mh_warmup_steps=int(mh_warmup_steps),

        logZ=np.asarray(out3.get("logZ", []), dtype=float),

        keys=np.array(list(out3.keys()), dtype=object),

        settings=np.array([out3.get("settings", {})], dtype=object),

    )

    return out_path



# ============================================================

# LOWEST-POOL CONSTRUCTION

# ============================================================


def build_lowest_pool_logL_block_minima(

    ns_out_example: dict,

    *,

    n_keep: int,

    block_size: int,

    seed: int,

    tau0: float,

) -> np.ndarray:

    payload = ns_out_example.get("payload", None)

    if payload is None:

        raise KeyError("ns_out_example missing 'payload'. Ensure attach_sim=True.")


    X = np.asarray(payload["X"], dtype=float)

    loglik = payload["loglik"]

    inc0 = bool(payload["include_intercept"])


    n_obs, d_beta = X.shape

    XtX = X.T @ X

    jitter = 1e-10


    Sigma_beta = n_obs * np.linalg.inv(XtX + jitter * np.eye(d_beta))

    try:

        L_beta = np.linalg.cholesky(Sigma_beta)

    except np.linalg.LinAlgError:

        Sigma_beta = n_obs * np.linalg.inv(XtX + 1e-6 * np.eye(d_beta))

        L_beta = np.linalg.cholesky(Sigma_beta)


    rng = np.random.default_rng(int(seed))


    total = int(n_keep) * int(block_size)

    if total <= 0:

        raise ValueError("n_keep and block_size must be positive.")


    Z = rng.standard_normal(size=(total, d_beta))

    beta = Z @ L_beta.T


    if inc0:

        beta0 = float(tau0) * rng.standard_normal(size=(total, 1))

        thetas = np.concatenate([beta0, beta], axis=1)

    else:

        thetas = beta


    logLs = np.array([float(loglik(th)) for th in thetas], dtype=float)

    logLs = np.where(np.isfinite(logLs), logLs, np.inf)


    logLs_blk = logLs.reshape((int(n_keep), int(block_size)))

    idx_min = np.argmin(logLs_blk, axis=1)

    logL_min = logLs_blk[np.arange(int(n_keep)), idx_min]

    logL_min = logL_min[np.isfinite(logL_min)]


    if logL_min.size == 0:

        raise RuntimeError("Lowest pool block-minima produced no finite values.")


    return logL_min



# ============================================================

# METHOD (3) HELPERS

# ============================================================


def simulate_stochastic_log_weights_static_ns(n_live: int, N: int, rng: np.random.RandomState):

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



def _rand_uniform_from_sorted_in_interval(

    arr_sorted: np.ndarray,

    low: float,

    high: float,

    rng: np.random.RandomState,

    allow_equal_low: bool = True,

    allow_equal_high: bool = True,

):

    arr_sorted = np.asarray(arr_sorted, dtype=float)

    if arr_sorted.size == 0:

        return None


    if not np.isfinite(low):

        lo_idx = 0

    else:

        lo_idx = np.searchsorted(arr_sorted, low, side="left" if allow_equal_low else "right")


    if not np.isfinite(high):

        hi_idx = arr_sorted.size - 1

    else:

        hi_idx = np.searchsorted(arr_sorted, high, side="right" if allow_equal_high else "left") - 1


    if hi_idx < lo_idx or lo_idx >= arr_sorted.size or hi_idx < 0:

        return None


    j = rng.randint(lo_idx, hi_idx + 1)

    return float(arr_sorted[j])



def forward_sweep_upper_coupled_force0(

    cand: np.ndarray,

    prev: np.ndarray,

    lowest_pool: np.ndarray,

    rng: np.random.RandomState,

    allow_equal_lower: bool = True,

    eps: float = 1e-12,

) -> np.ndarray:

    prev = np.asarray(prev, dtype=float).ravel()

    N = prev.size

    ell = np.empty(N, dtype=float)


    last = -np.inf

    high0 = float(prev[1]) if N >= 2 else np.inf

    val0 = _rand_uniform_from_sorted_in_interval(

        lowest_pool, low=last, high=high0, rng=rng,

        allow_equal_low=allow_equal_lower, allow_equal_high=True

    )

    if val0 is None:

        val0 = _rand_uniform_from_sorted_in_interval(

            cand, low=last, high=high0, rng=rng,

            allow_equal_low=allow_equal_lower, allow_equal_high=True

        )

    if val0 is None:

        val0 = float(prev[0])

        if (not allow_equal_lower) and (val0 <= last):

            val0 = last + eps

        if np.isfinite(high0) and (val0 > high0):

            val0 = high0


    ell[0] = float(val0)

    last = ell[0]


    for i in range(1, N):

        low = last

        high = float(prev[i + 1]) if i < N - 1 else np.inf


        val = _rand_uniform_from_sorted_in_interval(

            cand, low=low, high=high, rng=rng,

            allow_equal_low=allow_equal_lower, allow_equal_high=True

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


        ell[i] = float(val)

        last = ell[i]


    return ell



def backward_sweep_lower_coupled_lock0(

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


    ell0_fixed = float(ell[0])


    for i in range(N - 1, -1, -1):

        if i == 0:

            continue


        low = float(prev[i - 1]) if i > 0 else -np.inf

        high = float(ell[i + 1]) if i < N - 1 else np.inf


        val = _rand_uniform_from_sorted_in_interval(

            cand, low=low, high=high, rng=rng,

            allow_equal_low=True, allow_equal_high=allow_equal_lower

        )

        if val is None:

            val = float(prev[i])

            if val < low:

                val = low

            if np.isfinite(high) and val > high:

                val = high

            if (not allow_equal_lower) and (i < N - 1) and (val >= high):

                val = min(high - eps, val)


        ell[i] = float(val)


    ell[0] = ell0_fixed

    ell = np.maximum.accumulate(ell)

    return ell



def swish_block_mix_lock0(

    cand: np.ndarray,

    prev: np.ndarray,

    ell: np.ndarray,

    rng: np.random.RandomState,

    n_blocks: int = 40,

    mean_block_len: int = 80,

    n_site_sweeps_per_block: int = 2,

    allow_equal_lower: bool = True,

    eps: float = 1e-12,

    p_use_upper_coupling: float = 0.5,

) -> np.ndarray:

    prev = np.asarray(prev, dtype=float).ravel()

    ell = np.asarray(ell, dtype=float).ravel().copy()

    N = ell.size

    ell0_fixed = float(ell[0])

    ell = np.maximum.accumulate(ell)


    for _ in range(int(n_blocks)):

        L = max(2, int(rng.exponential(scale=max(2.0, float(mean_block_len)))))

        a = int(rng.randint(0, N))

        b = int(min(N - 1, a + L - 1))


        if a == 0:

            a = 1

        if b == 0:

            b = 1

        if a > b:

            continue


        use_upper = (rng.rand() < float(p_use_upper_coupling))

        idxs = np.arange(a, b + 1)


        for _ in range(int(n_site_sweeps_per_block)):

            rng.shuffle(idxs)

            for i in idxs:

                if i == 0:

                    continue


                left = float(ell[i - 1]) if i > 0 else -np.inf

                right = float(ell[i + 1]) if i < N - 1 else np.inf


                if use_upper:

                    ub = float(prev[i + 1]) if i < N - 1 else np.inf

                    low, high = left, min(right, ub)

                else:

                    lb = float(prev[i - 1]) if i > 0 else -np.inf

                    low, high = max(left, lb), right


                val = _rand_uniform_from_sorted_in_interval(

                    cand, low=low, high=high, rng=rng,

                    allow_equal_low=True, allow_equal_high=allow_equal_lower

                )

                if val is None:

                    continue


                ell[i] = float(val)

                if i > 0 and ell[i] < ell[i - 1]:

                    ell[i] = ell[i - 1]

                if i < N - 1 and ell[i] > ell[i + 1]:

                    ell[i] = ell[i + 1]

                if (not allow_equal_lower) and i > 0 and ell[i] <= ell[i - 1]:

                    ell[i] = ell[i - 1] + eps

                    if i < N - 1:

                        ell[i] = min(ell[i], ell[i + 1])


            ell[0] = ell0_fixed

            ell = np.maximum.accumulate(ell)


    ell[0] = ell0_fixed

    ell = np.maximum.accumulate(ell)

    return ell



def method3_stochshrink_two_sided_chain_force_lowest0(

    ns_out: dict,

    n_live: int,

    lowest_pool_logL: np.ndarray,

    n_boot: int = 2000,

    n_shrink: int = 5,

    seed: int | None = 415,

    pool_decimals: int | None = None,

    allow_equal_lower: bool = True,

    chain_prev: bool = True,

    use_swish: bool = True,

    swish_blocks: int = 40,

    swish_mean_block_len: int = 80,

    swish_site_sweeps_per_block: int = 2,

    swish_p_upper: float = 0.5,

    lag_every: int = 10,

) -> dict:

    rng = np.random.RandomState(seed)


    dead_base = np.asarray(ns_out["dead_logLs"], dtype=float).ravel()

    ell_tail = float(ns_out["tail_logL"])

    N = int(dead_base.size)


    cand = build_global_candidate_pool_logL_all(ns_out, pool_decimals=pool_decimals)

    cand = cand[np.isfinite(cand)]

    cand = np.sort(cand)


    lowest_pool = np.asarray(lowest_pool_logL, dtype=float).ravel()

    lowest_pool = lowest_pool[np.isfinite(lowest_pool)]

    lowest_pool = np.sort(lowest_pool)


    base_ref = compute_base_det_reference(ns_out, n_live=int(n_live))


    n_boot_total = int(n_boot) * int(lag_every)


    ladders_kept = np.empty((int(n_boot), N), dtype=float)

    logZ_kept = np.empty((int(n_boot), int(n_shrink)), dtype=float)

    ess_dead_kept = np.empty((int(n_boot), int(n_shrink)), dtype=float)

    ess_dead_tail_kept = np.empty((int(n_boot), int(n_shrink)), dtype=float)


    prev = dead_base.copy()

    kept = 0


    for t in range(int(n_boot_total)):

        ell = forward_sweep_upper_coupled_force0(

            cand=cand,

            prev=prev,

            lowest_pool=lowest_pool,

            rng=rng,

            allow_equal_lower=bool(allow_equal_lower),

        )

        ell = backward_sweep_lower_coupled_lock0(

            cand=cand,

            prev=prev,

            ell_init=ell,

            rng=rng,

            allow_equal_lower=bool(allow_equal_lower),

        )

        if bool(use_swish):

            ell = swish_block_mix_lock0(

                cand=cand,

                prev=prev,

                ell=ell,

                rng=rng,

                n_blocks=int(swish_blocks),

                mean_block_len=int(swish_mean_block_len),

                n_site_sweeps_per_block=int(swish_site_sweeps_per_block),

                allow_equal_lower=bool(allow_equal_lower),

                p_use_upper_coupling=float(swish_p_upper),

            )


        prev = ell if bool(chain_prev) else dead_base


        if ((t + 1) % int(lag_every)) != 0:

            continue


        ladders_kept[kept, :] = ell


        for s in range(int(n_shrink)):

            log_w_dead_s, log_w_tail_s = simulate_stochastic_log_weights_static_ns(

                n_live=int(n_live),

                N=N,

                rng=rng,

            )

            (

                logZ_kept[kept, s],

                ess_dead_kept[kept, s],

                ess_dead_tail_kept[kept, s],

            ) = compute_logZ_and_ESS_from_logw_and_ell(

                log_w_dead=log_w_dead_s,

                ell_dead=ell,

                log_w_tail=log_w_tail_s,

                ell_tail=ell_tail,

            )


        kept += 1

        if kept >= int(n_boot):

            break


    return {

        "logZ": logZ_kept,

        "ESS_dead": ess_dead_kept,

        "ESS_dead_tail": ess_dead_tail_kept,

        "ell_ladders": ladders_kept,

        "candidate_pool_size": int(cand.size),

        "lowest_pool_size": int(lowest_pool.size),

        "base": base_ref,

        "settings": {

            "method": "3 (stoch shrink + two-sided ladder chain) + forced ell[0] from separate lowest-pool",

            "n_boot": int(n_boot),

            "n_shrink": int(n_shrink),

            "lag_every": int(lag_every),

            "use_swish": bool(use_swish),

            "seed": seed,

        },

    }



# ============================================================

# INITIAL NS RUN TO BUILD LOWEST POOL

# ============================================================


print(f"[lowest-pool] building fixed lowest pool using LOWEST_POOL_NS_SEED={LOWEST_POOL_NS_SEED}")


ns_out_lowest = run_ns_mh_phantom(

    n=n,

    p=p,

    use_correlated_X=use_correlated_X,

    rho=rho,

    sigma_beta=sigma_beta,

    sparsity=sparsity,

    include_intercept=include_intercept,

    data_seed=DATA_SEED,

    tau_prior=tau_prior,

    n_live=n_live,

    ns_mcmc_steps=ns_mcmc_steps,

    n_iter_max=n_iter_max,

    ns_seed=LOWEST_POOL_NS_SEED,

    tol_logZ=tol_logZ,

    tol_tail=tol_tail,

    patience=patience,

    stable_repeats=stable_repeats,

    verbose=verbose,

    verbose_interval=verbose_interval,

    mh_step_size=mh_step_size,

    mh_target_accept=mh_target_accept,

    mh_adapt_rate=mh_adapt_rate,

    mh_warmup_steps=mh_warmup_steps,

    mh_step_size_min=mh_step_size_min,

    mh_step_size_max=mh_step_size_max,

    mh_store_warmup=mh_store_warmup,

    attach_sim=True,

)


lowest_pool_logL = build_lowest_pool_logL_block_minima(

    ns_out_lowest,

    n_keep=LOWEST_POOL_N,

    block_size=LOWEST_BLOCK_SIZE,

    seed=LOWEST_POOL_SEED,

    tau0=tau_prior,

)


print(

    f"[lowest pool] kept={lowest_pool_logL.size} "

    f"mean={lowest_pool_logL.mean(): .6f} "

    f"sd={lowest_pool_logL.std(ddof=1): .6f}"

)


# ============================================================

# MAIN CHUNK LOOP

# ============================================================


logZ_boot_by_seed: List[np.ndarray] = []

seed_list: List[int] = []

boot_seed_list: List[int] = []

ns_logZ_list: List[float] = []

saved_ns_paths: List[Path] = []

saved_boot_paths: List[Path] = []


for local_i, global_i in enumerate(range(RUN_START, RUN_STOP), start=1):

    run_start_time = time.perf_counter()


    ns_seed = int(NS_BASE_SEED + global_i)

    boot_seed = int(BOOT_BASE_SEED + global_i)


    seed_list.append(ns_seed)

    boot_seed_list.append(boot_seed)


    t_ns0 = time.perf_counter()

    ns_out_i = run_ns_mh_phantom(

        n=n,

        p=p,

        use_correlated_X=use_correlated_X,

        rho=rho,

        sigma_beta=sigma_beta,

        sparsity=sparsity,

        include_intercept=include_intercept,

        data_seed=DATA_SEED,

        tau_prior=tau_prior,

        n_live=n_live,

        ns_mcmc_steps=ns_mcmc_steps,

        n_iter_max=n_iter_max,

        ns_seed=ns_seed,

        tol_logZ=tol_logZ,

        tol_tail=tol_tail,

        patience=patience,

        stable_repeats=stable_repeats,

        verbose=False,

        verbose_interval=verbose_interval,

        mh_step_size=mh_step_size,

        mh_target_accept=mh_target_accept,

        mh_adapt_rate=mh_adapt_rate,

        mh_warmup_steps=mh_warmup_steps,

        mh_step_size_min=mh_step_size_min,

        mh_step_size_max=mh_step_size_max,

        mh_store_warmup=mh_store_warmup,

        attach_sim=False,

    )

    t_ns = time.perf_counter() - t_ns0


    ns_logZ = float(ns_out_i["logZ"])

    ns_logZ_list.append(ns_logZ)


    p_ns = save_ns_out(

        ns_out_i,

        ns_seed=ns_seed,

        data_seed=DATA_SEED,

        lowest_pool_logL=lowest_pool_logL,

    )

    saved_ns_paths.append(p_ns)


    t_m30 = time.perf_counter()

    out3_i = method3_stochshrink_two_sided_chain_force_lowest0(

        ns_out=ns_out_i,

        n_live=n_live,

        lowest_pool_logL=lowest_pool_logL,

        n_boot=int(N_BOOT_PER_RUN),

        n_shrink=int(N_SHRINK),

        seed=int(boot_seed),

        pool_decimals=None,

        allow_equal_lower=True,

        chain_prev=True,

        use_swish=True,

        swish_blocks=40,

        swish_mean_block_len=80,

        swish_site_sweeps_per_block=2,

        swish_p_upper=0.5,

        lag_every=10,

    )

    t_m3 = time.perf_counter() - t_m30


    z = np.asarray(out3_i["logZ"], dtype=float).reshape(-1)

    z = z[np.isfinite(z)]

    logZ_boot_by_seed.append(z)


    p_boot = save_boot_out(out3_i, ns_seed=ns_seed, boot_seed=boot_seed)

    saved_boot_paths.append(p_boot)


    run_time = time.perf_counter() - run_start_time

    elapsed_total = time.perf_counter() - GLOBAL_START


    z_mean = float(np.mean(z)) if z.size > 0 else np.nan

    z_sd = float(np.std(z, ddof=1)) if z.size > 1 else np.nan


    print(

        f"[chunk {CHUNK_ID:03d} | run {local_i}/{RUN_COUNT} | global {global_i}] "

        f"NS seed={ns_seed} -> NS logZ={ns_logZ: .6f} | "

        f"M3 seed={boot_seed} -> n={z.size} mean={z_mean: .6f} sd={z_sd: .6f} | "

        f"t_NS={format_time(t_ns)} t_M3={format_time(t_m3)} t_run={format_time(run_time)} | "

        f"elapsed={format_time(elapsed_total)} ETA={eta_str(elapsed_total, local_i, RUN_COUNT)} | "

        f"saved: {p_ns.name}"

    )


logZ_boot_pooled = (

    np.concatenate([z for z in logZ_boot_by_seed if z.size > 0])

    if logZ_boot_by_seed

    else np.array([], dtype=float)

)


summary_path = CHUNK_SUMMARY_DIR / (

    f"bootstrap_summary_{LABEL}_{TAG_FULL}_chunk{CHUNK_ID:03d}.npz"

)


np.savez_compressed(

    summary_path,

    label=np.array([LABEL], dtype=object),

    tag=np.array([TAG_FULL], dtype=object),

    tag_base=np.array([TAG_BASE], dtype=object),

    tag_low=np.array([TAG_LOW], dtype=object),

    multi_run_ref_path=np.array([str(MULTI_RUN_REF_PATH)], dtype=object),

    chunk_id=int(CHUNK_ID),

    chunk_size=int(CHUNK_SIZE),

    run_start=int(RUN_START),

    run_stop=int(RUN_STOP),

    run_count=int(RUN_COUNT),

    boot_runs_nominal=int(BOOT_RUNS),

    n_boot_per_run=int(N_BOOT_PER_RUN),

    n_shrink=int(N_SHRINK),

    ns_base_seed=int(NS_BASE_SEED),

    boot_base_seed=int(BOOT_BASE_SEED),

    lowest_pool_ns_seed=int(LOWEST_POOL_NS_SEED),

    ns_seeds=np.asarray(seed_list, dtype=np.int64),

    boot_seeds=np.asarray(boot_seed_list, dtype=np.int64),

    ns_logZs=np.asarray(ns_logZ_list, dtype=float),

    pooled_boot_logZ=np.asarray(logZ_boot_pooled, dtype=float),

    saved_ns_paths=np.array([str(p) for p in saved_ns_paths], dtype=object),

    saved_boot_paths=np.array([str(p) for p in saved_boot_paths], dtype=object),

)


print("\n=== Chunk-pooled Method (3) across NS seeds ===")

if logZ_boot_pooled.size > 0:

    boot_mean = float(np.mean(logZ_boot_pooled))

    boot_sd = float(np.std(logZ_boot_pooled, ddof=1)) if logZ_boot_pooled.size > 1 else np.nan

    print(f"[chunk pooled] n={logZ_boot_pooled.size} mean={boot_mean: .6f} sd={boot_sd: .6f}")

else:

    print("[chunk pooled] n=0 mean=nan sd=nan")


if len(ns_logZ_list) > 0:

    ns_mean = float(np.mean(ns_logZ_list))

    ns_sd = float(np.std(ns_logZ_list, ddof=1)) if len(ns_logZ_list) > 1 else np.nan

    print(f"[NS logZs] mean={ns_mean: .6f} sd={ns_sd: .6f}")

else:

    print("[NS logZs] mean=nan sd=nan")


print("\nSaved per-run NS outputs to:", NS_RUNS_DIR)

print("Saved per-run bootstrap outputs to:", BOOT_DIR)

print("Saved chunk summary to:", summary_path)

print(f"\n[TIMER] total runtime = {format_time(time.perf_counter() - GLOBAL_START)}")
