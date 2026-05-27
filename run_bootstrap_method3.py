from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ============================================================
# Path setup
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from ns_mh_phantom import ( 
    _prior_sampler,
    make_payload,
    run_ns_mh_phantom,
    simulate_logistic_data,
)

from bootstrap_method3 import method3_stochshrink_anchored_two_sweep_chain


# ============================================================
# Timer helpers
# ============================================================

GLOBAL_START = time.perf_counter()


def format_time(seconds: float) -> str:
    seconds = int(max(0, round(float(seconds))))

    d = seconds // 86400
    h = (seconds % 86400) // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if d > 0:
        return f"{d}d {h:02d}:{m:02d}:{s:02d}"

    return f"{h:02d}:{m:02d}:{s:02d}"


def eta_str(elapsed: float, done: int, total: int) -> str:
    if done <= 0 or total <= 0:
        return "??:??:??"

    per = float(elapsed) / int(done)
    rem = per * max(0, int(total) - int(done))

    return format_time(rem)


# ============================================================
# Environment helpers
# ============================================================

def get_env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name, str(default))
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_chunk_id() -> int:
    """
    Prefer SLURM_ARRAY_TASK_ID when running as a Slurm array job.
    Fall back to CHUNK_ID if set manually, otherwise 0.
    """
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        return int(os.environ["SLURM_ARRAY_TASK_ID"])

    return int(os.environ.get("CHUNK_ID", "0"))


# ============================================================
# Config
# ============================================================

LABEL = os.environ.get("LABEL", "nl100_m20_w20")

# Family-level seed bases.
BOOT_RUNS = int(os.environ.get("BOOT_RUNS", "35"))
DATA_SEED = int(os.environ.get("DATA_SEED", "415"))
NS_BASE_SEED = int(os.environ.get("NS_BASE_SEED", "1000"))
BOOT_BASE_SEED = int(os.environ.get("BOOT_BASE_SEED", "2000"))

# Chunking.
CHUNK_ID = get_chunk_id()
SLURM_ARRAY_TASK_ID = os.environ.get("SLURM_ARRAY_TASK_ID", None)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", str(BOOT_RUNS)))

RUN_START = CHUNK_ID * CHUNK_SIZE
RUN_STOP = min(RUN_START + CHUNK_SIZE, BOOT_RUNS)
RUN_COUNT = max(0, RUN_STOP - RUN_START)

# Data/model config.
n = int(os.environ.get("N", "600"))
p = int(os.environ.get("P", "12"))

use_correlated_X = get_env_bool("USE_CORRELATED_X", False)
rho = float(os.environ.get("RHO", "1.0"))
sigma_beta = float(os.environ.get("SIGMA_BETA", "1.0"))
sparsity = float(os.environ.get("SPARSITY", "0.0"))
include_intercept = get_env_bool("INCLUDE_INTERCEPT", False)
tau_prior = float(os.environ.get("TAU_PRIOR", "1.0"))

# NS config.
n_live = int(os.environ.get("N_LIVE", "100"))
ns_mcmc_steps = int(os.environ.get("NS_MCMC_STEPS", "20"))
n_iter_max = int(os.environ.get("N_ITER_MAX", "50000"))

tol_logZ = float(os.environ.get("TOL_LOGZ", "1e-3"))
tol_tail = float(os.environ.get("TOL_TAIL", "1e-2"))
patience = int(os.environ.get("PATIENCE", "40"))
stable_repeats = int(os.environ.get("STABLE_REPEATS", "2"))

verbose = get_env_bool("VERBOSE", False)
verbose_interval = int(os.environ.get("VERBOSE_INTERVAL", "500"))

# MH config.
mh_step_size = float(os.environ.get("MH_STEP_SIZE", "0.10"))
mh_target_accept = float(os.environ.get("MH_TARGET_ACCEPT", "0.234"))
mh_adapt_rate = float(os.environ.get("MH_ADAPT_RATE", "0.05"))
mh_warmup_steps = int(os.environ.get("MH_WARMUP_STEPS", "20"))
mh_step_size_min = float(os.environ.get("MH_STEP_SIZE_MIN", "1e-6"))
mh_step_size_max = float(os.environ.get("MH_STEP_SIZE_MAX", "10.0"))

mh_store_warmup = False

# Bootstrap controls.
N_BOOT_PER_RUN = int(os.environ.get("N_BOOT_PER_RUN", "200"))

N_SHRINK = 1

LADDER_SWEEPS_S = 10

ANCHOR_POOL_SIZE_H = 100

ANCHOR_BLOCK_SIZE = int(n_live)

ANCHOR_POOL_SEED = int(os.environ.get("ANCHOR_POOL_SEED", "415"))

# ============================================================
# Paths
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
# Tags
# ============================================================

dim = int(p + (1 if include_intercept else 0))

TAG_BASE = (
    f"d{dim}_nl{int(n_live)}_m{int(ns_mcmc_steps)}_w{int(mh_warmup_steps)}"
    f"_n{int(n)}_p{int(p)}_ds{int(DATA_SEED)}"
)

TAG_ANCHOR = (
    f"anchorH{int(ANCHOR_POOL_SIZE_H)}"
    f"_min1ofnlive{int(ANCHOR_BLOCK_SIZE)}"
    f"_aseed{int(ANCHOR_POOL_SEED)}"
)

TAG_FULL = f"{TAG_BASE}_{TAG_ANCHOR}"


print(f"[label] {LABEL}")
print(f"[repo root] {REPO_ROOT}")
print(f"[config tag base] {TAG_BASE}")
print(f"[config tag anchor] {TAG_ANCHOR}")
print(f"[config tag full] {TAG_FULL}")
print(
    f"[chunk] SLURM_ARRAY_TASK_ID={SLURM_ARRAY_TASK_ID} "
    f"CHUNK_ID={CHUNK_ID} CHUNK_SIZE={CHUNK_SIZE} "
    f"RUN_START={RUN_START} RUN_STOP={RUN_STOP} RUN_COUNT={RUN_COUNT}"
)

print(
    "[dissertation bootstrap settings] "
    f"H={ANCHOR_POOL_SIZE_H}, "
    f"anchor block size=n_live={ANCHOR_BLOCK_SIZE}, "
    f"S={LADDER_SWEEPS_S}, "
    f"n_shrink={N_SHRINK}, "
    f"warmup production restart=True"
)


# ============================================================
# Optional multi-run reference loading
# ============================================================

if MULTI_RUN_REF_PATH.exists():
    ref = np.load(MULTI_RUN_REF_PATH, allow_pickle=True)
    logZs_multi = np.asarray(ref["logZs"], dtype=float)
    logZs_multi = logZs_multi[np.isfinite(logZs_multi)]

    print(f"[multi-run ref] {MULTI_RUN_REF_PATH}")
    print(f"[multi-run ref] loaded {logZs_multi.size} finite logZ values")

    if logZs_multi.size > 1:
        print(
            f"[multi-run ref] mean={logZs_multi.mean(): .6f}, "
            f"sd={logZs_multi.std(ddof=1): .6f}"
        )
    elif logZs_multi.size == 1:
        print(f"[multi-run ref] mean={logZs_multi.mean(): .6f}, sd=nan")
else:
    logZs_multi = np.array([], dtype=float)
    print(f"[multi-run ref] not found, continuing without it: {MULTI_RUN_REF_PATH}")


# ============================================================
# Anchor pool construction
# ============================================================

def build_anchor_pool_logL(
    *,
    n_keep: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    """
    Build the dissertation anchor pool.

    Each anchor is the minimum log-likelihood over n_live prior draws.
    Therefore, with the dissertation settings:

        n_keep = H = 100
        block_size = n_live
    """
    sim = simulate_logistic_data(
        n=int(n),
        p=int(p),
        use_correlated_X=bool(use_correlated_X),
        rho=float(rho),
        beta_generation="inferential_prior",
        sigma_beta=float(sigma_beta),
        sparsity=float(sparsity),
        include_intercept=bool(include_intercept),
        tau0=float(tau_prior),
        seed=int(DATA_SEED),
    )

    payload = make_payload(
        sim,
        include_intercept=bool(include_intercept),
        tau0_for_intercept=float(tau_prior),
    )

    loglik_fn = payload["loglik"]

    prior_draw = _prior_sampler(
        d_beta=int(p),
        X=sim.X,
        tau0=float(tau_prior),
        include_intercept=bool(include_intercept),
    )

    np.random.seed(int(seed))

    anchors = np.empty(int(n_keep), dtype=float)

    for h in range(int(n_keep)):
        vals = np.empty(int(block_size), dtype=float)

        for j in range(int(block_size)):
            theta = prior_draw()
            vals[j] = float(loglik_fn(theta))

        anchors[h] = float(np.min(vals))

        if (h + 1) % 10 == 0:
            print(
                f"[anchor pool] built {h + 1:3d}/{int(n_keep)} anchors "
                f"mean={anchors[:h + 1].mean(): .6f}"
            )

    return anchors


print("[anchor pool] constructing dissertation anchor pool...")
anchor_pool_logL = build_anchor_pool_logL(
    n_keep=int(ANCHOR_POOL_SIZE_H),
    block_size=int(ANCHOR_BLOCK_SIZE),
    seed=int(ANCHOR_POOL_SEED),
)

print(
    "[anchor pool] done: "
    f"H={anchor_pool_logL.size}, "
    f"min={anchor_pool_logL.min(): .6f}, "
    f"median={np.median(anchor_pool_logL): .6f}, "
    f"max={anchor_pool_logL.max(): .6f}"
)


# ============================================================
# Save helpers
# ============================================================

def save_ns_out(
    ns_out: dict,
    ns_seed: int,
    data_seed: int,
) -> Path:
    out_path = NS_RUNS_DIR / f"ns_out_{TAG_FULL}_nsseed{int(ns_seed)}.npz"

    np.savez_compressed(
        out_path,
        label=np.array([LABEL], dtype=object),
        tag=np.array([TAG_FULL], dtype=object),
        tag_base=np.array([TAG_BASE], dtype=object),
        tag_anchor=np.array([TAG_ANCHOR], dtype=object),
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
        mh_production_restarts_from_seed=True,
        mh_store_warmup=False,
        anchor_pool_mode=np.array(["minimum_of_nlive_prior_loglikelihoods"], dtype=object),
        anchor_pool_size_H=int(ANCHOR_POOL_SIZE_H),
        anchor_block_size=int(ANCHOR_BLOCK_SIZE),
        anchor_pool_seed=int(ANCHOR_POOL_SEED),
        anchor_pool_logL=np.asarray(anchor_pool_logL, dtype=float),
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


def save_boot_out(
    out3: dict,
    ns_seed: int,
    boot_seed: int,
) -> Path:
    out_path = BOOT_DIR / f"boot3_{TAG_FULL}_nsseed{int(ns_seed)}_bootseed{int(boot_seed)}.npz"

    np.savez_compressed(
        out_path,
        label=np.array([LABEL], dtype=object),
        tag=np.array([TAG_FULL], dtype=object),
        tag_base=np.array([TAG_BASE], dtype=object),
        tag_anchor=np.array([TAG_ANCHOR], dtype=object),
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
        mh_production_restarts_from_seed=True,
        mh_store_warmup=False,
        anchor_pool_mode=np.array(["minimum_of_nlive_prior_loglikelihoods"], dtype=object),
        anchor_pool_size_H=int(ANCHOR_POOL_SIZE_H),
        anchor_block_size=int(ANCHOR_BLOCK_SIZE),
        anchor_pool_seed=int(ANCHOR_POOL_SEED),
        anchor_pool_logL=np.asarray(anchor_pool_logL, dtype=float),
        ladder_sweeps_S=int(LADDER_SWEEPS_S),
        n_boot_per_run=int(N_BOOT_PER_RUN),
        n_shrink=int(N_SHRINK),
        logZ=np.asarray(out3.get("logZ", []), dtype=float),
        logZ_flat=np.asarray(out3.get("logZ", []), dtype=float).ravel(),
        ESS_dead=np.asarray(out3.get("ESS_dead", []), dtype=float),
        ESS_dead_tail=np.asarray(out3.get("ESS_dead_tail", []), dtype=float),
        ell_ladders=np.asarray(out3.get("ell_ladders", []), dtype=float),
        anchors=np.asarray(out3.get("anchors", []), dtype=float),
        candidate_pool_size=int(out3.get("candidate_pool_size", -1)),
        bootstrap_anchor_pool_size=int(out3.get("anchor_pool_size", -1)),
        base=np.array([out3.get("base", {})], dtype=object),
        settings=np.array([out3.get("settings", {})], dtype=object),
    )

    return out_path


# ============================================================
# Main loop
# ============================================================

chunk_logZ_ns = []
chunk_logZ_boot_mean = []
chunk_logZ_boot_sd = []
chunk_ns_seeds = []
chunk_boot_seeds = []
chunk_ns_times = []
chunk_boot_times = []

print("[main] starting runs")

for run_index in range(int(RUN_START), int(RUN_STOP)):
    run_no = run_index + 1

    ns_seed = int(NS_BASE_SEED + run_index)
    boot_seed = int(BOOT_BASE_SEED + run_index)

    print("\n" + "=" * 80)
    print(f"[run {run_no}/{BOOT_RUNS}] ns_seed={ns_seed}, boot_seed={boot_seed}")
    print("=" * 80)

    # ------------------------------------------------------------
    # Observed NS run.
    # ------------------------------------------------------------
    t_ns0 = time.perf_counter()

    ns_out_i = run_ns_mh_phantom(
        n=int(n),
        p=int(p),
        use_correlated_X=bool(use_correlated_X),
        rho=float(rho),
        sigma_beta=float(sigma_beta),
        sparsity=float(sparsity),
        include_intercept=bool(include_intercept),
        data_seed=int(DATA_SEED),
        tau_prior=float(tau_prior),
        n_live=int(n_live),
        ns_mcmc_steps=int(ns_mcmc_steps),
        n_iter_max=int(n_iter_max),
        ns_seed=int(ns_seed),
        tol_logZ=float(tol_logZ),
        tol_tail=float(tol_tail),
        patience=int(patience),
        stable_repeats=int(stable_repeats),
        verbose=bool(verbose),
        verbose_interval=int(verbose_interval),
        mh_step_size=float(mh_step_size),
        mh_target_accept=float(mh_target_accept),
        mh_adapt_rate=float(mh_adapt_rate),
        mh_warmup_steps=int(mh_warmup_steps),
        mh_step_size_min=float(mh_step_size_min),
        mh_step_size_max=float(mh_step_size_max),
        mh_store_warmup=False,
        attach_sim=True,
    )

    t_ns1 = time.perf_counter()
    ns_time = float(t_ns1 - t_ns0)

    ns_path = save_ns_out(
        ns_out=ns_out_i,
        ns_seed=int(ns_seed),
        data_seed=int(DATA_SEED),
    )

    print(
        f"[run {run_no}] NS logZ={float(ns_out_i['logZ']): .6f}, "
        f"N_dead={len(ns_out_i['dead_logLs'])}, "
        f"time={format_time(ns_time)}"
    )
    print(f"[run {run_no}] saved NS: {ns_path}")

    # ------------------------------------------------------------
    # Bootstrap.
    # ------------------------------------------------------------
    t_boot0 = time.perf_counter()

    out3_i = method3_stochshrink_anchored_two_sweep_chain(
        ns_out=ns_out_i,
        n_live=int(n_live),
        anchor_pool_logL=anchor_pool_logL,
        n_boot=int(N_BOOT_PER_RUN),
        n_shrink=int(N_SHRINK),
        seed=int(boot_seed),
        S=int(LADDER_SWEEPS_S),
        pool_decimals=None,
        allow_equal_lower=True,
        chain_prev=True,
        print_every=0,
    )

    t_boot1 = time.perf_counter()
    boot_time = float(t_boot1 - t_boot0)

    boot_path = save_boot_out(
        out3=out3_i,
        ns_seed=int(ns_seed),
        boot_seed=int(boot_seed),
    )

    logZ_boot_flat = np.asarray(out3_i["logZ"], dtype=float).ravel()

    boot_mean = float(np.mean(logZ_boot_flat))
    boot_sd = float(np.std(logZ_boot_flat, ddof=1)) if logZ_boot_flat.size > 1 else np.nan

    print(
        f"[run {run_no}] bootstrap mean={boot_mean: .6f}, "
        f"sd={boot_sd: .6f}, "
        f"time={format_time(boot_time)}"
    )
    print(f"[run {run_no}] saved bootstrap: {boot_path}")

    chunk_logZ_ns.append(float(ns_out_i["logZ"]))
    chunk_logZ_boot_mean.append(boot_mean)
    chunk_logZ_boot_sd.append(boot_sd)
    chunk_ns_seeds.append(int(ns_seed))
    chunk_boot_seeds.append(int(boot_seed))
    chunk_ns_times.append(ns_time)
    chunk_boot_times.append(boot_time)

    done = run_index - int(RUN_START) + 1
    elapsed = time.perf_counter() - GLOBAL_START

    print(
        f"[progress] done {done}/{RUN_COUNT} in {format_time(elapsed)}, "
        f"ETA {eta_str(elapsed, done, RUN_COUNT)}"
    )


# ============================================================
# Chunk summary
# ============================================================

summary_path = CHUNK_SUMMARY_DIR / f"summary_{TAG_FULL}_chunk{int(CHUNK_ID)}.npz"

np.savez_compressed(
    summary_path,
    label=np.array([LABEL], dtype=object),
    tag=np.array([TAG_FULL], dtype=object),
    tag_base=np.array([TAG_BASE], dtype=object),
    tag_anchor=np.array([TAG_ANCHOR], dtype=object),
    chunk_id=int(CHUNK_ID),
    chunk_size=int(CHUNK_SIZE),
    run_start=int(RUN_START),
    run_stop=int(RUN_STOP),
    run_count=int(RUN_COUNT),
    ns_seeds=np.asarray(chunk_ns_seeds, dtype=int),
    boot_seeds=np.asarray(chunk_boot_seeds, dtype=int),
    logZ_ns=np.asarray(chunk_logZ_ns, dtype=float),
    logZ_boot_mean=np.asarray(chunk_logZ_boot_mean, dtype=float),
    logZ_boot_sd=np.asarray(chunk_logZ_boot_sd, dtype=float),
    ns_times=np.asarray(chunk_ns_times, dtype=float),
    boot_times=np.asarray(chunk_boot_times, dtype=float),
    anchor_pool_logL=np.asarray(anchor_pool_logL, dtype=float),
    anchor_pool_size_H=int(ANCHOR_POOL_SIZE_H),
    anchor_block_size=int(ANCHOR_BLOCK_SIZE),
    ladder_sweeps_S=int(LADDER_SWEEPS_S),
    n_shrink=int(N_SHRINK),
    n_boot_per_run=int(N_BOOT_PER_RUN),
)

elapsed_total = time.perf_counter() - GLOBAL_START

print("\n" + "=" * 80)
print("[done]")
print(f"[summary] saved: {summary_path}")
print(f"[total time] {format_time(elapsed_total)}")
print("=" * 80)
