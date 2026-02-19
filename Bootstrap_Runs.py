# ============================================================
# BOOTSTRAP vs MULTI-RUN REFERENCE
#
# - Loads multi-run NS reference distribution (logZs) from .npz
# - Runs BOOT_RUNS fresh NS runs (same DATA_SEED, varying ns_seed)
# - Saves EACH ns_out to disk (dead + phantom "points" included)
# - Runs Method (3) on each ns_out
# - QQ plot (per-seed + pooled) + histogram overlay
#
# Assumes you have installed your package (editable):
#   pip install -e .
#
# Assumes Method (3) function is importable (update import below)
# ============================================================

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ---- Canonical NS wrapper + storage report (from your package)
from nsdissert import run_ns_mh_phantom, report_storage

# ---- Method (3)
from nsdissert.bootstrap_method3 import method3_stochshrink_two_sided_chain

# ============================================================
# TIMER START
# ============================================================

GLOBAL_START = time.perf_counter()


def format_time(seconds: float) -> str:
    """Pretty HH:MM:SS formatter."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def eta_str(elapsed: float, done: int, total: int) -> str:
    """ETA given elapsed seconds and done/total iterations."""
    if done <= 0:
        return "??:??:??"
    per = elapsed / done
    rem = per * (total - done)
    return format_time(rem)

# ============================================================
# Paths
# ============================================================

# Repo-friendly default: put results under a top-level "results/" if it exists,
# otherwise fall back to your existing Desktop/ns_results/MCGoldenSave.
REPO_ROOT = Path.cwd()  # in a notebook, this is where the notebook is launched
RESULTS_DIR = (REPO_ROOT / "results" / "MCGoldenSave")

if not RESULTS_DIR.exists():
    RESULTS_DIR = Path.home() / "Desktop" / "ns_results" / "MCGoldenSave"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Where we save per-seed NS runs (includes ~1600 points: dead+phantom logLs)
NS_RUNS_DIR = RESULTS_DIR / "ns_runs_out"
NS_RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Where we save per-seed Method (3) bootstrap outputs (optional but recommended)
BOOT_DIR = RESULTS_DIR / "boot_method3_out"
BOOT_DIR.mkdir(parents=True, exist_ok=True)

# Multi-run reference file (your existing one)
MULTI_RUN_REF_PATH = RESULTS_DIR / "ns_multi_runs_1000_seed415.npz"
# If your reference file lives elsewhere, override this:
# MULTI_RUN_REF_PATH = Path.home() / "Desktop" / "ns_results" / "MCGoldenSave" / "ns_multi_runs_200_seed415.npz"


# ============================================================
# Tunables
# ============================================================

BOOT_RUNS        = 5

# Keep data fixed across all runs
DATA_SEED        = 415

# Vary NS randomness across runs
NS_BASE_SEED     = 1000     # run i uses NS_BASE_SEED + i

# Vary Method (3) randomness too (recommended)
BOOT_BASE_SEED   = 2000     # run i uses BOOT_BASE_SEED + i

N_BOOT_PER_RUN   = 200
N_SHRINK         = 1

# Plot controls
MAX_SEEDS_IN_QQ  = 12
POINT_ALPHA      = 0.45
POINT_SIZE       = 12
HIST_BINS        = 30


# ============================================================
# NS configuration (forwarded into run_ns_mh_phantom)
# ============================================================

# Data
n = 600
p = 12
use_correlated_X = False
rho = 1.0
sigma_beta = 1.0
sparsity = 0.0
include_intercept = False

# Prior
tau_prior = 1.0

# NS settings
n_live = 100
ns_mcmc_steps = 20
n_iter_max = 50_000
tol_logZ = 1e-3
tol_tail = 1e-2
patience = 40
stable_repeats = 2
verbose = False
verbose_interval = 500

# MH tuning
mh_step_size = 0.10
mh_target_accept = 0.234
mh_adapt_rate = 0.05
mh_warmup_steps = 20
mh_step_size_min = 1e-6
mh_step_size_max = 10.0
mh_store_warmup = False


# ============================================================
# (A) Load multi-run reference (logZs)
# ============================================================

if not MULTI_RUN_REF_PATH.exists():
    raise FileNotFoundError(
        f"Multi-run reference file not found:\n{MULTI_RUN_REF_PATH}\n"
        "Set MULTI_RUN_REF_PATH to the correct location."
    )

ref = np.load(MULTI_RUN_REF_PATH, allow_pickle=True)
logZs_multi = np.asarray(ref["logZs"], dtype=float)
logZs_multi = logZs_multi[np.isfinite(logZs_multi)]

print(f"[multi-run ref] loaded {logZs_multi.size} finite logZ values from: {MULTI_RUN_REF_PATH.name}")
print(f"[multi-run ref] mean={logZs_multi.mean(): .6f}, sd={logZs_multi.std(ddof=1): .6f}")


# ============================================================
# Helper: save a single NS run (dead + phantom "points")
# ============================================================

def save_ns_out(ns_out: dict, ns_seed: int, data_seed: int) -> Path:
    """
    Saves minimal + point-level outputs needed for later diagnostics:
      - dead_logLs
      - phantom_bins_logL (list of arrays)  [the "extra points"]
      - trace_logZ (optional)
      - step_sizes_used (optional)
    """
    out_path = NS_RUNS_DIR / f"ns_out_seed{int(ns_seed)}.npz"

    np.savez_compressed(
        out_path,
        ns_seed=int(ns_seed),
        data_seed=int(data_seed),
        logZ=float(ns_out["logZ"]),
        H=float(ns_out.get("H", np.nan)),
        tail_logL=float(ns_out.get("tail_logL", np.nan)),

        dead_logLs=np.asarray(ns_out["dead_logLs"], dtype=float),

        # "points" for your later methods/diagnostics
        phantom_bins_logL=np.array(ns_out.get("phantom_bins_logL", []), dtype=object),

        # optional diagnostics
        trace_logZ=np.asarray(ns_out.get("trace_logZ", []), dtype=float),
        step_sizes_used=np.asarray(ns_out.get("step_sizes_used", ns_out.get("mh_step_sizes_used", [])), dtype=float),
        settings=np.array([ns_out.get("settings", {})], dtype=object),
    )
    return out_path


# ============================================================
# Helper: save a single Method (3) run (optional)
# ============================================================

def save_boot_out(out3: dict, ns_seed: int, boot_seed: int) -> Path:
    out_path = BOOT_DIR / f"boot3_nsseed{int(ns_seed)}_bootseed{int(boot_seed)}.npz"
    np.savez_compressed(
        out_path,
        ns_seed=int(ns_seed),
        boot_seed=int(boot_seed),
        logZ=np.asarray(out3.get("logZ", []), dtype=float),
        # Add more fields if you want (ESS, ladder info, etc.)
        keys=np.array(list(out3.keys()), dtype=object),
    )
    return out_path


# ============================================================
# (B) For each run: NS with different seed -> Method (3) bootstrap
# ============================================================

logZ_boot_by_seed = []
seed_list = []
ns_logZ_list = []
saved_ns_paths = []
saved_boot_paths = []

for i in range(int(BOOT_RUNS)):
    run_start = time.perf_counter()

    ns_seed   = int(NS_BASE_SEED + i)
    boot_seed = int(BOOT_BASE_SEED + i)

    seed_list.append(ns_seed)

    # --- run NS fresh (same data, different algorithm seed)
    t_ns0 = time.perf_counter()
    ns_out_i = run_ns_mh_phantom(
        # data
        n=n, p=p,
        use_correlated_X=use_correlated_X,
        rho=rho,
        sigma_beta=sigma_beta,
        sparsity=sparsity,
        include_intercept=include_intercept,
        data_seed=DATA_SEED,          # <-- fixed data

        # prior
        tau_prior=tau_prior,

        # ns
        n_live=n_live,
        ns_mcmc_steps=ns_mcmc_steps,
        n_iter_max=n_iter_max,
        ns_seed=ns_seed,              # <-- varies across runs
        tol_logZ=tol_logZ,
        tol_tail=tol_tail,
        patience=patience,
        stable_repeats=stable_repeats,
        verbose=False,
        verbose_interval=verbose_interval,

        # mh tuning
        mh_step_size=mh_step_size,
        mh_target_accept=mh_target_accept,
        mh_adapt_rate=mh_adapt_rate,
        mh_warmup_steps=mh_warmup_steps,
        mh_step_size_min=mh_step_size_min,
        mh_step_size_max=mh_step_size_max,
        mh_store_warmup=mh_store_warmup,
    )
    t_ns = time.perf_counter() - t_ns0

    ns_logZ = float(ns_out_i["logZ"])
    ns_logZ_list.append(ns_logZ)

    # --- save NS output (includes "points")
    p_ns = save_ns_out(ns_out_i, ns_seed=ns_seed, data_seed=DATA_SEED)
    saved_ns_paths.append(p_ns)

    # --- run Method (3) on THIS ns_out_i
    t_m30 = time.perf_counter()
    out3_i = method3_stochshrink_two_sided_chain(
        ns_out=ns_out_i,
        n_live=n_live,
        n_boot=int(N_BOOT_PER_RUN),
        n_shrink=int(N_SHRINK),
        seed=boot_seed,
        pool_decimals=None,
        allow_equal_lower=True,
        chain_prev=True,
        use_swish=True,
        swish_blocks=40,
        swish_mean_block_len=80,
        swish_site_sweeps_per_block=2,
        swish_p_upper=0.5,
        lag_every=10,
        print_every=0
    )
    t_m3 = time.perf_counter() - t_m30

    z = np.asarray(out3_i["logZ"], dtype=float).ravel()
    z = z[np.isfinite(z)]
    logZ_boot_by_seed.append(z)

    # --- save bootstrap output (optional, but nice)
    p_boot = save_boot_out(out3_i, ns_seed=ns_seed, boot_seed=boot_seed)
    saved_boot_paths.append(p_boot)

    run_time = time.perf_counter() - run_start
    elapsed_total = time.perf_counter() - GLOBAL_START

    print(
        f"[run {i+1}/{BOOT_RUNS}] NS seed={ns_seed} -> NS logZ={ns_logZ: .6f} | "
        f"M3 seed={boot_seed} -> n={z.size} mean={z.mean(): .6f} sd={z.std(ddof=1): .6f} | "
        f"t_NS={format_time(t_ns)} t_M3={format_time(t_m3)} t_run={format_time(run_time)} | "
        f"elapsed={format_time(elapsed_total)} ETA={eta_str(elapsed_total, i+1, int(BOOT_RUNS))} | "
        f"saved: {p_ns.name}"
    )

# pooled sample across runs
logZ_boot_pooled = np.concatenate([z for z in logZ_boot_by_seed if z.size > 0]) if logZ_boot_by_seed else np.array([])
print("\n=== Pooled Method (3) across NS seeds ===")
print(f"[pooled] n={logZ_boot_pooled.size} mean={np.mean(logZ_boot_pooled): .6f} sd={np.std(logZ_boot_pooled, ddof=1): .6f}")
print(f"[NS logZs] mean={np.mean(ns_logZ_list): .6f} sd={np.std(ns_logZ_list, ddof=1): .6f}")

print("\nSaved per-run NS outputs to:", NS_RUNS_DIR)
print("Saved per-run bootstrap outputs to:", BOOT_DIR)


# ============================================================
# (C) Q–Q plot: per-NS-seed coloured points + pooled
# ============================================================

def qq_quantiles(x: np.ndarray, y: np.ndarray):
    m = min(x.size, y.size)
    qs = (np.arange(1, m + 1) - 0.5) / m
    return np.quantile(x, qs), np.quantile(y, qs)

plt.figure(figsize=(7.2, 6.5))
cmap = plt.get_cmap("tab10")
n_show = min(len(logZ_boot_by_seed), int(MAX_SEEDS_IN_QQ))

for j in range(n_show):
    z = logZ_boot_by_seed[j]
    if z.size < 5:
        continue
    q_multi, q_boot = qq_quantiles(logZs_multi, z)
    plt.scatter(q_multi, q_boot, s=POINT_SIZE, alpha=POINT_ALPHA, color=cmap(j % 10),
                label=f"NS seed {seed_list[j]}" if j < 10 else None)

if logZ_boot_pooled.size >= 5:
    q_multi_p, q_boot_p = qq_quantiles(logZs_multi, logZ_boot_pooled)
    plt.scatter(q_multi_p, q_boot_p, s=22, alpha=0.9, color="black", label="Pooled (all NS seeds)")

lo = min(np.min(logZs_multi), np.min(logZ_boot_pooled))
hi = max(np.max(logZs_multi), np.max(logZ_boot_pooled))
plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)

plt.xlabel("Multi-run NS logZ quantiles")
plt.ylabel("Method (3) logZ quantiles")
plt.title("Q–Q: Method (3) vs Multi-run NS\n(each run uses a different NS seed; data fixed)")
plt.grid(alpha=0.3)
plt.legend(loc="best", fontsize=9, ncols=2)
plt.tight_layout()
plt.show()


# ============================================================
# (D) Overlay histograms: multi-run + pooled Method (3)
# ============================================================

plt.figure(figsize=(9.5, 4.2))
plt.hist(logZs_multi, bins=HIST_BINS, density=True, alpha=0.45, label="Multi-run NS (ref)")
plt.hist(logZ_boot_pooled, bins=HIST_BINS, density=True, alpha=0.45, label="Method (3) pooled (varying NS seeds)")
plt.title("Overlay histograms: logZ distributions")
plt.xlabel("logZ")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# (E) Optional: quick check of "points" count for first run
# ============================================================

if saved_ns_paths:
    d0 = np.load(saved_ns_paths[0], allow_pickle=True)
    dead0 = np.asarray(d0["dead_logLs"], float)
    ph0 = d0["phantom_bins_logL"]
    n_ph = int(np.sum([np.asarray(a).size for a in ph0])) if ph0.size > 0 else 0
    print(f"\n[first run points] dead={dead0.size}, phantom={n_ph}, total={dead0.size + n_ph}")

# ============================================================
# FINAL TOTAL TIMER
# ============================================================
print(f"\n[TIMER] total runtime = {format_time(time.perf_counter() - GLOBAL_START)}")

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# (B1) QQ curve comparisons: pooled vs median-of-runs vs mean-of-runs
# Uses ONLY what was saved in Chunk 1
# ============================================================

# ---- Load per-run bootstrap samples from saved files
z_runs = []
for p in saved_boot_paths:
    d = np.load(p, allow_pickle=True)
    z = np.asarray(d["logZ"], dtype=float).ravel()
    z = z[np.isfinite(z)]
    if z.size > 0:
        z_runs.append(z)

R = len(z_runs)
if R < 1:
    raise ValueError("No non-empty bootstrap runs found in saved_boot_paths.")

# ---- Pooled
z_pool = np.concatenate(z_runs)
print(f"[QQ curves] runs={R} | pooled n={z_pool.size}")

# ---- Quantile grid for curve
QQ_M = 200
qs_curve = (np.arange(1, QQ_M + 1) - 0.5) / QQ_M

# ---- Fixed x-axis: multi-run NS reference quantiles
q_multi_curve = np.quantile(logZs_multi, qs_curve)

# ---- Run-wise quantile curves
q_runs = np.vstack([np.quantile(z, qs_curve) for z in z_runs])  # shape (R, QQ_M)

# ---- Aggregate curves across runs
q_med_curve = np.median(q_runs, axis=0)
q_mean_curve = np.mean(q_runs, axis=0)

# ---- Pooled curve
q_pool_curve = np.quantile(z_pool, qs_curve)

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(7.6, 6.7))

# pooled
plt.plot(q_multi_curve, q_pool_curve, linewidth=2.4, label="Pooled curve (concat all bootstraps)")

# median-of-runs + mean-of-runs
plt.plot(q_multi_curve, q_med_curve, linewidth=2.0, linestyle="--",
         label="Median curve (median over runs at each quantile)")
plt.plot(q_multi_curve, q_mean_curve, linewidth=2.0, linestyle=":",
         label="Mean curve (mean over runs at each quantile)")

# y=x reference
lo = min(np.min(q_multi_curve), np.min(q_pool_curve), np.min(q_med_curve), np.min(q_mean_curve))
hi = max(np.max(q_multi_curve), np.max(q_pool_curve), np.max(q_med_curve), np.max(q_mean_curve))
plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)

plt.xlabel("Multi-run NS logZ quantiles")
plt.ylabel("Method (3) logZ quantiles")
plt.title("Q–Q curve comparisons: pooled vs median-of-runs vs mean-of-runs")
plt.grid(alpha=0.3)
plt.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# ============================================================
# 2nd-order bootstrap (robust: reloads boot files from BOOT_DIR)
# Requires in memory: logZs_multi, BOOT_DIR
# If BOOT_DIR not in memory (kernel restart), set it below.
# ============================================================

# -------------------------
# If you restarted the kernel, uncomment & set BOOT_DIR and MULTI_RUN_REF_PATH:
# -------------------------
REPO_ROOT = Path.cwd()
RESULTS_DIR = (REPO_ROOT / "results" / "MCGoldenSave")
if not RESULTS_DIR.exists():
    RESULTS_DIR = Path.home() / "Desktop" / "ns_results" / "MCGoldenSave"
BOOT_DIR = RESULTS_DIR / "boot_method3_out"
MULTI_RUN_REF_PATH = RESULTS_DIR / "ns_multi_runs_1000_seed415.npz"
ref = np.load(MULTI_RUN_REF_PATH, allow_pickle=True)
logZs_multi = np.asarray(ref["logZs"], dtype=float)
logZs_multi = logZs_multi[np.isfinite(logZs_multi)]

# -------------------------
# Tunables
# -------------------------
MODE          = "across"   # "single" or "across"
TARGET_NS_SEED = 1002      # used only if MODE="single" (choose which run/seed to test)

B2            = 2000
CI_ALPHA      = 0.05
INNER_N_MODE  = "same"     # "same" uses len(z_r); or set an int
QQ_M          = 200
qs_curve      = (np.arange(1, QQ_M + 1) - 0.5) / QQ_M

RNG_SEED      = 12345
PRINT_EVERY   = 1000

NS_BASE_SEED = 1000
BOOT_RUNS    = 5

# -------------------------
# Timer helpers
# -------------------------
GLOBAL_T0 = time.perf_counter()

def fmt_time(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

rng2 = np.random.default_rng(RNG_SEED)

# -------------------------
# Load per-run bootstrap samples from disk (BOOT_DIR)
# We also keep the ns_seed stored in each file so you can select one
# -------------------------
boot_files = sorted(Path(BOOT_DIR).glob("boot3_nsseed*_bootseed*.npz"))
if len(boot_files) == 0:
    raise FileNotFoundError(f"No boot files found in BOOT_DIR={BOOT_DIR}")

runs = []  # list of dicts: {"ns_seed": int, "path": Path, "z": np.ndarray}
for fp in boot_files:
    d = np.load(fp, allow_pickle=True)
    ns_seed = int(d["ns_seed"]) if "ns_seed" in d.files else None
    z = np.asarray(d["logZ"], dtype=float).ravel()
    z = z[np.isfinite(z)]
    if z.size > 0 and ns_seed is not None:
        runs.append({"ns_seed": ns_seed, "path": fp, "z": z})

if len(runs) == 0:
    raise ValueError("Found boot files, but none had usable (finite) logZ samples + ns_seed.")

# Sort runs by ns_seed for nice indexing/printing
runs = sorted(runs, key=lambda r: r["ns_seed"])

# ---- FILTER TO ONLY THE SEEDS USED IN THE CURRENT EXPERIMENT ----
allowed_seeds = set(NS_BASE_SEED + i for i in range(BOOT_RUNS))
runs = [r for r in runs if r["ns_seed"] in allowed_seeds]

# (optional but helpful sanity check)
print("Using seeds:", sorted({r["ns_seed"] for r in runs}))

# For across-seed mode: a list of arrays
z_runs = [r["z"] for r in runs]
seed_runs = [r["ns_seed"] for r in runs]
R = len(z_runs)

# Fixed x-axis quantiles (multi-run ref)
q_multi_curve = np.quantile(logZs_multi, qs_curve)

# -------------------------
# Choose baseline curve and eligible runs depending on MODE
# -------------------------
if MODE.lower() == "single":
    if TARGET_NS_SEED not in seed_runs:
        raise ValueError(
            f"TARGET_NS_SEED={TARGET_NS_SEED} not found. Available ns_seed values:\n{seed_runs}"
        )
    idx = seed_runs.index(TARGET_NS_SEED)
    z_base = z_runs[idx]
    if z_base.size < 2:
        raise ValueError(f"Selected seed {TARGET_NS_SEED} has too few points: {z_base.size}")

    q_base_curve = np.quantile(z_base, qs_curve)
    base_label = f"Method (3) quantile curve (single ns_seed={TARGET_NS_SEED})"
    base_mean = z_base.mean()
    base_sd   = z_base.std(ddof=1)

elif MODE.lower() == "across":
    if R < 2:
        raise ValueError(f"Need at least 2 non-empty runs for MODE='across'; got R={R}.")
    z_base = np.concatenate(z_runs)
    q_base_curve = np.quantile(z_base, qs_curve)
    base_label = "Pooled Method (3) quantile curve (all ns_seeds)"
    base_mean = z_base.mean()
    base_sd   = z_base.std(ddof=1)

else:
    raise ValueError("MODE must be either 'single' or 'across'.")

# -------------------------
# Storage for replicate statistics
# -------------------------
mean_rep = np.empty(B2, dtype=float)
sd_rep   = np.empty(B2, dtype=float)
q_rep    = np.empty((B2, QQ_M), dtype=float)

print("\n============================================================")
print("2nd-order bootstrap starting")
print("============================================================")
print(f"MODE          : {MODE}")
print(f"BOOT_DIR      : {BOOT_DIR}")
print(f"Runs found    : R={R}  (ns_seeds={seed_runs})")
if MODE.lower() == "single":
    print(f"TARGET_NS_SEED: {TARGET_NS_SEED}  (n={z_base.size})")
print(f"B2            : {B2}")
print(f"INNER_N_MODE  : {INNER_N_MODE}")
print("------------------------------------------------------------")

# -------------------------
# Bootstrap loop with timer + ETA
# -------------------------
loop_t0 = time.perf_counter()

for b in range(B2):
    if MODE.lower() == "single":
        # within-run resampling only
        z = z_base
        n_in = z.size if INNER_N_MODE == "same" else int(INNER_N_MODE)
        pooled_b = z[rng2.integers(0, z.size, size=n_in)]

    else:
        # across runs (outer) + within-run (inner)
        run_idx = rng2.integers(0, R, size=R)
        parts = []
        for k in run_idx:
            z = z_runs[k]
            n_in = z.size if INNER_N_MODE == "same" else int(INNER_N_MODE)
            parts.append(z[rng2.integers(0, z.size, size=n_in)])
        pooled_b = np.concatenate(parts)

    mean_rep[b] = pooled_b.mean()
    sd_rep[b]   = pooled_b.std(ddof=1) if pooled_b.size > 1 else np.nan
    q_rep[b]    = np.quantile(pooled_b, qs_curve)

    if (b + 1) % PRINT_EVERY == 0 or (b + 1) == B2:
        elapsed = time.perf_counter() - loop_t0
        per_rep = elapsed / (b + 1)
        eta = per_rep * (B2 - (b + 1))
        print(f"[{b+1:>5}/{B2}] elapsed={fmt_time(elapsed)}  per-rep≈{per_rep:.3f}s  ETA≈{fmt_time(eta)}")

# -------------------------
# CIs and bands
# -------------------------
lo_q = CI_ALPHA / 2
hi_q = 1 - CI_ALPHA / 2

mean_ci = (np.quantile(mean_rep, lo_q), np.quantile(mean_rep, hi_q))
sd_ci   = (np.nanquantile(sd_rep, lo_q), np.nanquantile(sd_rep, hi_q))

q_lo = np.quantile(q_rep, lo_q, axis=0)
q_hi = np.quantile(q_rep, hi_q, axis=0)

total_elapsed = time.perf_counter() - GLOBAL_T0

print("\n============================================================")
if MODE.lower() == "single":
    print("2nd-order bootstrap error bounds (WITHIN ONE NS seed run)")
    print("============================================================")
    print(f"[base mean logZ] {base_mean: .6f} | {100*(1-CI_ALPHA):.0f}% CI=({mean_ci[0]: .6f}, {mean_ci[1]: .6f})")
    print(f"[base sd logZ]   {base_sd: .6f} | {100*(1-CI_ALPHA):.0f}% CI=({sd_ci[0]: .6f}, {sd_ci[1]: .6f})")
else:
    print("2nd-order bootstrap error bounds (ACROSS NS seeds)")
    print("============================================================")
    print(f"[pooled mean logZ] {base_mean: .6f} | {100*(1-CI_ALPHA):.0f}% CI=({mean_ci[0]: .6f}, {mean_ci[1]: .6f})")
    print(f"[pooled sd logZ]   {base_sd: .6f} | {100*(1-CI_ALPHA):.0f}% CI=({sd_ci[0]: .6f}, {sd_ci[1]: .6f})")

print(f"\nTotal runtime: {fmt_time(total_elapsed)}")

# -------------------------
# Plot: QQ + band
# -------------------------
plt.figure(figsize=(7.6, 6.7))

plt.fill_between(
    q_multi_curve, q_lo, q_hi, alpha=0.25,
    label=f"{int(100*(1-CI_ALPHA))}% band (bootstrap, MODE={MODE})"
)

plt.plot(q_multi_curve, q_base_curve, linewidth=2.2, label=base_label)

lo = min(np.min(q_multi_curve), np.min(q_lo))
hi = max(np.max(q_multi_curve), np.max(q_hi))
plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)

plt.xlabel("Multi-run NS logZ quantiles")
plt.ylabel("Method (3) logZ quantiles")
plt.title(f"Q–Q with bootstrap band (MODE={MODE})")
plt.grid(alpha=0.3)
plt.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.show()
