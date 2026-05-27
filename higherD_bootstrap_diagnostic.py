from __future__ import annotations


from pathlib import Path

import math

import numpy as np

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D



# =====================================================

# USER SETTINGS

# =====================================================


LABEL = "d72_ds72_nl1800_m50_w50"

ROOT = Path(f"results/bootstrap_runs/{LABEL}")

CHUNK_DIR = ROOT / "chunk_summaries"

OUT_DIR = ROOT / "diagnostics"

OUT_DIR.mkdir(parents=True, exist_ok=True)


QQ_N_QUANTILES = 200

QQ_RUNS_PER_FIG = 10

QQ_AGG_MODE = "mean"   # "median" or "mean"


SECOND_ORDER_RUN_INDEX = 0

SECOND_ORDER_REPS = 1000

SECOND_ORDER_SEED = 123


WASS_K_LIST = [10, 30, 50, 70, 100, None]   # None => all runs

POINT_COV_RUNS_PER_FIG = 50


FIG_DPI = 220

ALPHA_LINE = 0.55



# =====================================================

# HELPERS

# =====================================================


def finite_1d(x: np.ndarray) -> np.ndarray:

    x = np.asarray(x, dtype=float).ravel()

    return x[np.isfinite(x)]



def bootstrap_ci(sample: np.ndarray, alpha: float) -> tuple[float, float]:

    lo = 100.0 * (alpha / 2.0)

    hi = 100.0 * (1.0 - alpha / 2.0)

    return tuple(np.percentile(sample, [lo, hi]))



def qcurve(x: np.ndarray, n_q: int = 200) -> tuple[np.ndarray, np.ndarray]:

    q = np.linspace(0.01, 0.99, n_q)

    return q, np.quantile(finite_1d(x), q)



def aggregate_curves(curves: np.ndarray, mode: str) -> np.ndarray:

    if mode == "mean":

        return np.mean(curves, axis=0)

    if mode == "median":

        return np.median(curves, axis=0)

    raise ValueError(f"Unknown aggregation mode: {mode}")



def approx_w1_from_quantiles(x: np.ndarray, y: np.ndarray, n_q: int = 200) -> float:

    x = finite_1d(x)

    y = finite_1d(y)

    if x.size == 0 or y.size == 0:

        return np.nan

    q = np.linspace(0.01, 0.99, n_q)

    xq = np.quantile(x, q)

    yq = np.quantile(y, q)

    return float(np.mean(np.abs(xq - yq)))



def approx_w1_curve_to_ref(curve: np.ndarray, ref_curve: np.ndarray) -> float:

    curve = np.asarray(curve, dtype=float)

    ref_curve = np.asarray(ref_curve, dtype=float)

    return float(np.mean(np.abs(curve - ref_curve)))



def percentile_of_value(sample: np.ndarray, value: float) -> float:

    sample = finite_1d(sample)

    return float(np.mean(sample <= value))



def load_all_runs_from_chunks(chunk_dir: Path) -> tuple[list[np.ndarray], np.ndarray]:

    """

    Returns

    -------

    boot_runs : list[np.ndarray]

        One bootstrap logZ array per NS run.

    ns_logZs : np.ndarray

        Corresponding NS logZ values, same ordering as boot_runs.

    """

    chunk_paths = sorted(chunk_dir.glob("*.npz"))

    if not chunk_paths:

        raise FileNotFoundError(f"No chunk summaries found in {chunk_dir}")


    boot_runs: list[np.ndarray] = []

    ns_logZs_all: list[float] = []


    for cp in chunk_paths:

        dat = np.load(cp, allow_pickle=True)


        if "saved_boot_paths" not in dat.files or "ns_logZs" not in dat.files:

            raise KeyError(f"{cp} missing saved_boot_paths or ns_logZs")


        boot_paths = [Path(p) for p in dat["saved_boot_paths"]]

        ns_logZs = np.asarray(dat["ns_logZs"], dtype=float)


        if len(boot_paths) != len(ns_logZs):

            raise ValueError(f"{cp}: saved_boot_paths and ns_logZs lengths differ")


        for bp, nsz in zip(boot_paths, ns_logZs):

            if not bp.exists():

                raise FileNotFoundError(f"Saved bootstrap path missing: {bp}")


            bdat = np.load(bp, allow_pickle=True)

            if "logZ" not in bdat.files:

                raise KeyError(f"{bp} missing logZ")


            z = finite_1d(bdat["logZ"])

            if z.size == 0:

                raise RuntimeError(f"{bp} has empty finite logZ array")


            boot_runs.append(z)

            ns_logZs_all.append(float(nsz))


    return boot_runs, np.asarray(ns_logZs_all, dtype=float)



# =====================================================

# LOAD DATA

# =====================================================


boot_runs, ns_logZs = load_all_runs_from_chunks(CHUNK_DIR)

ns_ref = finite_1d(ns_logZs)


if len(boot_runs) == 0:

    raise RuntimeError("No bootstrap runs loaded.")

if ns_ref.size == 0:

    raise RuntimeError("No NS logZs loaded.")


n_runs = len(boot_runs)

print(f"Loaded {n_runs} bootstrap runs from {CHUNK_DIR}")

print(f"NS reference runs: {ns_ref.size}")


q_ref, ref_curve = qcurve(ns_ref, n_q=QQ_N_QUANTILES)

multi_run_mean = float(np.mean(ns_ref))

multi_run_median = float(np.median(ns_ref))

multi_run_sd = float(np.std(ns_ref, ddof=1)) if ns_ref.size > 1 else np.nan



# =====================================================

# RUN-LEVEL SUMMARY STATISTICS

# =====================================================


boot_means = np.array([np.mean(finite_1d(z)) for z in boot_runs], dtype=float)

boot_medians = np.array([np.median(finite_1d(z)) for z in boot_runs], dtype=float)

boot_sds = np.array([

    np.std(finite_1d(z), ddof=1) if finite_1d(z).size > 1 else np.nan

    for z in boot_runs

], dtype=float)


ns_minus_boot_mean = ns_ref - boot_means

ns_minus_boot_median = ns_ref - boot_medians

z_scores = ns_minus_boot_mean / boot_sds

boot_percentiles_of_ns = np.array(

    [percentile_of_value(z, ns_val) for z, ns_val in zip(boot_runs, ns_ref)],

    dtype=float

)


point_cover_90 = []

point_cover_95 = []

point_cover_99 = []


for z, ns_val in zip(boot_runs, ns_ref):

    lo90, hi90 = bootstrap_ci(z, 0.10)

    lo95, hi95 = bootstrap_ci(z, 0.05)

    lo99, hi99 = bootstrap_ci(z, 0.01)


    point_cover_90.append(lo90 <= ns_val <= hi90)

    point_cover_95.append(lo95 <= ns_val <= hi95)

    point_cover_99.append(lo99 <= ns_val <= hi99)


point_cover_90 = np.array(point_cover_90, dtype=bool)

point_cover_95 = np.array(point_cover_95, dtype=bool)

point_cover_99 = np.array(point_cover_99, dtype=bool)



# =====================================================

# 1) QQ PAGES

# =====================================================


n_figs = math.ceil(n_runs / QQ_RUNS_PER_FIG)


for fig_idx in range(n_figs):

    a = fig_idx * QQ_RUNS_PER_FIG

    b = min(n_runs, (fig_idx + 1) * QQ_RUNS_PER_FIG)

    subset = boot_runs[a:b]


    plt.figure(figsize=(8.2, 6.2))

    subset_curves = []


    for j, z in enumerate(subset, start=a):

        _, zq = qcurve(z, n_q=QQ_N_QUANTILES)

        subset_curves.append(zq)

        plt.plot(ref_curve, zq, linewidth=1.3, alpha=ALPHA_LINE, label=f"Run {j+1}")


    subset_curves = np.asarray(subset_curves, dtype=float)

    agg_curve = aggregate_curves(subset_curves, QQ_AGG_MODE)


    plt.plot(

        ref_curve,

        agg_curve,

        color="black",

        linewidth=2.6,

        label=f"{QQ_AGG_MODE.capitalize()} across runs",

    )


    mn = float(min(np.min(ref_curve), np.min(agg_curve)))

    mx = float(max(np.max(ref_curve), np.max(agg_curve)))

    plt.plot([mn, mx], [mn, mx], "--", linewidth=1.5)


    plt.xlabel("Multi-run NS logZ quantiles")

    plt.ylabel("Method (3) bootstrap logZ quantiles")

    plt.title(

        f"Q-Q: Method (3) vs multi-run NS\n"

        f"Runs {a+1}–{b} ({LABEL}, AGG={QQ_AGG_MODE})"

    )

    plt.legend(ncol=2, fontsize=8)

    plt.tight_layout()

    plt.savefig(OUT_DIR / f"qq_runs_{a+1:03d}_{b:03d}_{QQ_AGG_MODE}_{LABEL}.png", dpi=FIG_DPI)

    plt.close()


print("Saved QQ pages.")



# =====================================================

# 2) SECOND-ORDER BOOTSTRAP BAND FOR ONE RUN

# =====================================================


z0 = finite_1d(boot_runs[SECOND_ORDER_RUN_INDEX])

rng = np.random.default_rng(SECOND_ORDER_SEED)


q = np.linspace(0.01, 0.99, QQ_N_QUANTILES)

z0_curve = np.quantile(z0, q)


boot2_curves = np.empty((SECOND_ORDER_REPS, QQ_N_QUANTILES), dtype=float)

for r in range(SECOND_ORDER_REPS):

    samp = rng.choice(z0, size=z0.size, replace=True)

    boot2_curves[r, :] = np.quantile(samp, q)


lo = np.quantile(boot2_curves, 0.025, axis=0)

hi = np.quantile(boot2_curves, 0.975, axis=0)


plt.figure(figsize=(8.0, 6.0))

plt.fill_between(ref_curve, lo, hi, alpha=0.22, label="95% band (2nd-order bootstrap)")

plt.plot(ref_curve, z0_curve, linewidth=2.0, label=f"Run {SECOND_ORDER_RUN_INDEX+1} quantile curve")


mn = float(min(np.min(ref_curve), np.min(lo)))

mx = float(max(np.max(ref_curve), np.max(hi)))

plt.plot([mn, mx], [mn, mx], "--", linewidth=1.5)


plt.xlabel("Multi-run NS logZ quantiles")

plt.ylabel("Method (3) bootstrap logZ quantiles")

plt.title(f"Q-Q with second-order bootstrap band\nRun {SECOND_ORDER_RUN_INDEX+1} ({LABEL})")

plt.legend()

plt.tight_layout()

plt.savefig(OUT_DIR / f"qq_second_order_run{SECOND_ORDER_RUN_INDEX+1:03d}_{LABEL}.png", dpi=FIG_DPI)

plt.close()


print("Saved second-order QQ band.")



# =====================================================

# 3) WASSERSTEIN VS k

# =====================================================


for K in WASS_K_LIST:

    if K is None:

        K_use = n_runs

        suffix = "all"

    else:

        K_use = min(K, n_runs)

        suffix = f"first_{K_use}"


    mean_vals = []

    median_vals = []

    ks = np.arange(1, K_use + 1)


    for k in ks:

        subset = boot_runs[:k]

        run_curves = np.array([np.quantile(finite_1d(z), q_ref) for z in subset], dtype=float)


        mean_curve = aggregate_curves(run_curves, "mean")

        median_curve = aggregate_curves(run_curves, "median")


        mean_vals.append(approx_w1_curve_to_ref(mean_curve, ref_curve))

        median_vals.append(approx_w1_curve_to_ref(median_curve, ref_curve))


    plt.figure(figsize=(8.2, 5.6))

    plt.plot(ks, mean_vals, marker="o", linewidth=1.6, label="mean")

    plt.plot(ks, median_vals, marker="o", linewidth=1.6, label="median")

    plt.xlabel("Number of NS runs used (k)")

    plt.ylabel("Approx. Wasserstein distance")

    plt.title(f"Wasserstein vs k ({suffix})\n{LABEL}")

    plt.legend()

    plt.tight_layout()

    plt.savefig(OUT_DIR / f"wasserstein_vs_k_{suffix}_{LABEL}.png", dpi=FIG_DPI)

    plt.close()


print("Saved Wasserstein plots.")



# =====================================================

# 4) POINT-CONTAINMENT PAGES

#    (Does bootstrap CI contain the corresponding NS run value?)

# =====================================================


n_cov_figs = math.ceil(n_runs / POINT_COV_RUNS_PER_FIG)


for fig_idx in range(n_cov_figs):

    a = fig_idx * POINT_COV_RUNS_PER_FIG

    b = min(n_runs, (fig_idx + 1) * POINT_COV_RUNS_PER_FIG)

    subset = boot_runs[a:b]

    ns_subset = ns_ref[a:b]


    fig_h = max(7.0, 0.26 * len(subset) + 2.0)

    plt.figure(figsize=(9.0, fig_h))


    for row, (z, ns_val) in enumerate(zip(subset, ns_subset), start=1):

        z = finite_1d(z)

        m = float(np.mean(z))


        lo90, hi90 = bootstrap_ci(z, alpha=0.10)

        lo95, hi95 = bootstrap_ci(z, alpha=0.05)

        lo99, hi99 = bootstrap_ci(z, alpha=0.01)


        hit90 = (lo90 <= ns_val <= hi90)

        hit95 = (lo95 <= ns_val <= hi95)

        hit99 = (lo99 <= ns_val <= hi99)


        c90 = "tab:green" if hit90 else "tab:red"

        c95 = "tab:green" if hit95 else "tab:red"

        c99 = "tab:green" if hit99 else "tab:red"


        plt.hlines(row, lo99, hi99, color=c99, linewidth=1.8, alpha=0.30)

        plt.hlines(row, lo95, hi95, color=c95, linewidth=1.8, alpha=0.55)

        plt.hlines(row, lo90, hi90, color=c90, linewidth=1.8, alpha=0.90)


        plt.plot(ns_val, row, "o", markersize=3.2, alpha=0.85)

        plt.plot(m, row, "x", markersize=4.0, alpha=0.85)


    plt.xlabel("logZ")

    plt.ylabel("Run index within page")

    plt.title(f"Bootstrap CI containment of corresponding NS run\nRuns {a+1}–{b} ({LABEL})")


    legend_items = [

        Line2D([0], [0], color="tab:green", lw=1.8, alpha=0.90, label="90% CI"),

        Line2D([0], [0], color="tab:green", lw=1.8, alpha=0.55, label="95% CI"),

        Line2D([0], [0], color="tab:green", lw=1.8, alpha=0.30, label="99% CI"),

        Line2D([0], [0], marker="o", color="black", lw=0, label="NS run logZ"),

        Line2D([0], [0], marker="x", color="black", lw=0, label="Bootstrap mean"),

    ]

    plt.legend(handles=legend_items, loc="best")

    plt.tight_layout()

    plt.savefig(OUT_DIR / f"point_containment_runs_{a+1:03d}_{b:03d}_{LABEL}.png", dpi=FIG_DPI)

    plt.close()


print("Saved point-containment pages.")



# =====================================================

# 5) HISTOGRAM: BOOTSTRAP SDs VS MULTI-RUN SD

# =====================================================


plt.figure(figsize=(8.0, 5.8))

plt.hist(boot_sds[np.isfinite(boot_sds)], bins=20, alpha=0.8)

plt.axvline(multi_run_sd, linestyle="--", linewidth=2.0, label=f"Multi-run NS SD = {multi_run_sd:.4f}")

plt.xlabel("Bootstrap SD by run")

plt.ylabel("Count")

plt.title(f"Bootstrap SD distribution vs multi-run NS SD\n{LABEL}")

plt.legend()

plt.tight_layout()

plt.savefig(OUT_DIR / f"bootstrap_sd_hist_{LABEL}.png", dpi=FIG_DPI)

plt.close()


print("Saved bootstrap SD histogram.")



# =====================================================

# 6) HISTOGRAM: STANDARDISED Z-SCORES

# =====================================================


finite_z = z_scores[np.isfinite(z_scores)]


plt.figure(figsize=(8.0, 5.8))

plt.hist(finite_z, bins=20, alpha=0.8)

plt.axvline(0.0, linestyle="--", linewidth=1.8, label="Ideal centre = 0")

plt.xlabel(r"$z_r = (\mathrm{NS}_r - \mathrm{boot\ mean}_r) / \mathrm{boot\ sd}_r$")

plt.ylabel("Count")

plt.title(f"Standardised calibration scores\n{LABEL}")

plt.legend()

plt.tight_layout()

plt.savefig(OUT_DIR / f"zscore_hist_{LABEL}.png", dpi=FIG_DPI)

plt.close()


print("Saved z-score histogram.")



# =====================================================

# 7) HISTOGRAM: NS PERCENTILE WITHIN BOOTSTRAP

# =====================================================


plt.figure(figsize=(8.0, 5.8))

plt.hist(boot_percentiles_of_ns, bins=20, range=(0.0, 1.0), alpha=0.8)

plt.axvline(0.5, linestyle="--", linewidth=1.8, label="Centre = 0.5")

plt.xlabel("Percentile of corresponding NS run within bootstrap distribution")

plt.ylabel("Count")

plt.title(f"Rank / percentile diagnostic\n{LABEL}")

plt.legend()

plt.tight_layout()

plt.savefig(OUT_DIR / f"ns_percentile_hist_{LABEL}.png", dpi=FIG_DPI)

plt.close()


print("Saved percentile histogram.")



# =====================================================

# 8) TEXT SUMMARY

# =====================================================


summary_txt = OUT_DIR / f"diagnostic_summary_generic_{LABEL}.txt"


with open(summary_txt, "w", encoding="utf-8") as f:

    f.write(f"LABEL = {LABEL}\n")

    f.write(f"Chunk directory = {CHUNK_DIR}\n")

    f.write(f"Output directory = {OUT_DIR}\n")

    f.write(f"Number of runs = {n_runs}\n")

    f.write(f"QQ_AGG_MODE = {QQ_AGG_MODE}\n")

    f.write(f"QQ_N_QUANTILES = {QQ_N_QUANTILES}\n")

    f.write(f"SECOND_ORDER_RUN_INDEX = {SECOND_ORDER_RUN_INDEX}\n")

    f.write(f"SECOND_ORDER_REPS = {SECOND_ORDER_REPS}\n")

    f.write("\n")

    f.write(f"Multi-run NS mean   = {multi_run_mean:.10f}\n")

    f.write(f"Multi-run NS median = {multi_run_median:.10f}\n")

    f.write(f"Multi-run NS sd     = {multi_run_sd:.10f}\n")

    f.write("\n")

    f.write(f"Mean(ns - boot_mean)   = {np.mean(ns_minus_boot_mean):.10f}\n")

    f.write(f"Median(ns - boot_mean) = {np.median(ns_minus_boot_mean):.10f}\n")

    f.write(f"Mean(ns - boot_median) = {np.mean(ns_minus_boot_median):.10f}\n")

    f.write(f"Median bootstrap sd    = {np.nanmedian(boot_sds):.10f}\n")

    f.write(f"Mean bootstrap sd      = {np.nanmean(boot_sds):.10f}\n")

    f.write("\n")

    f.write(f"Point-containment rate 90% = {np.mean(point_cover_90):.6f}\n")

    f.write(f"Point-containment rate 95% = {np.mean(point_cover_95):.6f}\n")

    f.write(f"Point-containment rate 99% = {np.mean(point_cover_99):.6f}\n")

    f.write("\n")

    f.write(f"Mean z-score   = {np.nanmean(z_scores):.10f}\n")

    f.write(f"SD z-score     = {np.nanstd(z_scores, ddof=1):.10f}\n")

    f.write(f"Mean percentile of NS within bootstrap = {np.nanmean(boot_percentiles_of_ns):.10f}\n")

    f.write("\nSaved files:\n")

    for p in sorted(OUT_DIR.glob("*")):

        f.write(f"  {p.name}\n")


print(f"Saved diagnostics to: {OUT_DIR}")

for p in sorted(OUT_DIR.glob("*")):

    print(" ", p.name)
