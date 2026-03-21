from __future__ import annotations


from pathlib import Path

import numpy as np

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt



# =====================================================

# USER SETTINGS

# =====================================================


LABEL = "d36_ds36_nl900_m50_w50"


ROOT = Path(f"results/bootstrap_runs/{LABEL}")

CHUNK_DIR = ROOT / "chunk_summaries"

OUT_DIR = ROOT / "report_plots"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# Aggregation choice: "mean" or "median"

AGG_MODE = "mean"


# Ribbon choice across runs

# e.g. (0.25, 0.75) for IQR ribbon

# e.g. (0.10, 0.90) for wider ribbon

RIBBON_QUANTILES = (0.25, 0.75)


QQ_N_QUANTILES = 200


FIG_DPI = 250


# Optional: sort runs for cleaner dot plots

SORT_BOOT_SD = True

SORT_PERCENTILES = False

SORT_ZSCORES = False


# Styling

QQ_FIGSIZE = (7.8, 6.0)

DOT_FIGSIZE = (7.4, 4.8)

MARKER_SIZE = 42



# =====================================================

# HELPERS

# =====================================================


def finite_1d(x: np.ndarray) -> np.ndarray:

    x = np.asarray(x, dtype=float).ravel()

    return x[np.isfinite(x)]



def qcurve(x: np.ndarray, n_q: int = 200) -> tuple[np.ndarray, np.ndarray]:

    q = np.linspace(0.01, 0.99, n_q)

    return q, np.quantile(finite_1d(x), q)



def aggregate_curves(curves: np.ndarray, mode: str) -> np.ndarray:

    if mode == "mean":

        return np.mean(curves, axis=0)

    if mode == "median":

        return np.median(curves, axis=0)

    raise ValueError(f"Unknown AGG_MODE: {mode}")



def percentile_of_value(sample: np.ndarray, value: float) -> float:

    sample = finite_1d(sample)

    return float(np.mean(sample <= value))



def load_all_runs_from_chunks(chunk_dir: Path) -> tuple[list[np.ndarray], np.ndarray]:

    """

    Returns

    -------

    boot_runs : list[np.ndarray]

        Bootstrap logZ samples, one array per NS run.

    ns_logZs : np.ndarray

        Corresponding NS logZ values.

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

                raise FileNotFoundError(f"Missing bootstrap file: {bp}")


            bdat = np.load(bp, allow_pickle=True)

            if "logZ" not in bdat.files:

                raise KeyError(f"{bp} missing logZ")


            z = finite_1d(bdat["logZ"])

            if z.size == 0:

                raise RuntimeError(f"{bp} has no finite logZ values")


            boot_runs.append(z)

            ns_logZs_all.append(float(nsz))


    return boot_runs, np.asarray(ns_logZs_all, dtype=float)



def maybe_sorted(values: np.ndarray, do_sort: bool) -> tuple[np.ndarray, np.ndarray]:

    """

    Returns sorted values and corresponding display positions.

    """

    values = np.asarray(values, dtype=float)

    if do_sort:

        vals = np.sort(values)

    else:

        vals = values.copy()

    xpos = np.arange(1, len(vals) + 1)

    return xpos, vals



# =====================================================

# LOAD DATA

# =====================================================


boot_runs, ns_logZs = load_all_runs_from_chunks(CHUNK_DIR)

ns_ref = finite_1d(ns_logZs)


if len(boot_runs) == 0:

    raise RuntimeError("No bootstrap runs loaded.")

if ns_ref.size == 0:

    raise RuntimeError("No NS logZ values loaded.")

if len(boot_runs) != len(ns_ref):

    raise RuntimeError("boot_runs and ns_ref lengths do not match.")


n_runs = len(boot_runs)


print(f"Loaded {n_runs} runs for {LABEL}")

print(f"Chunk directory: {CHUNK_DIR}")

print(f"Output directory: {OUT_DIR}")


# Multi-run reference distribution

q_ref, ref_curve = qcurve(ns_ref, n_q=QQ_N_QUANTILES)

multi_run_mean = float(np.mean(ns_ref))

multi_run_median = float(np.median(ns_ref))

multi_run_sd = float(np.std(ns_ref, ddof=1)) if ns_ref.size > 1 else np.nan


# Per-run bootstrap summaries

boot_means = np.array([np.mean(finite_1d(z)) for z in boot_runs], dtype=float)

boot_medians = np.array([np.median(finite_1d(z)) for z in boot_runs], dtype=float)

boot_sds = np.array(

    [np.std(finite_1d(z), ddof=1) if finite_1d(z).size > 1 else np.nan for z in boot_runs],

    dtype=float,

)


percentiles_of_ns = np.array(

    [percentile_of_value(z, ns_val) for z, ns_val in zip(boot_runs, ns_ref)],

    dtype=float,

)


z_scores = (ns_ref - boot_means) / boot_sds


# Run-level quantile curves

run_curves = np.array([np.quantile(finite_1d(z), q_ref) for z in boot_runs], dtype=float)


agg_curve = aggregate_curves(run_curves, AGG_MODE)

rib_lo = np.quantile(run_curves, RIBBON_QUANTILES[0], axis=0)

rib_hi = np.quantile(run_curves, RIBBON_QUANTILES[1], axis=0)



# =====================================================

# 1) QQ RIBBON PLOT

# =====================================================


plt.figure(figsize=QQ_FIGSIZE)


plt.fill_between(

    ref_curve,

    rib_lo,

    rib_hi,

    alpha=0.22,

    label=f"Across-run ribbon ({int(100*RIBBON_QUANTILES[0])}–{int(100*RIBBON_QUANTILES[1])}%)",

)


plt.plot(

    ref_curve,

    agg_curve,

    linewidth=2.6,

    label=f"{AGG_MODE.capitalize()} bootstrap quantile curve",

)


mn = float(min(np.min(ref_curve), np.min(rib_lo), np.min(agg_curve)))

mx = float(max(np.max(ref_curve), np.max(rib_hi), np.max(agg_curve)))

plt.plot([mn, mx], [mn, mx], "--", linewidth=1.6, label="Identity line")


plt.xlabel("Multi-run NS logZ quantiles")

plt.ylabel("Bootstrap logZ quantiles")

plt.title(f"Q-Q comparison: bootstrap vs multi-run NS\n{LABEL} ({AGG_MODE} aggregate)")

plt.legend()

plt.tight_layout()

plt.savefig(OUT_DIR / f"qq_ribbon_{AGG_MODE}_{LABEL}.png", dpi=FIG_DPI)

plt.close()


print("Saved QQ ribbon plot.")



# =====================================================

# 2) BOOTSTRAP SD DOT PLOT

# =====================================================


x_sd, y_sd = maybe_sorted(boot_sds, SORT_BOOT_SD)


plt.figure(figsize=DOT_FIGSIZE)

plt.scatter(y_sd, x_sd, s=MARKER_SIZE)

plt.axvline(

    multi_run_sd,

    linestyle="--",

    linewidth=1.8,

    label=f"Multi-run NS SD = {multi_run_sd:.4f}",

)


plt.xlabel("Run-level bootstrap SD")

plt.ylabel("Ordered run index" if SORT_BOOT_SD else "Run index")

plt.title(f"Bootstrap SDs compared with empirical multi-run SD\n{LABEL}")

plt.legend()

plt.tight_layout()

plt.savefig(OUT_DIR / f"bootstrap_sd_dotplot_{LABEL}.png", dpi=FIG_DPI)

plt.close()


print("Saved bootstrap SD dot plot.")



# =====================================================

# 3) NS-WITHIN-BOOTSTRAP PERCENTILE DOT PLOT

# =====================================================


x_pct, y_pct = maybe_sorted(percentiles_of_ns, SORT_PERCENTILES)


plt.figure(figsize=DOT_FIGSIZE)

plt.scatter(y_pct, x_pct, s=MARKER_SIZE)

plt.axvline(0.5, linestyle="--", linewidth=1.8, label="Ideal centre = 0.5")


plt.xlim(-0.02, 1.02)

plt.xlabel("Percentile of corresponding NS run within bootstrap distribution")

plt.ylabel("Ordered run index" if SORT_PERCENTILES else "Run index")

plt.title(f"Bootstrap centring diagnostic\n{LABEL}")

plt.legend()

plt.tight_layout()

plt.savefig(OUT_DIR / f"percentile_dotplot_{LABEL}.png", dpi=FIG_DPI)

plt.close()


print("Saved percentile dot plot.")



# =====================================================

# 4) Z-SCORE DOT PLOT

# =====================================================


finite_mask = np.isfinite(z_scores)

z_scores_finite = z_scores[finite_mask]


x_z, y_z = maybe_sorted(z_scores_finite, SORT_ZSCORES)


plt.figure(figsize=DOT_FIGSIZE)

plt.scatter(y_z, x_z, s=MARKER_SIZE)

plt.axvline(0.0, linestyle="--", linewidth=1.8, label="Ideal centre = 0")


plt.xlabel(r"Standardised score $(\mathrm{NS}_r - \mathrm{boot\ mean}_r)/\mathrm{boot\ sd}_r$")

plt.ylabel("Ordered run index" if SORT_ZSCORES else "Run index")

plt.title(f"Standardised calibration scores\n{LABEL}")

plt.legend()

plt.tight_layout()

plt.savefig(OUT_DIR / f"zscore_dotplot_{LABEL}.png", dpi=FIG_DPI)

plt.close()


print("Saved z-score dot plot.")



# =====================================================

# 5) OPTIONAL: SMALL SUMMARY BAR FOR CENTRE/SPREAD

# =====================================================


summary_names = [

    "multi-run SD",

    "mean boot SD",

    "median boot SD",

]

summary_vals = [

    multi_run_sd,

    float(np.nanmean(boot_sds)),

    float(np.nanmedian(boot_sds)),

]


plt.figure(figsize=(6.8, 4.6))

plt.bar(summary_names, summary_vals)

plt.ylabel("logZ scale")

plt.title(f"Spread summary\n{LABEL}")

plt.tight_layout()

plt.savefig(OUT_DIR / f"spread_summary_bar_{LABEL}.png", dpi=FIG_DPI)

plt.close()


print("Saved spread summary bar plot.")



# =====================================================

# 6) TEXT SUMMARY

# =====================================================


summary_path = OUT_DIR / f"report_summary_{AGG_MODE}_{LABEL}.txt"


with open(summary_path, "w", encoding="utf-8") as f:

    f.write(f"LABEL = {LABEL}\n")

    f.write(f"CHUNK_DIR = {CHUNK_DIR}\n")

    f.write(f"OUT_DIR = {OUT_DIR}\n")

    f.write(f"Number of runs = {n_runs}\n")

    f.write(f"AGG_MODE = {AGG_MODE}\n")

    f.write(f"RIBBON_QUANTILES = {RIBBON_QUANTILES}\n")

    f.write(f"QQ_N_QUANTILES = {QQ_N_QUANTILES}\n")

    f.write("\n")


    f.write(f"Multi-run NS mean   = {multi_run_mean:.10f}\n")

    f.write(f"Multi-run NS median = {multi_run_median:.10f}\n")

    f.write(f"Multi-run NS sd     = {multi_run_sd:.10f}\n")

    f.write("\n")


    f.write(f"Mean bootstrap mean   = {float(np.mean(boot_means)):.10f}\n")

    f.write(f"Median bootstrap mean = {float(np.median(boot_means)):.10f}\n")

    f.write(f"Mean bootstrap median = {float(np.mean(boot_medians)):.10f}\n")

    f.write(f"Median bootstrap SD   = {float(np.nanmedian(boot_sds)):.10f}\n")

    f.write(f"Mean bootstrap SD     = {float(np.nanmean(boot_sds)):.10f}\n")

    f.write("\n")


    f.write(f"Mean(NS - boot mean)     = {float(np.mean(ns_ref - boot_means)):.10f}\n")

    f.write(f"Median(NS - boot mean)   = {float(np.median(ns_ref - boot_means)):.10f}\n")

    f.write(f"Mean(NS - boot median)   = {float(np.mean(ns_ref - boot_medians)):.10f}\n")

    f.write(f"Median(NS - boot median) = {float(np.median(ns_ref - boot_medians)):.10f}\n")

    f.write("\n")


    f.write(f"Mean percentile of NS within bootstrap = {float(np.mean(percentiles_of_ns)):.10f}\n")

    f.write(f"Median percentile of NS within bootstrap = {float(np.median(percentiles_of_ns)):.10f}\n")

    f.write(f"Mean z-score = {float(np.nanmean(z_scores)):.10f}\n")

    f.write(f"SD z-score   = {float(np.nanstd(z_scores, ddof=1)):.10f}\n")

    f.write("\n")


    f.write("Saved files:\n")

    for p in sorted(OUT_DIR.glob("*")):

        f.write(f"  {p.name}\n")


print(f"Saved summary to {summary_path}")

print("Saved files:")

for p in sorted(OUT_DIR.glob("*")):

    print(" ", p.name)
