from __future__ import annotations


from pathlib import Path


import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =====================================================

# USER SETTINGS

# =====================================================

LABEL = "d72_ds72_nl1800_m50_w50"

ROOT = Path(f"results/bootstrap_runs/{LABEL}")

CHUNK_DIR = ROOT / "chunk_summaries"

OUT_DIR = ROOT / "sd_diagnostics"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# Plot settings

FIG_DPI = 250

DOT_FIGSIZE = (7.6, 5.0)

BAR_FIGSIZE = (6.8, 4.6)

MARKER_SIZE = 42

SORT_BOOT_SD = True

INCLUDE_SUMMARY_BAR = True



# =====================================================

# HELPERS

# =====================================================

def finite_1d(x: np.ndarray) -> np.ndarray:

    x = np.asarray(x, dtype=float).ravel()

    return x[np.isfinite(x)]



def maybe_sorted(values: np.ndarray, do_sort: bool) -> tuple[np.ndarray, np.ndarray]:

    values = np.asarray(values, dtype=float)

    if do_sort:

        vals = np.sort(values)

    else:

        vals = values.copy()

    xpos = np.arange(1, len(vals) + 1)

    return xpos, vals



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



# =====================================================

# SD COMPUTATIONS

# =====================================================

multi_run_sd = float(np.std(ns_ref, ddof=1)) if ns_ref.size > 1 else np.nan


boot_sds = np.array(

    [np.std(finite_1d(z), ddof=1) if finite_1d(z).size > 1 else np.nan for z in boot_runs],

    dtype=float,

)


mean_boot_sd = float(np.nanmean(boot_sds))

median_boot_sd = float(np.nanmedian(boot_sds))

ratio_mean_to_multirun = mean_boot_sd / multi_run_sd if np.isfinite(multi_run_sd) and multi_run_sd != 0 else np.nan

ratio_median_to_multirun = median_boot_sd / multi_run_sd if np.isfinite(multi_run_sd) and multi_run_sd != 0 else np.nan



# =====================================================

# 1) BOOTSTRAP SD DOT PLOT

# =====================================================

x_sd, y_sd = maybe_sorted(boot_sds, SORT_BOOT_SD)


plt.figure(figsize=DOT_FIGSIZE)

plt.scatter(y_sd, x_sd, s=MARKER_SIZE, label="Run-level bootstrap SD")

plt.axvline(

    multi_run_sd,

    linestyle="--",

    linewidth=1.8,

    label=f"Multi-run NS SD = {multi_run_sd:.4f}",

)

plt.axvline(

    mean_boot_sd,

    linestyle=":",

    linewidth=1.8,

    label=f"Mean bootstrap SD = {mean_boot_sd:.4f}",

)

plt.axvline(

    median_boot_sd,

    linestyle="-.",

    linewidth=1.8,

    label=f"Median bootstrap SD = {median_boot_sd:.4f}",

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

# 2) OPTIONAL SPREAD SUMMARY BAR PLOT

# =====================================================

if INCLUDE_SUMMARY_BAR:

    summary_names = [

        "multi-run SD",

        "mean boot SD",

        "median boot SD",

    ]

    summary_vals = [

        multi_run_sd,

        mean_boot_sd,

        median_boot_sd,

    ]


    plt.figure(figsize=BAR_FIGSIZE)

    plt.bar(summary_names, summary_vals)

    plt.ylabel("logZ scale")

    plt.title(f"Spread summary\n{LABEL}")

    plt.tight_layout()

    plt.savefig(OUT_DIR / f"spread_summary_bar_{LABEL}.png", dpi=FIG_DPI)

    plt.close()

    print("Saved spread summary bar plot.")



# =====================================================

# 3) TEXT SUMMARY

# =====================================================

summary_path = OUT_DIR / f"sd_summary_{LABEL}.txt"

with open(summary_path, "w", encoding="utf-8") as f:

    f.write(f"LABEL = {LABEL}\n")

    f.write(f"CHUNK_DIR = {CHUNK_DIR}\n")

    f.write(f"OUT_DIR = {OUT_DIR}\n")

    f.write(f"Number of runs = {n_runs}\n")

    f.write(f"SORT_BOOT_SD = {SORT_BOOT_SD}\n")

    f.write(f"INCLUDE_SUMMARY_BAR = {INCLUDE_SUMMARY_BAR}\n")

    f.write("\n")

    f.write(f"Multi-run NS SD = {multi_run_sd:.10f}\n")

    f.write(f"Mean bootstrap SD = {mean_boot_sd:.10f}\n")

    f.write(f"Median bootstrap SD = {median_boot_sd:.10f}\n")

    f.write(f"Mean bootstrap SD / multi-run SD = {ratio_mean_to_multirun:.10f}\n")

    f.write(f"Median bootstrap SD / multi-run SD = {ratio_median_to_multirun:.10f}\n")

    f.write("\n")

    f.write("Per-run bootstrap SDs:\n")

    for i, sd in enumerate(boot_sds, start=1):

        f.write(f" Run {i:03d}: {sd:.10f}\n")

    f.write("\nSaved files:\n")

    for p in sorted(OUT_DIR.glob("*")):

        f.write(f" {p.name}\n")


print(f"Saved summary to {summary_path}")

print("Saved files:")

for p in sorted(OUT_DIR.glob("*")):

    print(" ", p.name)

