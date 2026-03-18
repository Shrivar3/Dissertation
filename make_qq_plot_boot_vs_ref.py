from __future__ import annotations


import sys

from pathlib import Path


import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================

# PATH SETUP

# ============================================================


REPO_ROOT = Path(__file__).resolve().parent

RESULTS_ROOT = REPO_ROOT / "results"

BOOT_ROOT = RESULTS_ROOT / "bootstrap_runs"

QQ_DIR = RESULTS_ROOT / "qq_plots"

QQ_DIR.mkdir(parents=True, exist_ok=True)


MULTI_RUN_REF_PATH = RESULTS_ROOT / "large_runs" / "combined_1000_runs.npz"

BOOT_SUMMARY_PATH = sorted(BOOT_ROOT.glob("bootstrap_summary_*.npz"))[-1]

# Uses latest summary file automatically.


MAX_SEEDS_IN_QQ = 12

POINT_ALPHA = 0.45

POINT_SIZE = 12



def qq_quantiles(x: np.ndarray, y: np.ndarray):

    m = min(x.size, y.size)

    qs = (np.arange(1, m + 1) - 0.5) / m

    return np.quantile(x, qs), np.quantile(y, qs)



ref = np.load(MULTI_RUN_REF_PATH, allow_pickle=True)

logZs_multi = np.asarray(ref["logZs"], dtype=float)

logZs_multi = logZs_multi[np.isfinite(logZs_multi)]


summary = np.load(BOOT_SUMMARY_PATH, allow_pickle=True)

tag = str(summary["tag"][0])

pooled_boot = np.asarray(summary["pooled_boot_logZ"], dtype=float)

pooled_boot = pooled_boot[np.isfinite(pooled_boot)]


boot_paths = [Path(p) for p in summary["saved_boot_paths"]]

seed_list = np.asarray(summary["ns_seeds"], dtype=int)


logZ_boot_by_seed = []

for p in boot_paths:

    d = np.load(p, allow_pickle=True)

    z = np.asarray(d["logZ"], dtype=float).reshape(-1)

    z = z[np.isfinite(z)]

    logZ_boot_by_seed.append(z)


plt.figure(figsize=(7.2, 6.5))

cmap = plt.get_cmap("tab10")

n_show = min(len(logZ_boot_by_seed), int(MAX_SEEDS_IN_QQ))


for j in range(n_show):

    z = logZ_boot_by_seed[j]

    if z.size < 5:

        continue

    q_multi, q_boot = qq_quantiles(logZs_multi, z)

    plt.scatter(

        q_multi,

        q_boot,

        s=POINT_SIZE,

        alpha=POINT_ALPHA,

        color=cmap(j % 10),

        label=f"NS seed {seed_list[j]}" if j < 10 else None,

    )


if pooled_boot.size >= 5:

    q_multi_p, q_boot_p = qq_quantiles(logZs_multi, pooled_boot)

    plt.scatter(

        q_multi_p,

        q_boot_p,

        s=22,

        alpha=0.9,

        color="black",

        label="Pooled (all NS seeds)",

    )


lo = min(np.min(logZs_multi), np.min(pooled_boot))

hi = max(np.max(logZs_multi), np.max(pooled_boot))

plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)


plt.xlabel("Multi-run NS logZ quantiles")

plt.ylabel("Method (3) logZ quantiles")

plt.title(f"Q–Q: Method (3) vs Multi-run NS\n(tag={tag})")

plt.grid(alpha=0.3)

plt.legend(loc="best", fontsize=9, ncols=2)

plt.tight_layout()


out_path = QQ_DIR / f"qq_boot_vs_ref_{tag}.png"

plt.savefig(out_path, dpi=220)

print(f"Saved Q–Q plot to: {out_path}")
