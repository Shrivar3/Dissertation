from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt


RESULTS_ROOT = Path("results")

HIST_DIR = RESULTS_ROOT / "histograms"

HIST_DIR.mkdir(parents=True, exist_ok=True)


BOOT_PATH = RESULTS_ROOT / "bootstrap_runs" / "nl300_m50_w50" / "merged_bootstrap_nl300_m50_w50.npz"


data = np.load(BOOT_PATH, allow_pickle=True)


boot = np.asarray(data["pooled_boot_logZ"], dtype=float)

boot = boot[np.isfinite(boot)]


ns = np.asarray(data["ns_logZs"], dtype=float)

ns = ns[np.isfinite(ns)]


plt.figure(figsize=(8,5))


plt.hist(ns, bins=30, density=True, alpha=0.5, label="NS runs")

plt.hist(boot, bins=30, density=True, alpha=0.5, label="Bootstrap")


plt.xlabel("logZ")

plt.ylabel("Density")

plt.title("Bootstrap vs own NS runs")

plt.legend()


out = HIST_DIR / "hist_boot_vs_own_ns.png"

plt.savefig(out, dpi=200)

print("Saved:", out)
