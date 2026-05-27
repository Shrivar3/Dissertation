from __future__ import annotations

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
HIST_DIR = RESULTS_ROOT / "histograms"
HIST_DIR.mkdir(parents=True, exist_ok=True)

MULTI_RUN_REF_PATH = RESULTS_ROOT / "nl300_m50_w50" / "combined_nl300_m50_w50.npz"
BOOT_SUMMARY_PATH = RESULTS_ROOT / "bootstrap_runs" / "nl300_m50_w50" / "merged_bootstrap_nl300_m50_w50.npz"

HIST_BINS = 30

ref = np.load(MULTI_RUN_REF_PATH, allow_pickle=True)
logZs_multi = np.asarray(ref["logZs"], dtype=float)
logZs_multi = logZs_multi[np.isfinite(logZs_multi)]

summary = np.load(BOOT_SUMMARY_PATH, allow_pickle=True)
tag = str(summary["tag"][0])
pooled_boot = np.asarray(summary["pooled_boot_logZ"], dtype=float)
pooled_boot = pooled_boot[np.isfinite(pooled_boot)]

plt.figure(figsize=(9.5, 4.8))
plt.hist(logZs_multi, bins=HIST_BINS, density=True, alpha=0.45, label="Multi-run NS (ref)")
plt.hist(pooled_boot, bins=HIST_BINS, density=True, alpha=0.45, label="Method (3) pooled")

plt.axvline(logZs_multi.mean(), linestyle="--", linewidth=2, label="Ref mean")
plt.axvline(pooled_boot.mean(), linestyle=":", linewidth=2, label="Bootstrap mean")

plt.title(f"Overlay histograms: logZ distributions\n(tag={tag})")
plt.xlabel("logZ")
plt.ylabel("Density")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

out_path = HIST_DIR / f"hist_boot_vs_ref_{tag}.png"
plt.savefig(out_path, dpi=220)
print(f"Saved histogram to: {out_path}")
