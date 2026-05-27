from __future__ import annotations


import argparse

import csv

import math

from collections import defaultdict

from pathlib import Path


import numpy as np


try:

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True

except Exception:

    HAS_MATPLOTLIB = False



def safe_float(x):

    try:

        y = float(x)

        return y if np.isfinite(y) else np.nan

    except Exception:

        return np.nan



def safe_int(x):

    try:

        return int(float(x))

    except Exception:

        return -1



def finite(xs):

    arr = np.asarray([safe_float(x) for x in xs], dtype=float)

    return arr[np.isfinite(arr)]



def quantile(xs, q):

    xs = finite(xs)

    return float(np.quantile(xs, q)) if xs.size else np.nan



def mean(xs):

    xs = finite(xs)

    return float(np.mean(xs)) if xs.size else np.nan



def median(xs):

    xs = finite(xs)

    return float(np.median(xs)) if xs.size else np.nan



def sd(xs):

    xs = finite(xs)

    return float(np.std(xs, ddof=1)) if xs.size > 1 else np.nan



def read_csv(path: Path):

    with open(path, "r", newline="", encoding="utf-8") as f:

        return list(csv.DictReader(f))



def write_csv(path: Path, rows):

    rows = list(rows)

    if not rows:

        print(f"[warning] no rows to write: {path}")

        return


    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:

        w = csv.DictWriter(f, fieldnames=fieldnames)

        w.writeheader()

        w.writerows(rows)



def aggregate_to_observed_runs(rows):

    """

    The existing file is per bootstrap-output file. Usually this is one file per

    observed NS run, but some labels can have repeated files for the same ns_seed.

    So we group by label, ns_seed and tag_low.

    """

    groups = defaultdict(list)


    for row in rows:

        key = (

            row.get("label", ""),

            row.get("ns_seed", ""),

            row.get("tag_low", ""),

        )

        groups[key].append(row)


    out = []


    for (label, ns_seed, tag_low), group in sorted(groups.items()):

        boot_sds = finite(g.get("bootstrap_sd", np.nan) for g in group)

        skill_sds = finite(g.get("skilling_sd_sqrt_H_over_nlive", np.nan) for g in group)

        emp_sds = finite(g.get("empirical_multirun_sd", np.nan) for g in group)

        Hs = finite(g.get("H", np.nan) for g in group)


        if boot_sds.size == 0 or skill_sds.size == 0:

            continue


        boot_sd = float(np.mean(boot_sds))

        skill_sd = float(np.mean(skill_sds))

        emp_sd = float(emp_sds[0]) if emp_sds.size else np.nan

        H = float(np.mean(Hs)) if Hs.size else np.nan


        ratio = boot_sd / skill_sd if skill_sd > 0 else np.nan

        diff = boot_sd - skill_sd

        rel_diff_pct = 100.0 * (ratio - 1.0) if np.isfinite(ratio) else np.nan


        out.append({

            "label": label,

            "ns_seed": ns_seed,

            "tag_low": tag_low,

            "n_boot_files_for_this_ns_run": len(group),

            "bootstrap_sd": boot_sd,

            "skilling_sd": skill_sd,

            "bootstrap_minus_skilling_sd": diff,

            "bootstrap_over_skilling": ratio,

            "bootstrap_over_skilling_minus_1_pct": rel_diff_pct,

            "abs_bootstrap_over_skilling_minus_1_pct": abs(rel_diff_pct) if np.isfinite(rel_diff_pct) else np.nan,

            "H": H,

            "empirical_multirun_sd": emp_sd,

            "bootstrap_over_empirical": boot_sd / emp_sd if np.isfinite(emp_sd) and emp_sd > 0 else np.nan,

            "skilling_over_empirical": skill_sd / emp_sd if np.isfinite(emp_sd) and emp_sd > 0 else np.nan,

        })


    return out



def make_label_summary(per_run_rows):

    groups = defaultdict(list)

    for row in per_run_rows:

        groups[row["label"]].append(row)


    summaries = []


    for label, group in sorted(groups.items()):

        ratios = finite(g["bootstrap_over_skilling"] for g in group)

        abs_pct = finite(g["abs_bootstrap_over_skilling_minus_1_pct"] for g in group)

        boot = finite(g["bootstrap_sd"] for g in group)

        skill = finite(g["skilling_sd"] for g in group)


        if ratios.size == 0:

            continue


        corr = np.nan

        if boot.size > 1 and skill.size > 1 and np.std(boot) > 0 and np.std(skill) > 0:

            corr = float(np.corrcoef(boot, skill)[0, 1])


        summaries.append({

            "label": label,

            "n_observed_runs": len(group),

            "mean_ratio_boot_over_skilling": mean(ratios),

            "median_ratio_boot_over_skilling": median(ratios),

            "sd_ratio_boot_over_skilling": sd(ratios),

            "min_ratio": float(np.min(ratios)),

            "q05_ratio": quantile(ratios, 0.05),

            "q25_ratio": quantile(ratios, 0.25),

            "q75_ratio": quantile(ratios, 0.75),

            "q95_ratio": quantile(ratios, 0.95),

            "max_ratio": float(np.max(ratios)),

            "mean_abs_pct_difference": mean(abs_pct),

            "median_abs_pct_difference": median(abs_pct),

            "frac_abs_difference_gt_5pct": float(np.mean(abs_pct > 5.0)),

            "frac_abs_difference_gt_10pct": float(np.mean(abs_pct > 10.0)),

            "frac_boot_gt_skilling_by_10pct": float(np.mean(ratios > 1.10)),

            "frac_boot_lt_skilling_by_10pct": float(np.mean(ratios < 0.90)),

            "corr_bootstrap_sd_with_skilling_sd": corr,

            "mean_bootstrap_sd": mean(boot),

            "mean_skilling_sd": mean(skill),

        })


    return summaries



def make_plots(per_run_rows, out_dir: Path):

    if not HAS_MATPLOTLIB:

        print("[warning] matplotlib not available, skipping plots")

        return


    plot_dir = out_dir / "plots"

    plot_dir.mkdir(parents=True, exist_ok=True)


    groups = defaultdict(list)

    for row in per_run_rows:

        groups[row["label"]].append(row)


    for label, group in sorted(groups.items()):

        ratios = finite(g["bootstrap_over_skilling"] for g in group)

        boot = finite(g["bootstrap_sd"] for g in group)

        skill = finite(g["skilling_sd"] for g in group)


        if ratios.size == 0:

            continue


        # Histogram of per-run ratios

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.hist(ratios, bins=25)

        ax.axvline(1.0, linestyle="--", linewidth=1.5)

        ax.set_xlabel("bootstrap SD / Skilling SD")

        ax.set_ylabel("Number of observed NS runs")

        ax.set_title(f"Per-run bootstrap/Skilling SD ratios: {label}")

        fig.tight_layout()

        fig.savefig(plot_dir / f"{label}_ratio_histogram.png", dpi=200)

        plt.close(fig)


        # Scatter of bootstrap SD against Skilling SD

        if boot.size and skill.size:

            fig, ax = plt.subplots(figsize=(6, 6))

            ax.scatter(skill, boot, s=18)

            lo = min(np.min(skill), np.min(boot))

            hi = max(np.max(skill), np.max(boot))

            pad = 0.05 * (hi - lo) if hi > lo else 0.01

            lo -= pad

            hi += pad

            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)

            ax.set_xlim(lo, hi)

            ax.set_ylim(lo, hi)

            ax.set_aspect("equal", adjustable="box")

            ax.set_xlabel("Skilling SD")

            ax.set_ylabel("Bootstrap SD")

            ax.set_title(f"Bootstrap SD vs Skilling SD: {label}")

            fig.tight_layout()

            fig.savefig(plot_dir / f"{label}_bootstrap_vs_skilling_scatter.png", dpi=200)

            plt.close(fig)


        # Ratio by run index

        order = sorted(group, key=lambda r: safe_int(r["ns_seed"]))

        xs = np.arange(len(order))

        ys = finite(r["bootstrap_over_skilling"] for r in order)


        if ys.size:

            fig, ax = plt.subplots(figsize=(9, 4.8))

            ax.scatter(xs, ys, s=18)

            ax.axhline(1.0, linestyle="--", linewidth=1.5)

            ax.axhline(1.10, linestyle=":", linewidth=1.2)

            ax.axhline(0.90, linestyle=":", linewidth=1.2)

            ax.set_xlabel("Observed NS run index")

            ax.set_ylabel("bootstrap SD / Skilling SD")

            ax.set_title(f"Run-to-run variation in SD ratio: {label}")

            fig.tight_layout()

            fig.savefig(plot_dir / f"{label}_ratio_by_run.png", dpi=200)

            plt.close(fig)



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--input",

        type=Path,

        default=Path("results/bootstrap_vs_skilling_all/per_boot_file_bootstrap_vs_skilling.csv"),

    )

    parser.add_argument(

        "--out-dir",

        type=Path,

        default=Path("results/bootstrap_vs_skilling_individual_runs"),

    )

    args = parser.parse_args()


    args.out_dir.mkdir(parents=True, exist_ok=True)


    rows = read_csv(args.input)

    per_run = aggregate_to_observed_runs(rows)

    summary = make_label_summary(per_run)


    top_outliers = sorted(

        per_run,

        key=lambda r: safe_float(r["abs_bootstrap_over_skilling_minus_1_pct"]),

        reverse=True,

    )[:30]


    per_run_path = args.out_dir / "per_observed_run_bootstrap_vs_skilling.csv"

    summary_path = args.out_dir / "per_label_ratio_dispersion.csv"

    outlier_path = args.out_dir / "largest_individual_run_disagreements.csv"


    write_csv(per_run_path, per_run)

    write_csv(summary_path, summary)

    write_csv(outlier_path, top_outliers)


    make_plots(per_run, args.out_dir)


    print("\nSaved:")

    print(f"  {per_run_path}")

    print(f"  {summary_path}")

    print(f"  {outlier_path}")

    if HAS_MATPLOTLIB:

        print(f"  {args.out_dir / 'plots'}")


    print("\nPer-label ratio dispersion:")

    for row in summary:

        print(row)



if __name__ == "__main__":

    main()
