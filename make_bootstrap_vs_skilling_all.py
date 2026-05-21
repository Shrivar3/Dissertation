from __future__ import annotations


import argparse

import csv

import re

from collections import defaultdict

from pathlib import Path


import numpy as np



def finite_1d(x):

    x = np.asarray(x, dtype=float).ravel()

    return x[np.isfinite(x)]



def scalar_from_npz(d, key, default=np.nan):

    if key not in d.files:

        return default

    arr = np.asarray(d[key])

    if arr.size == 0:

        return default

    return arr.item() if arr.shape == () or arr.size == 1 else arr



def str_from_npz(d, key, default=""):

    if key not in d.files:

        return default

    arr = np.asarray(d[key], dtype=object).ravel()

    if arr.size == 0:

        return default

    return str(arr[0])



def safe_float(x):

    try:

        return float(x)

    except Exception:

        return np.nan



def mean_finite(xs):

    xs = np.asarray([safe_float(x) for x in xs], dtype=float)

    xs = xs[np.isfinite(xs)]

    return float(np.mean(xs)) if xs.size else np.nan



def median_finite(xs):

    xs = np.asarray([safe_float(x) for x in xs], dtype=float)

    xs = xs[np.isfinite(xs)]

    return float(np.median(xs)) if xs.size else np.nan



def first_value(xs, default=""):

    xs = list(xs)

    return xs[0] if xs else default



def unique_count(xs):

    return len(set(xs))



def infer_n_live(label, d=None):

    if d is not None and "n_live" in d.files:

        val = scalar_from_npz(d, "n_live", default=np.nan)

        if np.isfinite(val):

            return int(val)


    m = re.search(r"nl(\d+)", label)

    if m:

        return int(m.group(1))


    return np.nan



def discover_experiment_roots(bootstrap_root: Path):

    roots = []


    if not bootstrap_root.exists():

        raise RuntimeError(f"Could not find bootstrap root: {bootstrap_root}")


    for p in sorted(bootstrap_root.iterdir()):

        if not p.is_dir():

            continue


        if (p / "boot_method3_out").is_dir() and (p / "ns_runs_out").is_dir():

            roots.append(p)


    return roots



def build_ns_index(ns_dir: Path):

    by_seed = {}

    by_seed_and_tag_low = {}


    for fp in sorted(ns_dir.glob("*.npz")):

        try:

            with np.load(fp, allow_pickle=True) as d:

                if "ns_seed" not in d.files:

                    continue


                ns_seed = int(scalar_from_npz(d, "ns_seed"))

                tag_low = str_from_npz(d, "tag_low", default="")


                n_live = scalar_from_npz(d, "n_live", default=np.nan)

                n_live = int(n_live) if np.isfinite(n_live) else np.nan


                entry = {

                    "ns_path": str(fp),

                    "ns_seed": ns_seed,

                    "tag_low": tag_low,

                    "H": float(scalar_from_npz(d, "H", default=np.nan)),

                    "ns_logZ": float(scalar_from_npz(d, "logZ", default=np.nan)),

                    "n_live": n_live,

                }


                by_seed_and_tag_low[(ns_seed, tag_low)] = entry

                by_seed.setdefault(ns_seed, entry)


        except Exception as e:

            print(f"[warning] Skipping NS file {fp}: {e}")


    return by_seed, by_seed_and_tag_low



def empirical_multirun_sd(results_root: Path, label: str, fallback_ns_logzs=None):

    combined = results_root / label / f"combined_{label}.npz"


    if combined.exists():

        try:

            with np.load(combined, allow_pickle=True) as d:

                if "logZs" in d.files:

                    z = finite_1d(d["logZs"])

                    if z.size > 1:

                        return float(np.std(z, ddof=1)), str(combined)

        except Exception:

            pass


    label_dir = results_root / label

    logzs = []


    if label_dir.exists():

        for fp in sorted(label_dir.glob("chunk*.npz")):

            try:

                with np.load(fp, allow_pickle=True) as d:

                    if "logZs" in d.files:

                        logzs.append(finite_1d(d["logZs"]))

            except Exception:

                pass


    if logzs:

        z = finite_1d(np.concatenate(logzs))

        if z.size > 1:

            return float(np.std(z, ddof=1)), f"{label_dir}/chunk*.npz"


    if fallback_ns_logzs is not None:

        z = finite_1d(fallback_ns_logzs)

        if z.size > 1:

            return float(np.std(z, ddof=1)), "fallback: ns_runs_out logZ values"


    return np.nan, ""



def write_csv(path: Path, rows):

    rows = list(rows)


    if not rows:

        print(f"[warning] No rows to write for {path}")

        return


    fieldnames = list(rows[0].keys())


    with open(path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        writer.writerows(rows)



def make_summary(rows, group_keys):

    groups = defaultdict(list)


    for row in rows:

        key = tuple(row[k] for k in group_keys)

        groups[key].append(row)


    summaries = []


    for key, group in sorted(groups.items()):

        out = {k: v for k, v in zip(group_keys, key)}


        out["n_boot_files"] = len(group)

        out["n_unique_ns_runs"] = unique_count(g["ns_seed"] for g in group)

        out["n_live"] = first_value(g["n_live"] for g in group)

        out["empirical_multirun_sd"] = first_value(g["empirical_multirun_sd"] for g in group)


        out["mean_bootstrap_sd"] = mean_finite(g["bootstrap_sd"] for g in group)

        out["median_bootstrap_sd"] = median_finite(g["bootstrap_sd"] for g in group)


        out["mean_skilling_sd"] = mean_finite(g["skilling_sd_sqrt_H_over_nlive"] for g in group)

        out["median_skilling_sd"] = median_finite(g["skilling_sd_sqrt_H_over_nlive"] for g in group)


        out["mean_boot_over_skilling"] = mean_finite(g["bootstrap_sd_over_skilling_sd"] for g in group)

        out["median_boot_over_skilling"] = median_finite(g["bootstrap_sd_over_skilling_sd"] for g in group)


        out["mean_boot_over_empirical"] = mean_finite(g["bootstrap_sd_over_empirical_multirun_sd"] for g in group)

        out["median_boot_over_empirical"] = median_finite(g["bootstrap_sd_over_empirical_multirun_sd"] for g in group)


        out["mean_skilling_over_empirical"] = mean_finite(g["skilling_sd_over_empirical_multirun_sd"] for g in group)

        out["empirical_multirun_source"] = first_value(g["empirical_multirun_source"] for g in group)


        summaries.append(out)


    return summaries



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--results-root", type=Path, default=Path("results"))

    parser.add_argument("--out-dir", type=Path, default=Path("results/bootstrap_vs_skilling_all"))

    parser.add_argument("--bootstrap-root", type=Path, default=None)


    args, _ = parser.parse_known_args()


    results_root = args.results_root

    bootstrap_root = args.bootstrap_root or (results_root / "bootstrap_runs")

    out_dir = args.out_dir

    out_dir.mkdir(parents=True, exist_ok=True)


    experiment_roots = discover_experiment_roots(bootstrap_root)


    if not experiment_roots:

        raise RuntimeError(f"No labelled experiment folders found under {bootstrap_root}")


    print(f"Found {len(experiment_roots)} experiment folders:")

    for r in experiment_roots:

        print(f"  - {r.name}")


    rows = []


    for root in experiment_roots:

        label = root.name

        boot_dir = root / "boot_method3_out"

        ns_dir = root / "ns_runs_out"


        ns_by_seed, ns_by_seed_and_tag_low = build_ns_index(ns_dir)

        boot_files = sorted(boot_dir.glob("*.npz"))


        print(f"\nProcessing {label}: {len(boot_files)} bootstrap files")


        for boot_fp in boot_files:

            try:

                with np.load(boot_fp, allow_pickle=True) as b:

                    if "logZ" not in b.files:

                        continue


                    z_boot = finite_1d(b["logZ"])

                    if z_boot.size <= 1:

                        continue


                    ns_seed = int(scalar_from_npz(b, "ns_seed", default=np.nan))

                    boot_seed = int(scalar_from_npz(b, "boot_seed", default=np.nan))

                    tag = str_from_npz(b, "tag", default="")

                    tag_base = str_from_npz(b, "tag_base", default="")

                    tag_low = str_from_npz(b, "tag_low", default="")


                    n_live = infer_n_live(label, b)


                ns_info = ns_by_seed_and_tag_low.get((ns_seed, tag_low))

                if ns_info is None:

                    ns_info = ns_by_seed.get(ns_seed, {})


                H = safe_float(ns_info.get("H", np.nan))

                ns_logZ = safe_float(ns_info.get("ns_logZ", np.nan))


                ns_n_live = ns_info.get("n_live", np.nan)

                if np.isfinite(safe_float(ns_n_live)):

                    n_live = int(ns_n_live)


                boot_sd = float(np.std(z_boot, ddof=1))


                if np.isfinite(H) and np.isfinite(n_live) and n_live > 0:

                    skilling_sd = float(np.sqrt(H / n_live))

                else:

                    skilling_sd = np.nan


                rows.append({

                    "label": label,

                    "tag": tag,

                    "tag_base": tag_base,

                    "tag_low": tag_low,

                    "ns_seed": ns_seed,

                    "boot_seed": boot_seed,

                    "n_boot": int(z_boot.size),

                    "n_live": n_live,

                    "ns_logZ": ns_logZ,

                    "H": H,

                    "bootstrap_sd": boot_sd,

                    "skilling_sd_sqrt_H_over_nlive": skilling_sd,

                    "bootstrap_sd_over_skilling_sd": (

                        boot_sd / skilling_sd

                        if np.isfinite(skilling_sd) and skilling_sd > 0

                        else np.nan

                    ),

                    "boot_file": str(boot_fp),

                    "ns_file": str(ns_info.get("ns_path", "")),

                })


            except Exception as e:

                print(f"[warning] Failed on bootstrap file {boot_fp}: {e}")


    if not rows:

        raise RuntimeError("No usable bootstrap outputs were found.")


    labels = sorted(set(row["label"] for row in rows))


    for label in labels:

        fallback = [

            row["ns_logZ"]

            for row in rows

            if row["label"] == label and np.isfinite(safe_float(row["ns_logZ"]))

        ]


        emp_sd, emp_source = empirical_multirun_sd(results_root, label, fallback_ns_logzs=fallback)


        for row in rows:

            if row["label"] != label:

                continue


            row["empirical_multirun_sd"] = emp_sd

            row["empirical_multirun_source"] = emp_source


            boot_sd = safe_float(row["bootstrap_sd"])

            skill_sd = safe_float(row["skilling_sd_sqrt_H_over_nlive"])


            row["bootstrap_sd_over_empirical_multirun_sd"] = (

                boot_sd / emp_sd if np.isfinite(emp_sd) and emp_sd > 0 else np.nan

            )


            row["skilling_sd_over_empirical_multirun_sd"] = (

                skill_sd / emp_sd if np.isfinite(emp_sd) and emp_sd > 0 else np.nan

            )


    summary_by_experiment = make_summary(rows, ["label", "tag_low"])

    summary_by_label = make_summary(rows, ["label"])


    per_run_path = out_dir / "per_boot_file_bootstrap_vs_skilling.csv"

    summary_exp_path = out_dir / "summary_by_experiment_bootstrap_vs_skilling.csv"

    summary_label_path = out_dir / "summary_by_label_bootstrap_vs_skilling.csv"

    txt_path = out_dir / "summary_bootstrap_vs_skilling.txt"


    write_csv(per_run_path, rows)

    write_csv(summary_exp_path, summary_by_experiment)

    write_csv(summary_label_path, summary_by_label)


    with open(txt_path, "w", encoding="utf-8") as f:

        f.write("Bootstrap SD vs Skilling heuristic summary\n")

        f.write("=" * 60 + "\n\n")

        f.write("Formulae:\n")

        f.write("  bootstrap_sd = sd of bootstrap logZ samples for one observed NS run\n")

        f.write("  skilling_sd = sqrt(H / n_live)\n\n")


        f.write("Summary by label:\n")

        for row in summary_by_label:

            f.write("\n")

            f.write(str(row))

            f.write("\n")


        f.write("\nSummary by label and tag_low:\n")

        for row in summary_by_experiment:

            f.write("\n")

            f.write(str(row))

            f.write("\n")


    print("\nSummary by label:")

    for row in summary_by_label:

        print(row)


    print("\nSaved:")

    print(f"  {per_run_path}")

    print(f"  {summary_exp_path}")

    print(f"  {summary_label_path}")

    print(f"  {txt_path}")



if __name__ == "__main__":

    main()
