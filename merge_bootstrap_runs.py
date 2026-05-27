from __future__ import annotations


from pathlib import Path

import sys

import numpy as np



if len(sys.argv) != 3:

    raise SystemExit(

        "Usage: python merge_bootstrap_runs.py <bootstrap_family_dir> <output_name.npz>"

    )


BOOT_FAMILY_DIR = Path(sys.argv[1]).resolve()

OUT_PATH = BOOT_FAMILY_DIR / sys.argv[2]


BOOT_DIR = BOOT_FAMILY_DIR / "boot_method3_out"

NS_DIR = BOOT_FAMILY_DIR / "ns_runs_out"

CHUNK_SUMMARY_DIR = BOOT_FAMILY_DIR / "chunk_summaries"


boot_files = sorted(BOOT_DIR.glob("boot3_*.npz"))

ns_files = sorted(NS_DIR.glob("ns_out_*.npz"))

chunk_summary_files = sorted(CHUNK_SUMMARY_DIR.glob("bootstrap_summary_*.npz"))


if not BOOT_FAMILY_DIR.exists():

    raise FileNotFoundError(f"Bootstrap family directory not found: {BOOT_FAMILY_DIR}")


if not boot_files:

    raise FileNotFoundError(f"No bootstrap files found in {BOOT_DIR}")


print(f"Bootstrap family dir: {BOOT_FAMILY_DIR}")

print(f"Found {len(boot_files)} bootstrap files")

print(f"Found {len(ns_files)} NS files")

print(f"Found {len(chunk_summary_files)} chunk summary files")


pooled_logZ = []

boot_seeds = []

ns_seeds_from_boot = []

tags = set()

tag_bases = set()

tag_lows = set()

chunk_ids_from_boot = []


for f in boot_files:

    d = np.load(f, allow_pickle=True)


    z = np.asarray(d["logZ"], dtype=float).reshape(-1)

    z = z[np.isfinite(z)]

    pooled_logZ.append(z)


    if "boot_seed" in d:

        boot_seeds.append(int(d["boot_seed"]))

    if "ns_seed" in d:

        ns_seeds_from_boot.append(int(d["ns_seed"]))

    if "tag" in d:

        tags.add(str(d["tag"][0]))

    if "tag_base" in d:

        tag_bases.add(str(d["tag_base"][0]))

    if "tag_low" in d:

        tag_lows.add(str(d["tag_low"][0]))

    if "chunk_id" in d:

        chunk_ids_from_boot.append(int(d["chunk_id"]))


pooled_logZ = np.concatenate(pooled_logZ) if pooled_logZ else np.array([], dtype=float)


ns_logZs = []

ns_seeds_from_ns = []

chunk_ids_from_ns = []


for f in ns_files:

    d = np.load(f, allow_pickle=True)

    if "logZ" in d:

        ns_logZs.append(float(d["logZ"]))

    if "ns_seed" in d:

        ns_seeds_from_ns.append(int(d["ns_seed"]))

    if "chunk_id" in d:

        chunk_ids_from_ns.append(int(d["chunk_id"]))


ns_logZs = np.asarray(ns_logZs, dtype=float)

ns_seeds_from_ns = np.asarray(ns_seeds_from_ns, dtype=np.int64) if ns_seeds_from_ns else np.array([], dtype=np.int64)

boot_seeds = np.asarray(boot_seeds, dtype=np.int64) if boot_seeds else np.array([], dtype=np.int64)

ns_seeds_from_boot = np.asarray(ns_seeds_from_boot, dtype=np.int64) if ns_seeds_from_boot else np.array([], dtype=np.int64)


# ------------------------------------------------------------

# Chunk summary aggregation

# ------------------------------------------------------------


chunk_ids = []

chunk_run_starts = []

chunk_run_stops = []

chunk_ns_seed_arrays = []

chunk_boot_seed_arrays = []

chunk_pooled_sizes = []


for f in chunk_summary_files:

    d = np.load(f, allow_pickle=True)


    if "chunk_id" in d:

        chunk_ids.append(int(d["chunk_id"]))

    if "run_start" in d:

        chunk_run_starts.append(int(d["run_start"]))

    if "run_stop" in d:

        chunk_run_stops.append(int(d["run_stop"]))

    if "ns_seeds" in d:

        chunk_ns_seed_arrays.append(np.asarray(d["ns_seeds"], dtype=np.int64))

    if "boot_seeds" in d:

        chunk_boot_seed_arrays.append(np.asarray(d["boot_seeds"], dtype=np.int64))

    if "pooled_boot_logZ" in d:

        chunk_pooled_sizes.append(int(np.asarray(d["pooled_boot_logZ"]).size))


chunk_ids = np.asarray(chunk_ids, dtype=np.int64) if chunk_ids else np.array([], dtype=np.int64)

chunk_run_starts = np.asarray(chunk_run_starts, dtype=np.int64) if chunk_run_starts else np.array([], dtype=np.int64)

chunk_run_stops = np.asarray(chunk_run_stops, dtype=np.int64) if chunk_run_stops else np.array([], dtype=np.int64)

chunk_pooled_sizes = np.asarray(chunk_pooled_sizes, dtype=np.int64) if chunk_pooled_sizes else np.array([], dtype=np.int64)


# ------------------------------------------------------------

# Consistency checks

# ------------------------------------------------------------


def find_duplicates(arr: np.ndarray) -> np.ndarray:

    if arr.size == 0:

        return np.array([], dtype=arr.dtype)

    vals, counts = np.unique(arr, return_counts=True)

    return vals[counts > 1]


dup_boot_seeds = find_duplicates(boot_seeds)

dup_ns_seeds = find_duplicates(ns_seeds_from_ns)


if len(tags) > 1:

    print("\n[warning] Multiple tag values found:")

    for t in sorted(tags):

        print(" ", t)


if len(tag_bases) > 1:

    print("\n[warning] Multiple tag_base values found:")

    for t in sorted(tag_bases):

        print(" ", t)


if len(tag_lows) > 1:

    print("\n[warning] Multiple tag_low values found:")

    for t in sorted(tag_lows):

        print(" ", t)


if dup_boot_seeds.size > 0:

    print("\n[warning] Duplicate boot seeds found:")

    print(dup_boot_seeds)


if dup_ns_seeds.size > 0:

    print("\n[warning] Duplicate NS seeds found:")

    print(dup_ns_seeds)


tag_value = sorted(tags)[0] if tags else "unknown"

tag_base_value = sorted(tag_bases)[0] if tag_bases else "unknown"

tag_low_value = sorted(tag_lows)[0] if tag_lows else "unknown"


# ------------------------------------------------------------

# Save merged file

# ------------------------------------------------------------


np.savez_compressed(

    OUT_PATH,

    tag=np.array([tag_value], dtype=object),

    tag_base=np.array([tag_base_value], dtype=object),

    tag_low=np.array([tag_low_value], dtype=object),

    pooled_boot_logZ=pooled_logZ,

    ns_logZs=ns_logZs,

    ns_seeds=np.sort(ns_seeds_from_ns),

    ns_seeds_from_boot=np.sort(ns_seeds_from_boot),

    boot_seeds=np.sort(boot_seeds),

    duplicate_ns_seeds=dup_ns_seeds,

    duplicate_boot_seeds=dup_boot_seeds,

    chunk_ids=np.sort(np.unique(chunk_ids)) if chunk_ids.size > 0 else np.array([], dtype=np.int64),

    chunk_run_starts=chunk_run_starts,

    chunk_run_stops=chunk_run_stops,

    chunk_pooled_sizes=chunk_pooled_sizes,

    n_boot_files=len(boot_files),

    n_ns_files=len(ns_files),

    n_chunk_summary_files=len(chunk_summary_files),

    boot_files=np.array([str(f) for f in boot_files], dtype=object),

    ns_files=np.array([str(f) for f in ns_files], dtype=object),

    chunk_summary_files=np.array([str(f) for f in chunk_summary_files], dtype=object),

)


# ------------------------------------------------------------

# Print summary

# ------------------------------------------------------------


print(f"\nSaved merged bootstrap summary to: {OUT_PATH}")

print(f"Total pooled bootstrap draws: {pooled_logZ.size}")


if pooled_logZ.size > 0:

    mean_boot = float(np.mean(pooled_logZ))

    sd_boot = float(np.std(pooled_logZ, ddof=1)) if pooled_logZ.size > 1 else np.nan

    print(f"Bootstrap mean: {mean_boot:.6f}")

    print(f"Bootstrap sd:   {sd_boot:.6f}")


if ns_logZs.size > 0:

    mean_ns = float(np.mean(ns_logZs))

    sd_ns = float(np.std(ns_logZs, ddof=1)) if ns_logZs.size > 1 else np.nan

    print(f"NS mean:        {mean_ns:.6f}")

    print(f"NS sd:          {sd_ns:.6f}")


if dup_boot_seeds.size == 0 and dup_ns_seeds.size == 0:

    print("Seed check:     no duplicates detected")

else:

    print("Seed check:     duplicates detected; inspect merged file")
