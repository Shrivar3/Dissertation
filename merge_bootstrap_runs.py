from pathlib import Path

import sys

import numpy as np


if len(sys.argv) != 3:

    raise SystemExit(

        "Usage: python merge_bootstrap_runs.py <bootstrap_family_dir> <output_name.npz>"

    )


BOOT_FAMILY_DIR = Path(sys.argv[1])

OUT_PATH = BOOT_FAMILY_DIR / sys.argv[2]


BOOT_DIR = BOOT_FAMILY_DIR / "boot_method3_out"

NS_DIR = BOOT_FAMILY_DIR / "ns_runs_out"


boot_files = sorted(BOOT_DIR.glob("boot3_*.npz"))

ns_files = sorted(NS_DIR.glob("ns_out_*.npz"))


if not boot_files:

    raise FileNotFoundError(f"No bootstrap files found in {BOOT_DIR}")


print("Found bootstrap files:")

for f in boot_files:

    print(" ", f.name)


pooled_logZ = []

boot_seeds = []

ns_seeds = []

tags = set()


for f in boot_files:

    d = np.load(f, allow_pickle=True)


    z = np.asarray(d["logZ"], dtype=float).reshape(-1)

    z = z[np.isfinite(z)]

    pooled_logZ.append(z)


    if "boot_seed" in d:

        boot_seeds.append(int(d["boot_seed"]))

    if "ns_seed" in d:

        ns_seeds.append(int(d["ns_seed"]))

    if "tag" in d:

        tags.add(str(d["tag"][0]))


pooled_logZ = np.concatenate(pooled_logZ) if pooled_logZ else np.array([], dtype=float)


ns_logZs = []

for f in ns_files:

    d = np.load(f, allow_pickle=True)

    if "logZ" in d:

        ns_logZs.append(float(d["logZ"]))

ns_logZs = np.asarray(ns_logZs, dtype=float)


tag_value = sorted(tags)[0] if tags else "unknown"


np.savez_compressed(

    OUT_PATH,

    tag=np.array([tag_value], dtype=object),

    pooled_boot_logZ=pooled_logZ,

    ns_logZs=ns_logZs,

    ns_seeds=np.asarray(ns_seeds, dtype=np.int64),

    boot_seeds=np.asarray(boot_seeds, dtype=np.int64),

    n_boot_files=len(boot_files),

    n_ns_files=len(ns_files),

    boot_files=np.array([str(f) for f in boot_files], dtype=object),

    ns_files=np.array([str(f) for f in ns_files], dtype=object),

)


print(f"\nSaved merged bootstrap summary to: {OUT_PATH}")

print(f"Total pooled bootstrap draws: {pooled_logZ.size}")

if pooled_logZ.size > 1:

    print(f"Bootstrap mean: {pooled_logZ.mean():.6f}")

    print(f"Bootstrap sd:   {pooled_logZ.std(ddof=1):.6f}")

if ns_logZs.size > 1:

    print(f"NS mean:        {ns_logZs.mean():.6f}")

    print(f"NS sd:          {ns_logZs.std(ddof=1):.6f}")
