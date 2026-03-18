from pathlib import Path

import sys

import numpy as np


if len(sys.argv) != 3:

    raise SystemExit("Usage: python merge_chunks.py <results_dir> <output_name.npz>")


RESULTS_DIR = Path(sys.argv[1])

OUT_PATH = RESULTS_DIR / sys.argv[2]


files = sorted(RESULTS_DIR.glob("chunk_*.npz"))

if not files:

    raise FileNotFoundError(f"No chunk files found in {RESULTS_DIR}")


print("Found chunk files:")

for f in files:

    print(" ", f.name)


loaded = [np.load(f, allow_pickle=True) for f in files]


concat_keys = [

    "run_seeds", "logZs", "Hs", "mean_accs", "n_dead",

    "dead_logLs", "trace_logZ", "step_sizes"

]


first = loaded[0]

out = {}


for key in first.files:

    if key in concat_keys:

        out[key] = np.concatenate([d[key] for d in loaded], axis=0)

    else:

        out[key] = first[key]


out["n_runs"] = np.array(len(out["logZs"]))


np.savez(OUT_PATH, **out)


print(f"\nSaved merged file to: {OUT_PATH}")

print(f"Total merged runs: {len(out['logZs'])}")

print(f"logZ mean: {np.mean(out['logZs']):.6f}")

print(f"logZ sd:   {np.std(out['logZs'], ddof=1):.6f}")
