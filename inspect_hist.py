from __future__ import annotations


from pathlib import Path

import sys


import numpy as np


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt



# =====================================================

# ARGUMENTS

# =====================================================


if len(sys.argv) < 2:

    raise SystemExit(

        "Usage:\n"

        "  python inspect_hist.py <input_npz> [output_png] [plot_title]"

    )


NPZ_PATH = Path(sys.argv[1]).expanduser().resolve()


if len(sys.argv) >= 3:

    OUT_PNG = Path(sys.argv[2]).expanduser().resolve()

else:

    OUT_PNG = NPZ_PATH.with_name(f"{NPZ_PATH.stem}_hist.png")


if len(sys.argv) >= 4:

    PLOT_TITLE = sys.argv[3]

else:

    PLOT_TITLE = f"Histogram: {NPZ_PATH.stem}"



# =====================================================

# HELPERS

# =====================================================


def safe_summary(arr):

    try:

        return f"shape={arr.shape}, dtype={arr.dtype}"

    except Exception as e:

        return f"(could not summarise: {e})"



def try_extract_numeric_1d(arr):

    try:

        x = np.asarray(arr)


        if np.issubdtype(x.dtype, np.number):

            return x.ravel()


        if x.dtype == object:

            vals = []

            for v in x.ravel():

                try:

                    vals.append(float(v))

                except Exception:

                    return None

            return np.asarray(vals, dtype=float)


    except Exception:

        return None


    return None



def extract_logz_from_object_array(arr):

    try:

        x = np.asarray(arr)

        if x.dtype != object:

            return None


        vals = []

        for item in x.ravel():

            if isinstance(item, dict):

                for key in ["logZ", "logz", "est_logZ", "log_evidence"]:

                    if key in item:

                        vals.append(float(item[key]))

                        break

                else:

                    return None

            else:

                return None


        if len(vals) > 0:

            return np.asarray(vals, dtype=float)


    except Exception:

        return None


    return None



# =====================================================

# LOAD FILE

# =====================================================


if not NPZ_PATH.exists():

    raise FileNotFoundError(f"Could not find file: {NPZ_PATH}")


data = np.load(NPZ_PATH, allow_pickle=True)


print(f"Loaded: {NPZ_PATH}")

print("\nKeys in file:")

for k in data.files:

    arr = data[k]

    print(f"  {k}: {safe_summary(arr)}")



# =====================================================

# CHOOSE WHAT TO PLOT

# =====================================================


x = None

chosen_key = None


preferred_keys = ["logZs", "pooled_boot_logZ", "logZ", "logz", "est_logZ", "log_evidence"]


for key in preferred_keys:

    if key in data.files:

        candidate = try_extract_numeric_1d(data[key])

        if candidate is not None and candidate.size > 0:

            x = candidate

            chosen_key = key

            break


if x is None:

    for key in data.files:

        candidate = try_extract_numeric_1d(data[key])

        if candidate is not None and candidate.size > 1:

            x = candidate

            chosen_key = key

            break


if x is None:

    for key in data.files:

        candidate = extract_logz_from_object_array(data[key])

        if candidate is not None and candidate.size > 1:

            x = candidate

            chosen_key = f"{key} -> extracted logZ"

            break


if x is None:

    raise RuntimeError(

        "Could not find a plottable numeric array in the NPZ file. "

        "Please inspect the printed keys."

    )


x = np.asarray(x, dtype=float)

x = x[np.isfinite(x)]


if x.size == 0:

    raise RuntimeError("Chosen array became empty after removing NaN/inf values.")



# =====================================================

# PRINT SUMMARY

# =====================================================


print(f"\nUsing key: {chosen_key}")


if x.size >= 2:

    mean = np.mean(x)

    sd = np.std(x, ddof=1)

else:

    mean = np.mean(x)

    sd = 0.0


print(f"Count: {x.size}")

print(f"Mean:  {mean:.6f}")

print(f"SD:    {sd:.6f}")

print(f"Min:   {np.min(x):.6f}")

print(f"Max:   {np.max(x):.6f}")



# =====================================================

# HISTOGRAM

# =====================================================


plt.figure(figsize=(7, 5))

plt.hist(x, bins="auto", edgecolor="black")

plt.axvline(mean, linestyle="--", linewidth=1.5, label=f"Mean = {mean:.4f}")

plt.xlabel(chosen_key)

plt.ylabel("Frequency")

plt.title(PLOT_TITLE)

plt.legend()

plt.tight_layout()

plt.savefig(OUT_PNG, dpi=200)


print(f"\nSaved histogram to: {OUT_PNG}")
