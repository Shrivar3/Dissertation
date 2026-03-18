from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt



# =====================================================

# CONFIG

# =====================================================


NPZ_PATH = Path("results/test_runs/TEST_ns_multi_runs_5_ds415_nl100_m20_w20.npz")

OUT_PNG = NPZ_PATH.with_name("TEST_ns_multi_runs_5_ds415_nl100_m20_w20_hist.png")



# =====================================================

# HELPERS

# =====================================================


def safe_summary(arr):

    """Return a readable summary without crashing on object arrays."""

    try:

        return f"shape={arr.shape}, dtype={arr.dtype}"

    except Exception as e:

        return f"(could not summarise: {e})"



def try_extract_numeric_1d(arr):

    """

    Try to turn an array-like object into a 1D numeric numpy array.

    Returns None if that is not possible.

    """

    try:

        x = np.asarray(arr)


        # Already numeric

        if np.issubdtype(x.dtype, np.number):

            return x.ravel()


        # Object array: try elementwise float conversion

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

    """

    If arr is an object array of dict-like objects, try to extract logZ-ish values.

    """

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


# First try likely direct logZ keys

preferred_keys = ["logZs", "logZ", "logz", "est_logZ", "log_evidence"]


for key in preferred_keys:

    if key in data.files:

        candidate = try_extract_numeric_1d(data[key])

        if candidate is not None and candidate.size > 0:

            x = candidate

            chosen_key = key

            break


# If that failed, search all keys for a numeric 1D array of length > 1

if x is None:

    for key in data.files:

        candidate = try_extract_numeric_1d(data[key])

        if candidate is not None and candidate.size > 1:

            x = candidate

            chosen_key = key

            break


# If that failed, try extracting logZ values from object arrays of dicts

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

        "Please share the printed keys and I’ll help you target the exact one."

    )


# Remove NaN / inf just in case

x = np.asarray(x, dtype=float)

x = x[np.isfinite(x)]


if x.size == 0:

    raise RuntimeError("Chosen array became empty after removing NaN/inf values.")



# =====================================================

# PRINT SUMMARY

# =====================================================


print(f"\nUsing key: {chosen_key}")

print("Values:")

print(x)


if x.size >= 2:

    mean = np.mean(x)

    sd = np.std(x, ddof=1)

else:

    mean = np.mean(x)

    sd = 0.0


print(f"\nCount: {x.size}")

print(f"Mean: {mean:.6f}")

print(f"SD:   {sd:.6f}")



# =====================================================

# HISTOGRAM

# =====================================================


plt.figure(figsize=(7, 5))

plt.hist(x, bins="auto", edgecolor="black")

plt.xlabel(chosen_key)

plt.ylabel("Frequency")

plt.title("Histogram from multi-run NPZ")

plt.tight_layout()

plt.savefig(OUT_PNG, dpi=200)


print(f"\nSaved histogram to: {OUT_PNG}")
