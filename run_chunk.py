from pathlib import Path

import sys

import time

from datetime import timedelta

import os


REPO_ROOT = Path(__file__).resolve().parent

sys.path.append(str(REPO_ROOT / "src"))


from multi_run_driver import run_multi_ns_and_save


# -----------------------------

# SETTINGS FROM SLURM / ENV

# -----------------------------

CHUNK_ID = int(os.environ.get("CHUNK_ID", "0"))

RUNS_PER_CHUNK = int(os.environ.get("RUNS_PER_CHUNK", "50"))


# Seed spacing so chunks do not overlap

BASE_SEED0 = 415

BASE_SEED = BASE_SEED0 + CHUNK_ID * RUNS_PER_CHUNK


# NS settings

N_LIVE = 300

NS_MCMC_STEPS = 50

MH_WARMUP_STEPS = 50

N_ITER_MAX = 500_000


RESULTS_DIR = REPO_ROOT / "results" / "large_runs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


OUT_NAME = (

    f"chunk_{CHUNK_ID:02d}"

    f"_runs{RUNS_PER_CHUNK}"

    f"_seed{BASE_SEED}"

    f"_nl{N_LIVE}"

    f"_m{NS_MCMC_STEPS}"

    f"_w{MH_WARMUP_STEPS}.npz"

)


start = time.time()


save_path = run_multi_ns_and_save(

    out_dir=RESULTS_DIR,

    out_name=OUT_NAME,

    n_runs=RUNS_PER_CHUNK,

    base_seed=BASE_SEED,

    regenerate_data_each_run=False,

    n_live=N_LIVE,

    ns_mcmc_steps=NS_MCMC_STEPS,

    mh_warmup_steps=MH_WARMUP_STEPS,

    n_iter_max=N_ITER_MAX,

    verbose=False,

)


end = time.time()

runtime = timedelta(seconds=int(end - start))


print(f"Chunk ID: {CHUNK_ID}")

print(f"Base seed: {BASE_SEED}")

print(f"Saved to: {save_path}")

print(f"Runtime: {runtime}")
