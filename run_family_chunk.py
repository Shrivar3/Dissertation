from pathlib import Path

import os

import time

from datetime import timedelta


from multi_run_driver import run_multi_ns_and_save


REPO_ROOT = Path(__file__).resolve().parent


# -------- experiment parameters from environment --------

CHUNK_ID = int(os.environ["CHUNK_ID"])

RUNS_PER_CHUNK = int(os.environ["RUNS_PER_CHUNK"])


N_LIVE = int(os.environ["N_LIVE"])

NS_MCMC_STEPS = int(os.environ["NS_MCMC_STEPS"])

MH_WARMUP_STEPS = int(os.environ["MH_WARMUP_STEPS"])


DATA_SEED = int(os.environ.get("DATA_SEED", "415"))

BASE_SEED0 = int(os.environ.get("BASE_SEED0", "415"))

N_ITER_MAX = int(os.environ.get("N_ITER_MAX", "500000"))


LABEL = os.environ.get(

    "LABEL",

    f"nl{N_LIVE}_m{NS_MCMC_STEPS}_w{MH_WARMUP_STEPS}"

)


# -------- seed spacing across chunks --------

BASE_SEED = BASE_SEED0 + CHUNK_ID * RUNS_PER_CHUNK


# -------- output folder --------

RESULTS_DIR = REPO_ROOT / "results" / LABEL

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


OUT_NAME = (

    f"chunk_{CHUNK_ID:02d}"

    f"_runs{RUNS_PER_CHUNK}"

    f"_seed{BASE_SEED}"

    f"_ds{DATA_SEED}"

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

    data_seed=DATA_SEED,

    n_live=N_LIVE,

    ns_mcmc_steps=NS_MCMC_STEPS,

    mh_warmup_steps=MH_WARMUP_STEPS,

    n_iter_max=N_ITER_MAX,

    verbose=False,

)


end = time.time()

runtime = timedelta(seconds=int(end - start))


print(f"Label: {LABEL}")

print(f"Chunk ID: {CHUNK_ID}")

print(f"Base seed: {BASE_SEED}")

print(f"Data seed: {DATA_SEED}")

print(f"Saved to: {save_path}")

print(f"Runtime: {runtime}")
