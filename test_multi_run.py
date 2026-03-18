from pathlib import Path
import sys
import time
from datetime import timedelta

# =====================================================
# PATH SETUP
# =====================================================

REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT / "src"))

from multi_run_driver import run_multi_ns_and_save

RESULTS_DIR = REPO_ROOT / "results" / "test_runs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# SETTINGS
# =====================================================

N_RUNS = 10
N_LIVE = 100
NS_MCMC_STEPS = 20
MH_WARMUP_STEPS = 20

# =====================================================
# RUN
# =====================================================

start = time.time()

save_path = run_multi_ns_and_save(
    out_dir=RESULTS_DIR,
    out_name=(
        f"TEST_ns_multi_runs_{N_RUNS}"
        f"_ds{415}"
        f"_nl{N_LIVE}"
        f"_m{NS_MCMC_STEPS}"
        f"_w{MH_WARMUP_STEPS}.npz"
    ),
    n_runs=N_RUNS,
    base_seed=415,
    regenerate_data_each_run=False,
    n_live=N_LIVE,
    ns_mcmc_steps=NS_MCMC_STEPS,
    mh_warmup_steps=MH_WARMUP_STEPS,
    n_iter_max=50_000,
    verbose=True,
)

end = time.time()

runtime = timedelta(seconds=int(end - start))

print("Saved to:", save_path)
print("Total runtime:", runtime)
