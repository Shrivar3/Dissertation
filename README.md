# Uncertainty Quantification in Nested Sampling

Code repository for the Master's dissertation:

**Uncertainty in Nested Sampling**

MMORSE / Integrated Master's in Mathematics, Operational Research, Statistics and Economics  
Department of Statistics, University of Warwick  
2025–26

Author: **Shrivar Singh**  
Supervisors: **Prof. Christian Robert** and **Dr. Nicholas Tawn**

---

## Overview

This repository contains the code used for the dissertation experiments on uncertainty quantification for Nested Sampling evidence estimates.

The project studies whether the uncertainty of a Nested Sampling log-evidence estimate can be approximated from a **single completed MCMC-based Nested Sampling run**, rather than requiring many expensive independent reruns.

The main contribution implemented here is a **single-run bootstrap** for Nested Sampling. The bootstrap uses:

- the observed dead-point log-likelihood ladder;
- retained phantom-point log-likelihood values from constrained MCMC replacement steps;
- a lowest-point anchor pool;
- anchored two-sweep ladder reconstruction;
- independent stochastic shrinkage resimulation.

The resulting bootstrap replicates approximate the run-to-run uncertainty of the Nested Sampling log-evidence estimator, conditional on the information contained in the observed run.

---

## Main implementation path

The dissertation-matching implementation is contained in the following files.

```text
src/ns_mh_phantom.py
```

Nested Sampling implementation with constrained random-walk Metropolis--Hastings replacement and phantom-point storage.

Important behaviour:

- warmup adapts the proposal scale;
- production restarts from the original live-point seed after warmup;
- only production-phase MCMC states are stored as phantom points by default.

```text
src/bootstrap_method3.py
```

Dissertation-matching Method 3 bootstrap implementation.

This implements:

- candidate-pool construction from dead and phantom log-likelihood values;
- anchor drawing from a lowest-point anchor pool;
- anchored empirical two-sweep ladder updates;
- static Nested Sampling shrinkage resimulation;
- bootstrap log-evidence computation.

```text
run_bootstrap_method3.py
```

Main end-to-end runner.

This script:

1. simulates or rebuilds the logistic-regression data;
2. runs the observed MCMC-based Nested Sampling calculation;
3. constructs the lowest-point anchor pool;
4. runs the anchored Method 3 bootstrap;
5. saves the observed NS output, bootstrap output, and chunk summary.

---

## Dissertation-matching bootstrap settings

The canonical bootstrap settings are:

```text
H = 100
```

where `H` is the size of the lowest-point anchor pool.

Each anchor is generated as:

```text
minimum log-likelihood among n_live prior draws
```

For each bootstrap replicate:

```text
S = 10
```

anchored two-sweep ladder updates are applied using one fixed anchor for that replicate.

The bootstrap uses:

```text
N_SHRINK = 1
```

so each reconstructed ladder is paired with one independently simulated static shrinkage path.

The main saved-output tags should include strings of the form:

```text
anchorH100_min1ofnlive...
```

This indicates that the run used the dissertation-matching anchor construction.

---

## Repository structure

```text
.
├── README.md
├── pyproject.toml
├── enviroment.yml
├── run_bootstrap_method3.py
├── run_bootstrap.slurm
├── run_array.slurm
├── run_family.slurm
├── merge_bootstrap_runs.py
├── merge_chunks.py
├── make_bootstrap_diagnostics.py
├── make_bootstrap_sd_plot.py
├── make_bootstrap_vs_skilling_all.py
├── make_individual_bootstrap_vs_skilling.py
├── make_qq_plot_boot_vs_ref.py
├── report_diagnostics_clean.py
├── results/
└── src/
    ├── __init__.py
    ├── ns_mh_phantom.py
    ├── bootstrap_method3.py
    ├── bootstrap_common.py
    ├── bootstrap_method1.py
    ├── bootsrap_method2.py
    ├── diagnostics.py
    ├── multi_run_driver.py
    └── compute_reference_logz_ep.py
```

The three most important files for reproducing the submitted bootstrap method are:

```text
src/ns_mh_phantom.py
src/bootstrap_method3.py
run_bootstrap_method3.py
```

Older or auxiliary scripts are kept for diagnostics, plotting, comparisons, and earlier experiments.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Shrivar3/Dissertation.git
cd Dissertation
git checkout multi-run-test
```

Create the conda environment:

```bash
conda env create -f enviroment.yml
conda activate nsdissert
```

The environment file is named:

```text
enviroment.yml
```

rather than `environment.yml`.

The package is installed in editable mode by the environment file. If needed, reinstall manually using:

```bash
pip install -e .
```

---

## Quick syntax check

After editing the implementation files, check that the main files compile:

```bash
python -m py_compile src/ns_mh_phantom.py src/bootstrap_method3.py run_bootstrap_method3.py
```

---

## Running the dissertation bootstrap pipeline

A small smoke test can be run with:

```bash
BOOT_RUNS=1 CHUNK_SIZE=1 N_ITER_MAX=2000 VERBOSE=True python run_bootstrap_method3.py
```

This is mainly for checking that the code runs, not for producing final dissertation-quality results.

A typical dissertation-style run is:

```bash
LABEL=nl100_m20_w20 BOOT_RUNS=35 CHUNK_SIZE=35 N_LIVE=100 NS_MCMC_STEPS=20 MH_WARMUP_STEPS=20 python run_bootstrap_method3.py
```

Another d = 12 configuration is:

```bash
LABEL=nl100_m50_w50 BOOT_RUNS=35 CHUNK_SIZE=35 N_LIVE=100 NS_MCMC_STEPS=50 MH_WARMUP_STEPS=50 python run_bootstrap_method3.py
```

For the stronger d = 12 live-point configuration:

```bash
LABEL=nl300_m50_w50 BOOT_RUNS=35 CHUNK_SIZE=35 N_LIVE=300 NS_MCMC_STEPS=50 MH_WARMUP_STEPS=50 python run_bootstrap_method3.py
```

For higher-dimensional experiments, set `P` and `N_LIVE` explicitly. For example:

```bash
LABEL=d36_ds36_nl900_m50_w50 P=36 N_LIVE=900 NS_MCMC_STEPS=50 MH_WARMUP_STEPS=50 BOOT_RUNS=35 CHUNK_SIZE=35 python run_bootstrap_method3.py
```

and

```bash
LABEL=d72_ds72_nl1800_m50_w50 P=72 N_LIVE=1800 NS_MCMC_STEPS=50 MH_WARMUP_STEPS=50 BOOT_RUNS=35 CHUNK_SIZE=35 python run_bootstrap_method3.py
```

The exact number of runs can be changed using:

```text
BOOT_RUNS
CHUNK_SIZE
CHUNK_ID
SLURM_ARRAY_TASK_ID
```

---

## Important environment variables

The main runner reads settings from environment variables.

### Data and model

```text
N                       sample size, default 600
P                       logistic-regression dimension, default 12
DATA_SEED               dataset seed, default 415
INCLUDE_INTERCEPT        whether to include an intercept, default False
TAU_PRIOR                intercept prior scale, default 1.0
```

### Nested Sampling

```text
N_LIVE                   number of live points
NS_MCMC_STEPS            production MCMC steps per replacement
MH_WARMUP_STEPS          warmup MCMC steps per replacement
N_ITER_MAX               maximum NS iterations
TOL_LOGZ                 stopping tolerance for logZ trace
TOL_TAIL                 stopping tolerance for tail contribution
PATIENCE                 stability window
STABLE_REPEATS           number of stable windows required
```

### Bootstrap

```text
N_BOOT_PER_RUN           bootstrap replicates per observed NS run, default 200
```

The following are fixed in the dissertation-matching runner:

```text
ANCHOR_POOL_SIZE_H = 100
ANCHOR_BLOCK_SIZE = n_live
LADDER_SWEEPS_S = 10
N_SHRINK = 1
```

---

## Output structure

The main runner writes results under:

```text
results/bootstrap_runs/<LABEL>/
```

with subdirectories:

```text
ns_runs_out/
boot_method3_out/
chunk_summaries/
```

The observed Nested Sampling outputs are saved in:

```text
results/bootstrap_runs/<LABEL>/ns_runs_out/
```

The Method 3 bootstrap outputs are saved in:

```text
results/bootstrap_runs/<LABEL>/boot_method3_out/
```

Chunk-level summaries are saved in:

```text
results/bootstrap_runs/<LABEL>/chunk_summaries/
```

The saved bootstrap files include metadata such as:

```text
anchor_pool_size_H
anchor_block_size
ladder_sweeps_S
n_shrink
candidate_pool_size
bootstrap_anchor_pool_size
```

These fields are useful for checking that the output was generated using the dissertation-matching implementation.

---

## Merging outputs

For chunked or Slurm-array runs, merge chunk outputs using the provided merge scripts.

Typical utilities include:

```bash
python merge_chunks.py
python merge_bootstrap_runs.py
```

The exact merge command may depend on the labels and folder names used for the experiment.

---

## Diagnostic and plotting scripts

The repository contains several scripts used for dissertation diagnostics and plots, including:

```text
make_bootstrap_diagnostics.py
make_bootstrap_sd_plot.py
make_bootstrap_vs_skilling_all.py
make_individual_bootstrap_vs_skilling.py
make_qq_plot_boot_vs_ref.py
report_diagnostics_clean.py
```

These scripts are used to produce or inspect diagnostics such as:

- coverage summaries;
- Q--Q plots;
- Wasserstein diagnostics;
- bootstrap standard-deviation comparisons;
- comparisons with Skilling's information heuristic.

---

## Notes on legacy outputs

Some saved outputs may have been generated before the final dissertation-aligned code cleanup.

For final consistency with the submitted dissertation, prefer outputs whose filenames or metadata include:

```text
anchorH100_min1ofnlive
```

Older outputs with tags such as:

```text
lmin1of100
lN1000
```

may correspond to earlier anchor-pool experiments and should not be treated as the canonical dissertation-matching implementation unless explicitly intended.

---

## Method summary

The submitted Method 3 bootstrap can be summarised as follows.

1. Run one MCMC-based Nested Sampling calculation.
2. Store the dead-point log-likelihood ladder.
3. Store production-phase phantom log-likelihood values from constrained MCMC replacement steps.
4. Form a candidate pool from dead and phantom log-likelihood values, retaining multiplicities.
5. Simulate a lowest-point anchor pool with `H = 100`.
6. For each bootstrap replicate:
   - draw one anchor from the anchor pool;
   - apply `S = 10` anchored empirical two-sweep ladder updates;
   - simulate one independent static shrinkage path;
   - compute the bootstrap log-evidence by log-sum-exp.
7. Compare the resulting bootstrap distribution with repeated-run Nested Sampling benchmarks.

---

## Citation and reference

The core Nested Sampling reference is:

```text
Skilling, J. (2006). Nested Sampling for Bayesian Computations.
Bayesian Analysis, 1(4), 833--859.
```

This repository is dissertation code and is not intended as a polished software package.

---

## License

Academic use only.

---

## Contact

Shrivar Singh  
University of Warwick
