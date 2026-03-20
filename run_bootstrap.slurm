#!/bin/bash

#SBATCH --job-name=bootstrap_family
#SBATCH --time=1-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3700
#SBATCH --output=slurm-bootstrap-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Shrivar.Singh@warwick.ac.uk

module purge
module load PyTorch/2.0-Miniconda3-4.12.0-Python-3.10.8

cd /storage/stats/stuccn/Dissertation
source .venv/bin/activate

export PYTHONPATH=/storage/stats/stuccn/Dissertation/src

echo "Running bootstrap family:"
echo "LABEL=$LABEL"
echo "MULTI_RUN_REF_PATH=$MULTI_RUN_REF_PATH"
echo "BOOT_RUNS=$BOOT_RUNS"
echo "N_BOOT_PER_RUN=$N_BOOT_PER_RUN"
echo "N=$N"
echo "P=$P"
echo "N_LIVE=$N_LIVE"
echo "NS_MCMC_STEPS=$NS_MCMC_STEPS"
echo "MH_WARMUP_STEPS=$MH_WARMUP_STEPS"
echo "DATA_SEED=$DATA_SEED"
echo "NS_BASE_SEED=$NS_BASE_SEED"
echo "BOOT_BASE_SEED=$BOOT_BASE_SEED"

python run_bootstrap_method3.py

deactivate
