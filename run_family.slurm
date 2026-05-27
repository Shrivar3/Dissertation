#!/bin/bash

#SBATCH --job-name=ns_family
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3700
#SBATCH --output=slurm-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Shrivar.Singh@warwick.ac.uk
#SBATCH --array=0-5

module purge
module load PyTorch/2.0-Miniconda3-4.12.0-Python-3.10.8

cd /storage/stats/stuccn/Dissertation
source .venv/bin/activate

export PYTHONPATH=/storage/stats/stuccn/Dissertation/src
export CHUNK_ID=$SLURM_ARRAY_TASK_ID

# These are set when you submit
# export RUNS_PER_CHUNK=50
# export N=600
# export P=36 or 72
# export N_LIVE=...
# export NS_MCMC_STEPS=50
# export MH_WARMUP_STEPS=50
# export DATA_SEED=36 or 72
# export BASE_SEED0=1000
# export LABEL=...

echo "Running label: $LABEL"
echo "Chunk ID: $CHUNK_ID"
echo "N: $N"
echo "P: $P"
echo "N_LIVE: $N_LIVE"
echo "NS_MCMC_STEPS: $NS_MCMC_STEPS"
echo "MH_WARMUP_STEPS: $MH_WARMUP_STEPS"
echo "RUNS_PER_CHUNK: $RUNS_PER_CHUNK"
echo "DATA_SEED: $DATA_SEED"
echo "BASE_SEED0: $BASE_SEED0"

python run_family_chunk.py

deactivate
