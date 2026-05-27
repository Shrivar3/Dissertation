#!/bin/bash

#SBATCH --job-name=boot_diag
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Shrivar.Singh@warwick.ac.uk

module purge
module load PyTorch/2.0-Miniconda3-4.12.0-Python-3.10.8

cd /storage/stats/stuccn/Dissertation
source .venv/bin/activate

export PYTHONPATH=/storage/stats/stuccn/Dissertation/src

echo "=================================================="
echo "Bootstrap diagnostics job started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"
echo "=================================================="

python make_bootstrap_diagnostics.py

status=$?

echo "=================================================="
echo "Bootstrap diagnostics job finished"
echo "Exit status: $status"
echo "End time: $(date)"
echo "=================================================="

deactivate

exit $status
