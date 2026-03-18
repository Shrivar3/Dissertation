#!/bin/bash

#SBATCH --job-name=ns_small_test
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3700
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Shrivar.Singh@warwick.ac.uk

module purge
module load PyTorch/2.0-Miniconda3-4.12.0-Python-3.10.8

cd /storage/stats/stuccn/Dissertation
source .venv/bin/activate

export PYTHONPATH=/storage/stats/stuccn/Dissertation/src

echo "Running on node:"
hostname
echo "Working directory:"
pwd
echo "Python version:"
python --version
echo "PYTHONPATH:"
echo $PYTHONPATH
echo "Contents of src:"
ls src
echo "Starting test run..."

python test_small_run.py

echo "Finished run."

deactivate
