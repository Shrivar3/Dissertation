#!/bin/bash

#SBATCH --job-name=ns_array_test
#SBATCH --time=08:00:00 #yourtiming max: 1-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3700
#SBATCH --output=slurm-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Shrivar.Singh@warwick.ac.uk
#SBATCH --array=0-19

module purge
module load PyTorch/2.0-Miniconda3-4.12.0-Python-3.10.8

cd /storage/stats/stuccn/Dissertation
source .venv/bin/activate

export PYTHONPATH=/storage/stats/stuccn/Dissertation/src
export CHUNK_ID=$SLURM_ARRAY_TASK_ID
export RUNS_PER_CHUNK=50

echo "Running on node:"
hostname
echo "Working directory:"
pwd
echo "Python version:"
python --version
echo "PYTHONPATH:"
echo $PYTHONPATH
echo "CHUNK_ID:"
echo $CHUNK_ID
echo "RUNS_PER_CHUNK:"
echo $RUNS_PER_CHUNK
echo "Contents of src:"
ls src
echo "Starting chunk run..."

python run_chunk.py

echo "Finished chunk run."

deactivate
