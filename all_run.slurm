#! /bin/sh

#SBATCH --job-name=all_run_results
#SBATCH --output=all_run_output.out # redirect stdout
#SBATCH --error=all_run_errors.err # redirect stderr
#SBATCH --partition=studentkillable # (see resources section)
#SBATCH --time=1000 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=3 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total

python apply_scores_as_new_col.py 

