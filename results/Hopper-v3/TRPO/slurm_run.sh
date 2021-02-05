#!/usr/bin/env bash

###############################################################################
# SLURM Configurations
#SBATCH -J Hopper-v3/TRPO
#SBATCH -a 0-24
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=4000
#SBATCH -o /work/scratch/dt11kypo/logs/benchmark/Hopper-v3/TRPO/%A_%a.out
#SBATCH -e /work/scratch/dt11kypo/logs/benchmark/Hopper-v3/TRPO/%A_%a.err
###############################################################################
# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

# Program specific arguments
CMD="python3 /work/home/dt11kypo/mushroom-rl-benchmark/mushroom_rl_benchmark/experiment/slurm/run_script.py \
		${@:1}\
		--seed $SLURM_ARRAY_TASK_ID"

echo "$CMD"
eval $CMD
