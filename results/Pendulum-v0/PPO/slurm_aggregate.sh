#!/usr/bin/env bash

###############################################################################
# SLURM Configurations
#SBATCH -J Pendulum-v0/PPO_aggregate
#SBATCH -t 03:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=2000
#SBATCH -o /work/scratch/dt11kypo/logs/benchmark/Pendulum-v0/PPO/%A.out
#SBATCH -e /work/scratch/dt11kypo/logs/benchmark/Pendulum-v0/PPO/%A.err
###############################################################################
# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

# Program specific arguments
CMD="python3 /work/home/dt11kypo/mushroom-rl-benchmark/mushroom_rl_benchmark/experiment/slurm/aggregate_results.py \
		${@:1}\
		--seed 0"

echo "$CMD"
eval $CMD
