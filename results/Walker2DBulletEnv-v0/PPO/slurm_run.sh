#!/usr/bin/env bash

###############################################################################
# SLURM Configurations
#SBATCH -J Walker2DBulletEnv-v0/PPO
#SBATCH -a 0-24
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=4000
#SBATCH -o /work/home/dt11kypo/mushroom-rl-benchmark/logs/benchmark_2021-01-26-18-04-08/Walker2DBulletEnv-v0/PPO/%A_%a.out
#SBATCH -e /work/home/dt11kypo/mushroom-rl-benchmark/logs/benchmark_2021-01-26-18-04-08/Walker2DBulletEnv-v0/PPO/%A_%a.err
###############################################################################
# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

# Program specific arguments
CMD="python3 /work/home/dt11kypo/mushroom-rl-benchmark/mushroom_rl_benchmark/experiment/slurm/run_script.py \
		${@:1}\
		--seed $SLURM_ARRAY_TASK_ID"

echo "$CMD"
eval $CMD
