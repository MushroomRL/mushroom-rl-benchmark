#!/usr/bin/env bash

###############################################################################
# SLURM Configurations
#SBATCH -J LunarLanderContinuous-v2/TD3
#SBATCH -a 0-24
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=4000
#SBATCH -o /home/tateo/mushroom-rl-benchmark/logs/benchmark/LunarLanderContinuous-v2/TD3/%A_%a.out
#SBATCH -e /home/tateo/mushroom-rl-benchmark/logs/benchmark/LunarLanderContinuous-v2/TD3/%A_%a.err
###############################################################################
# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

# Program specific arguments
CMD="python3 /home/tateo/mushroom-rl-benchmark/mushroom_rl_benchmark/experiment/slurm/run_script.py \
		${@:1}\
		--seed $SLURM_ARRAY_TASK_ID"

echo "$CMD"
eval $CMD
