#!/usr/bin/env bash

###############################################################################
# SLURM Configurations

# Optional parameters

# Mandatory parameters
#SBATCH -J benchmark_2024-01-25_14-19-43
#SBATCH -a 4
#SBATCH -t 1-00:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=8000
#SBATCH -o ./logs/benchmark_2024-01-25_14-19-43/slurm_logs/%A_%a.out
#SBATCH -e ./logs/benchmark_2024-01-25_14-19-43/slurm_logs/%A_%a.err

###############################################################################
# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

# Program specific arguments


# Program specific arguments

echo "Running scripts in parallel..."
echo "########################################################################"
            
                
python3  /home/hendawy/code/mushroom-rl-benchmark/mushroom_rl_benchmark/core/run.py \
		--seed $SLURM_ARRAY_TASK_ID \
		--results_dir ./logs/benchmark_2024-01-25_14-19-43/Walker2d/TD3 --quiet False --show_progress_bar False --actor_lr 0.001 --critic_lr 0.001 --n_features 400 300 --initial_replay_size 1000 --max_replay_size 1000000 --tau 0.005 --batch_size 100 --n_epochs 50 --n_steps 30000 --n_episodes_test 10 --env_name Gym.Walker2d-v4 --env_params_horizon 1000 --env_params_gamma 0.99 --env Walker2d --agent TD3  &

            
wait # This will wait until both scripts finish
echo "########################################################################"
echo "...done."
