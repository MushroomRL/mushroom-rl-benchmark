#!/usr/bin/env bash

###############################################################################
# SLURM Configurations

# Optional parameters

# Mandatory parameters
#SBATCH -J benchmark_2024-01-25_14-19-43
#SBATCH -a 0-24
#SBATCH -t 1-00:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu=6000
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
		--results_dir ./logs/benchmark_2024-01-25_14-19-43/Hopper/SAC --quiet False --show_progress_bar False --actor_lr 0.0001 --critic_lr 0.0003 --warmup_transitions 10000 --initial_replay_size 5000 --max_replay_size 500000 --n_features 256 --batch_size 256 --tau 0.005 --lr_alpha 0.0003 --n_epochs 50 --n_steps 30000 --n_episodes_test 10 --env_name Gym.Hopper-v4 --env_params_horizon 1000 --env_params_gamma 0.99 --env Hopper --agent SAC  &

            
wait # This will wait until both scripts finish
echo "########################################################################"
echo "...done."
