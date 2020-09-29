
import numpy as np
from mushroom_rl.core import Core
from mushroom_rl.environments import Gym
from mushroom_rl.algorithms import Agent

from mushroom_rl_benchmark import BenchmarkVisualizer


if __name__ == "__main__":

    experiment_path= './logs/a2c_pendulum'
    
    visualizer = BenchmarkVisualizer.from_path(experiment_path)
    visualizer.show_agent(call_mdp_render=False)
    