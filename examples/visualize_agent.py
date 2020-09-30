from mushroom_rl_benchmark import BenchmarkVisualizer


if __name__ == "__main__":

    experiment_path = '../logs/small_benchmark/Gym_Pendulum-v0/A2C'
    
    visualizer = BenchmarkVisualizer.from_path(experiment_path)
    visualizer.show_agent(mdp_render=False)
