from mushroom_rl_benchmark.core import BenchmarkDataLoader, BenchmarkExperiment, aggregate_results


def test_experiment_persists_and_aggregates_value_function_trace(tmp_path):
    experiment = BenchmarkExperiment(agent_name='QLearning', env_name='GridWorld', agent_params={},
                                     env_params={'height': 5, 'width': 5, 'goal': [3, 3]},
                                     results_dir=tmp_path / 'GridWorld', n_epochs=1, n_steps=20,
                                     n_steps_test=50, log_console=False)

    experiment.run(seed=0)
    aggregated = aggregate_results(experiment.path)
    loader = BenchmarkDataLoader(experiment.path)

    assert aggregated['V'].shape == (1, 2)
    assert loader.value_function_found
    assert loader.load_aggregate_file('J').shape == (1, 2)
