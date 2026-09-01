import yaml

import benchmark
from mushroom_rl_benchmark.core import BenchmarkConfiguration


def test_cli_test_mode_expands_all_algorithms_without_running(tmp_path):
    output_dir = tmp_path / 'benchmark'

    benchmark.main(['--env', 'GridWorld', '--algorithm', 'all', '--seeds', '3',
                    '--output-dir', str(output_dir), '--test'])

    configuration = BenchmarkConfiguration(benchmark.CONFIG_DIR)
    expected = set(configuration.get_available_agents('GridWorld'))
    with (output_dir / 'params' / 'GridWorld.yaml').open(encoding='utf-8') as stream:
        parameters = yaml.safe_load(stream)

    assert set(parameters['agent_params']) == expected
    assert parameters['run_params']['n_runs'] == 3


def test_demo_prepares_two_short_seeds():
    configuration = BenchmarkConfiguration(benchmark.CONFIG_DIR)
    experiment = configuration.experiment('GridWorld', 'QLearning')

    prepared = benchmark._prepare_experiment(experiment, n_seeds=2, demo=True)

    assert prepared['run_params']['n_runs'] == 2
    assert prepared['run_params']['n_epochs'] == 2
    assert prepared['run_params']['n_steps'] <= 500


def test_experiment_name_describes_algorithm_and_environment():
    experiment = {'agent_name': 'PPO', 'env_id': 'Gymnasium/Pendulum-v1'}

    assert benchmark._experiment_name(experiment) == 'exp_PPO_Gymnasium_Pendulum-v1'
