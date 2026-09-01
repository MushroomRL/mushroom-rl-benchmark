from pathlib import Path
from unittest.mock import patch

import yaml

from mushroom_rl_benchmark.core import BenchmarkConfiguration, BenchmarkSuite


def test_cli_test_mode_expands_all_algorithms_without_running(tmp_path):
    output_dir = tmp_path / 'benchmark'
    config_dir = Path(__file__).parents[1] / 'cfg'

    suite = BenchmarkSuite(config_dir=config_dir)
    suite.add_environment('GridWorld')
    suite.run(exec_type='parallel_test', overrides=['+n_seeds=3', f'output_root={output_dir}'])

    configuration = BenchmarkConfiguration(config_dir)
    expected = set(configuration.get_available_agents('GridWorld'))
    with (output_dir / 'params' / 'GridWorld.yaml').open(encoding='utf-8') as stream:
        parameters = yaml.safe_load(stream)

    assert set(parameters['agent_params']) == expected
    assert parameters['run_params']['n_runs'] == 3


def test_demo_prepares_two_short_seeds(tmp_path):
    output_dir = tmp_path / 'benchmark'
    config_dir = Path(__file__).parents[1] / 'cfg'

    suite = BenchmarkSuite(config_dir=config_dir)
    suite.set_demo_run_params()
    suite.add_experiment('GridWorld', 'QLearning')
    suite.run(exec_type='parallel_test', overrides=['+n_seeds=2', f'output_root={output_dir}'])

    with (output_dir / 'params' / 'GridWorld.yaml').open(encoding='utf-8') as stream:
        parameters = yaml.safe_load(stream)

    assert parameters['run_params']['n_runs'] == 2
    assert parameters['run_params']['n_epochs'] == 10
    assert parameters['run_params']['n_steps'] == 15000


def test_suite_uses_configured_number_of_seeds_by_default(tmp_path):
    output_dir = tmp_path / 'benchmark'
    config_dir = Path(__file__).parents[1] / 'cfg'

    suite = BenchmarkSuite(config_dir=config_dir)
    suite.add_experiment('GridWorld', 'QLearning')
    suite.run(exec_type='sequential_test', overrides=[f'output_root={output_dir}'])

    with (output_dir / 'params' / 'GridWorld.yaml').open(encoding='utf-8') as stream:
        parameters = yaml.safe_load(stream)

    assert parameters['run_params']['n_runs'] == 25


def test_parallel_suite_uses_one_hydra_multirun(tmp_path):
    config_dir = Path(__file__).parents[1] / 'cfg'
    suite = BenchmarkSuite(config_dir=config_dir)
    suite.add_experiments('GridWorld', ['QLearning', 'SARSA'])

    with patch.object(suite, '_launch_jobs') as launch_jobs:
        with patch('mushroom_rl_benchmark.core.suite.aggregate_results') as aggregate_results:
            suite.run(exec_type='parallel', overrides=['+n_seeds=3', f'output_root={tmp_path}'])

    launch_jobs.assert_called_once()
    sweep_config_dir, sweep_overrides, exec_type, overrides, results_dir, _ = launch_jobs.call_args.args
    assert sweep_config_dir == tmp_path / 'params' / 'hydra'
    assert len(sweep_overrides[0].split(',')) == 6
    assert exec_type == 'parallel'
    assert overrides == ['+n_seeds=3', f'output_root={tmp_path}']
    assert results_dir == tmp_path
    assert aggregate_results.call_count == 2


def test_slurm_suite_does_not_aggregate_results_during_submission(tmp_path):
    config_dir = Path(__file__).parents[1] / 'cfg'
    suite = BenchmarkSuite(config_dir=config_dir)
    suite.add_experiment('GridWorld', 'QLearning')

    with patch.object(suite, '_launch_jobs'):
        with patch('mushroom_rl_benchmark.core.suite.aggregate_results') as aggregate_results:
            suite.run(exec_type='slurm', overrides=['+n_seeds=3', f'output_root={tmp_path}'])

    aggregate_results.assert_not_called()


def test_slurm_launcher_does_not_wait_for_completion(tmp_path):
    config_dir = Path(__file__).parents[1] / 'cfg'
    suite = BenchmarkSuite(config_dir=config_dir)

    with patch('mushroom_rl_benchmark.core.suite.subprocess.Popen') as popen:
        suite._launch_jobs(tmp_path, ['+benchmark_job=test'], 'slurm', None, tmp_path, False)

    popen.assert_called_once()


def test_suite_supports_different_configured_seed_counts(tmp_path):
    config_dir = Path(__file__).parents[1] / 'cfg'
    suite = BenchmarkSuite(config_dir=config_dir)
    suite.add_experiments('GridWorld', ['QLearning', 'SARSA'])
    suite._experiments[1]['run_params']['n_runs'] = 3

    with patch.object(suite, '_launch_jobs') as launch_jobs:
        with patch('mushroom_rl_benchmark.core.suite.aggregate_results'):
            suite.run(exec_type='parallel', overrides=[f'output_root={tmp_path}'])

    launch_jobs.assert_called_once()
    _, sweep_overrides, _, _, _, _ = launch_jobs.call_args.args
    assert len(sweep_overrides) == 1
    assert len(sweep_overrides[0].split(',')) == 28
