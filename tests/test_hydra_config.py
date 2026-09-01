from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore

from mushroom_rl_benchmark.core import BenchmarkConfiguration


def test_all_environment_configs_and_launcher_profiles_compose():
    config_dir = Path(__file__).parents[1] / 'cfg'
    experiment = dict(env_id='GridWorld', agent_name='QLearning')
    ConfigStore.instance().store(group='experiment', name='test_job', node=experiment)

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for profile in ('sequential', 'parallel', 'slurm'):
            profile_config = compose(config_name='benchmark',
                                     overrides=['+experiment=test_job', f'profile={profile}'],
                                     return_hydra_config=True)
            assert profile_config.experiment.env_id == 'GridWorld'
            assert profile_config.log_console
            if profile == 'slurm':
                launcher = profile_config.hydra.launcher
                assert launcher.timeout_min == 1440
                assert launcher.cpus_per_task == 1
                assert launcher.mem_gb == 8
                assert launcher.array_parallelism == 256


def test_programmatic_configuration_uses_the_same_environment_files():
    config_dir = Path(__file__).parents[1] / 'cfg'
    config = BenchmarkConfiguration(config_dir)

    env_params, run_params, _ = config.get_experiment_params('GridWorld', 'QLearning')

    assert config.get_environment_id('GridWorld') == 'GridWorld'
    assert env_params['name'] == 'GridWorld'
    assert env_params['params']['goal'] == [8, 7]
    assert run_params['n_runs'] == 25


def test_flattened_benchmark_job_composes_at_the_root():
    config_dir = Path(__file__).parents[1] / 'cfg'
    job = dict(experiment=dict(env_id='GridWorld', agent_name='QLearning'), seed=2)
    ConfigStore.instance().store(group='benchmark_job', name='test_job', node=job, package='_global_')

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name='benchmark', overrides=['+benchmark_job=test_job'])

    assert config.experiment.env_id == 'GridWorld'
    assert config.experiment.agent_name == 'QLearning'
    assert config.seed == 2


def test_configuration_expands_all_and_explicit_selections():
    config_dir = Path(__file__).parents[1] / 'cfg'
    config = BenchmarkConfiguration(config_dir)

    all_experiments = config.select(['all'], ['all'])
    expected = sum(len(config.get_available_agents(environment)) for environment in config.envs)
    assert len(all_experiments) == expected

    selected = config.select(['Ant', 'Walker2d'], ['PPO', 'TD3'])
    assert selected == (
        ('Ant', 'PPO'),
        ('Ant', 'TD3'),
        ('Walker2d', 'PPO'),
        ('Walker2d', 'TD3'),
    )

    with pytest.raises(ValueError, match='QLearning is not configured for Ant'):
        config.select(['Ant'], ['QLearning'])
