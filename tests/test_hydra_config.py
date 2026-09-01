from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore

from mushroom_rl_benchmark.core import BenchmarkConfiguration


def test_all_environment_configs_and_launcher_profiles_compose():
    config_dir = Path(__file__).parents[1] / 'cfg'
    configuration = BenchmarkConfiguration(config_dir)
    experiment = configuration.experiment('GridWorld', 'QLearning')
    ConfigStore.instance().store(group='experiment', name='test_job', node=experiment)

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for profile in ('basic', 'parallel', 'slurm'):
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

    experiment = config.experiment('GridWorld', 'QLearning')

    assert experiment['env_name'] == 'GridWorld'
    assert experiment['env_params']['goal'] == [8, 7]
    assert experiment['run_params']['n_runs'] == 25


def test_configuration_expands_all_and_explicit_selections():
    config_dir = Path(__file__).parents[1] / 'cfg'
    config = BenchmarkConfiguration(config_dir)

    all_experiments = config.select(['all'], ['all'])
    expected = sum(len(config.get_available_agents(environment)) for environment in config.envs)
    assert len(all_experiments) == expected

    selected = config.select(['Ant', 'Walker2d'], ['PPO', 'TD3'])
    assert [(job['env_id'], job['agent_name']) for job in selected] == [
        ('Ant-v5', 'PPO'),
        ('Ant-v5', 'TD3'),
        ('Walker2d-v5', 'PPO'),
        ('Walker2d-v5', 'TD3'),
    ]

    with pytest.raises(ValueError, match='QLearning is not configured for Ant'):
        config.select(['Ant'], ['QLearning'])
