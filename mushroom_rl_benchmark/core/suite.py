import sys
import json
import yaml
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir

from mushroom_rl_benchmark.core import BenchmarkConfiguration, BenchmarkParams, aggregate_results


class BenchmarkSuite:
    """
    Class to orchestrate the execution of multiple experiments.

    """
    def __init__(self, config_dir='cfg', logger=None):
        """
        Constructor.

        Args:
            config_dir (str, 'cfg'): config directory;
            logger (Logger, None): logger used to print the selected jobs.

        """
        self._config_dir = Path(config_dir).resolve()
        self._config = BenchmarkConfiguration(self._config_dir)
        self._logger = logger
        self._param_logger = BenchmarkParams()
        self._experiments = list()
        self._demo_run_params = None

    def set_demo_run_params(self, n_epochs=10, n_steps=15000, n_episodes=10, n_episodes_test=5, n_steps_test=1000):
        self._demo_run_params = dict(n_epochs=n_epochs,
                                     n_steps=n_steps,
                                     n_episodes=n_episodes,
                                     n_episodes_test=n_episodes_test,
                                     n_steps_test=n_steps_test)

    def add_selected(self, environment_names, agent_names):
        """
        Add the selected environments and agents to the suite.

        Args:
            environment_names (list): names of the environments, or ``all``;
            agent_names (list): names of the agents, or ``all``.

        """
        for environment_name, agent_name in self._config.select(environment_names, agent_names):
            self.add_experiment(environment_name, agent_name)

    def add_full_benchmark(self):
        for env in self._config.envs:
            self.add_environment(env)

    def add_environment(self, environment_name):
        """
        Add all configured experiments for the same environment to the suite.

        Args:
            environment_name (str): name of the environment for the experiment (E.g. Gym.Pendulum-v0).

        """
        agent_names_list = self._config.get_available_agents(environment_name)
        self.add_experiments(environment_name, agent_names_list)

    def add_experiments(self, environment_name, agent_names_list):
        """
        Add a set of experiments for the same environment to the suite.

        Args:
            environment_name (str): name of the environment for the experiment (E.g. Gym.Pendulum-v0);
            agent_names_list (list): list of names of the agents for the experiments.

        """
        for agent_name in agent_names_list:
            self.add_experiment(environment_name, agent_name)

    def add_experiment(self, environment_name, agent_name):
        """
        Add a single experiment to the benchmarking suite.

        Args:
            environment_name (str): name of the environment for the experiment (E.g. Gym.Pendulum-v0);
            agent_name (str): name of the agent for the experiments.

        """
        assert environment_name in self._config.envs
        assert agent_name in self._config.get_available_agents(environment_name)

        env_params, run_params, agent_params = self._config.get_experiment_params(environment_name, agent_name)
        env_id = self._config.get_environment_id(environment_name)
        env_name = env_params['name']
        env_params = (env_params.get('params') or dict()).copy()
        agent_params = (agent_params or dict()).copy()
        experiment_params = dict(env_id=env_id, env_name=env_name, env_params=env_params,
                                 agent_name=agent_name, agent_params=agent_params, run_params=run_params.copy())
        run_params = experiment_params['run_params']
        if self._demo_run_params is not None:
            self._overwrite_run_parameters(run_params)
        self._param_logger.add_experiment_params(**experiment_params)
        self._experiments.append(experiment_params)

    def run(self, exec_type=None, test=False, overrides=None):
        """
        Run the benchmarking suite.

        Args:
            exec_type (str, None): type of benchmark running. You can choose between sequential, parallel and slurm.
                If you append "_test", the set of jobs is printed instead of running the benchmark;
            test (bool, False): whether to print the set of jobs instead of running the benchmark;
            overrides (list, None): additional Hydra overrides.

        """
        if exec_type is not None and exec_type.endswith('_test'):
            test = True
            exec_type = exec_type[:-5]
        if exec_type is not None and not (self._config_dir / 'profile' / f'{exec_type}.yaml').is_file():
            raise AttributeError('wrong execution type selected')
        if not self._experiments:
            raise RuntimeError('No experiments have been added to the suite')

        config_overrides = list(overrides or [])
        if exec_type is not None:
            config_overrides.insert(0, f'profile={exec_type}')
        with initialize_config_dir(config_dir=str(self._config_dir), version_base=None):
            config = compose(config_name='benchmark', overrides=config_overrides)
        results_dir = Path(config.output_root)
        log_console = config.log_console

        n_seeds = config.get('n_seeds')
        if n_seeds is not None:
            for experiment in self._experiments:
                experiment['run_params']['n_runs'] = n_seeds

        self._param_logger.save_params(results_dir)
        for experiment in self._experiments:
            path = results_dir / experiment['env_id'] / experiment['agent_name']
            n_seeds = experiment['run_params']['n_runs']
            if self._logger is not None:
                self._logger.info(f'{path}: {n_seeds} seeds')

        if test:
            return

        sweep_config_dir, sweep_overrides = self._write_sweep(results_dir)
        self._launch_jobs(sweep_config_dir, sweep_overrides, exec_type, overrides, results_dir, log_console)

        if exec_type != 'slurm':
            for experiment in self._experiments:
                path = results_dir / experiment['env_id'] / experiment['agent_name']
                aggregate_results(path, log_console=log_console)

    def _launch_jobs(self, sweep_config_dir, sweep_overrides, exec_type, overrides, results_dir, log_console):
        hydra_args = [sys.executable, '-m', 'mushroom_rl_benchmark.core.run', '--multirun',
                      '--config-path', str(self._config_dir), '--config-dir', str(sweep_config_dir)]
        if exec_type is not None:
            hydra_args.append(f'profile={exec_type}')
        hydra_args.extend([*sweep_overrides,
                           f'output_root={json.dumps(str(results_dir))}',
                           f'log_console={str(log_console).lower()}',
                           *(overrides or [])])

        if exec_type == 'slurm':
            results_dir.mkdir(parents=True, exist_ok=True)
            with (results_dir / 'slurm.log').open('a') as log_file:
                process = subprocess.Popen(hydra_args, stdin=subprocess.DEVNULL, stdout=log_file,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            if self._logger is not None:
                self._logger.info(f'Slurm submission process started with PID {process.pid}')
        else:
            subprocess.run(hydra_args, check=True)

    def _write_sweep(self, results_dir):
        sweep_config_dir = results_dir / 'params' / 'hydra'
        job_config_dir = sweep_config_dir / 'benchmark_job'
        job_config_dir.mkdir(parents=True, exist_ok=True)

        names = list()
        for i, experiment in enumerate(self._experiments):
            for seed in range(experiment['run_params']['n_runs']):
                name = f'benchmark_job_{i}_{seed}'
                job = dict(experiment=experiment, seed=seed)
                with (job_config_dir / f'{name}.yaml').open('w') as config_file:
                    config_file.write('# @package _global_\n')
                    yaml.safe_dump(job, config_file, sort_keys=False)
                names.append(name)

        return sweep_config_dir, ['+benchmark_job=' + ','.join(names)]

    def _overwrite_run_parameters(self, run_params):
        run_params['n_epochs'] = self._demo_run_params['n_epochs']
        if 'n_steps' in run_params:
            run_params['n_steps'] = self._demo_run_params['n_steps']
        else:
            run_params['n_episodes'] = self._demo_run_params['n_episodes']
        if 'n_episodes_test' in run_params:
            run_params['n_episodes_test'] = self._demo_run_params['n_episodes_test']
        else:
            run_params['n_steps_test'] = self._demo_run_params['n_steps_test']
