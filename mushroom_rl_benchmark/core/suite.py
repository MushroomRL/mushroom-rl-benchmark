from experiment_launcher import Launcher
from experiment_launcher.utils import is_local

from mushroom_rl_benchmark.core import BenchmarkConfiguration, BenchmarkParams
from mushroom_rl_benchmark.utils.parameter_renaming import mask_env_parameters
from mushroom_rl_benchmark.utils.sweep import generate_sweep


class BenchmarkSuite:
    """
    Class to orchestrate the execution of multiple experiments.

    """
    def __init__(self, config_dir='cfg', n_seeds=25):
        """
        Constructor.

        Args:
            config_dir (str, 'cfg'): config directory;
            n_seeds (int, 25): number of seeds to evaluate.
        
        """
        self._config = BenchmarkConfiguration(config_dir)
        self._param_logger = BenchmarkParams()

        self._launcher = Launcher(
            exp_file="mushroom_rl_benchmark.core.run",
            n_seeds=n_seeds,
            compact_dirs=True,
            **self._config.suite_params
        )

        self._environment_dict = dict()
        self._demo_run_params = None

    def set_demo_run_params(self, n_epochs=10, n_steps=15000, n_episodes=10, n_episodes_test=5, n_steps_test=1000):
        self._demo_run_params = dict(n_epochs=n_epochs,
                                     n_steps=n_steps,
                                     n_episodes=n_episodes,
                                     n_episodes_test=n_episodes_test,
                                     n_steps_test=n_steps_test)

    def add_full_benchmark(self, sweep_name=''):
        for env in self._config.envs:
            self.add_environment(env, sweep_name)

    def add_environment(self, environment_name, sweep_name=''):
        """
        Add all configured experiments for the same environment to the suite.

        Args:
            environment_name (str): name of the environment for the experiment (E.g. Gym.Pendulum-v0);
            sweep_name (str,''): parameter sweep to be executed for the experiments.
                If not specified, no sweep is executed.

        """
        agent_names_list = self._config.get_available_agents(environment_name)
        self.add_experiments(environment_name, agent_names_list, sweep_name=sweep_name)

    def add_experiments(self, environment_name, agent_names_list, sweep_name=''):
        """
        Add a set of experiments for the same environment to the suite.

        Args:
            environment_name (str): name of the environment for the experiment (E.g. Gym.Pendulum-v0);
            agent_names_list (list): list of names of the agents for the experiments;
            sweep_name (str,''): parameter sweep to be executed for the experiments.
                If not specified, no sweep is executed.

        """
        for agent_name in agent_names_list:
            self.add_experiment(environment_name, agent_name, sweep_name=sweep_name)

    def add_experiment(self, environment_name, agent_name, sweep_name=''):
        """
        Add a single to the benchmarking suite.

        Args:
            environment_name (str): name of the environment for the experiment (E.g. Gym.Pendulum-v0);
            agent_name (str): name of the agent for the experiments;
            sweep_name (str,''): parameter sweep to be executed for the experiment.
                If not specified, no sweep is executed.

        """
        assert environment_name in self._config.envs
        assert agent_name in self._config.get_available_agents(environment_name)

        env_params, run_params, agent_params = self._config.get_experiment_params(environment_name, agent_name)

        if self._demo_run_params is not None:
            self._overwrite_run_parameters(run_params)

        masked_env_params = mask_env_parameters(env_params)

        self._param_logger.add_experiment_params(environment_name, env_params, agent_name, agent_params, run_params)

        if sweep_name:
            self._add_sweep_experiment(agent_name, agent_params, environment_name, masked_env_params, run_params,
                                       sweep_name)
        else:
            self._launcher.add_experiment(env__=environment_name,
                                          agent__=agent_name,
                                          quiet=self._config.quiet,
                                          show_progress_bar=self._config.show_progress_bar,
                                          **agent_params,
                                          **run_params,
                                          **masked_env_params)

    def _add_sweep_experiment(self, agent_name, agent_params, environment_name, masked_env_params, run_params,
                              sweep_name):

        assert sweep_name in self._config.get_available_sweeps()
        sweep_config = self._config.get_sweep_params(sweep_name, agent_name)

        for sweep_name, sweep_params in generate_sweep(sweep_config):
            self._param_logger.add_sweep_params(agent_name, sweep_name, sweep_params)
            new_params = agent_params.copy()
            new_params.update(sweep_params)
            self._launcher.add_experiment(env__=environment_name,
                                          agent__=agent_name,
                                          sweep_name__=sweep_name,
                                          quiet=self._config.quiet,
                                          show_progress_bar=self._config.show_progress_bar,
                                          **new_params,
                                          **run_params,
                                          **masked_env_params)

    def run(self, exec_type=None):
        """
        Run the benchmarking suite

        Args:
            exec_type (str, None): type of benchmark running. you can choose between sequential, parallel and slurm.
                If you append "_test" to slurm or parallel, a print will show the set of calls, instead of running
                the benchmark
        """
        sequential = False
        if exec_type is None:
            local = is_local()
            test = False
        elif exec_type == 'sequential':
            local = True
            sequential = True
            test = False
        elif exec_type == 'parallel':
            local = True
            test = False
        elif exec_type == 'slurm':
            local = False
            test = False
        elif exec_type == 'slurm_test':
            local = False
            test = True
        elif exec_type == 'parallel_test':
            local = True
            test = True
        elif exec_type == 'sequential_test':
            local = True
            sequential = True
            test = True
        else:
            raise AttributeError('wrong execution type selected')

        self._param_logger.save_params(self._launcher.log_dir(local))
        self._launcher.run(local, test, sequential)

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
