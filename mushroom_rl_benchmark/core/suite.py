import mushroom_rl_benchmark.builders
from mushroom_rl_benchmark.builders import EnvironmentBuilder
from mushroom_rl_benchmark.core.experiment import BenchmarkExperiment
from mushroom_rl_benchmark.core.logger import BenchmarkLogger
from mushroom_rl_benchmark.core.suite_visualizer import BenchmarkSuiteVisualizer


class BenchmarkSuite:
    """
    Class to orchestrate the execution of multiple experiments.

    """
    def __init__(self, log_dir=None, log_id=None, use_timestamp=True, parallel=None, slurm=None):
        """
        Constructor.

        Args:
            log_dir (str): path to the log directory (Default: ./logs or /work/scratch/$USER)
            log_id (str): log id (Default: benchmark[_YYYY-mm-dd-HH-MM-SS])
            use_timestamp (bool): select if a timestamp should be appended to the log id
            parallel (dict, None): parameters that are passed to the run_parallel method of the experiment
            slurm (dict, None): parameters that are passed to the run_slurm method of the experiment
        
        """
        self._experiment_structure = dict()
        self._environment_dict = dict()
        self._parameters_dict = dict()
        self._agent_list = []
        self._parallel = parallel
        self._slurm = slurm
        self.logger = BenchmarkLogger(log_dir=log_dir, log_id=log_id, use_timestamp=use_timestamp)

    def add_experiments(self, environment_name, environment_builder_params, agent_names_list,
                        agent_builders_params, **run_params):
        """
        Add a set of experiments for the same environment to the suite.

        Args:
            environment_name (str): name of the environment for the experiment (E.g. Gym.Pendulum-v0);
            environment_builder_params (dict): parameters for the environment builder;
            agent_names_list (list): list of names of the agents for the experiments;
            agent_builders_params (list): list of dictionaries containing the parameters for the agent builder;
            run_params: Parameters that are passed to the run method of the experiment.

        """
        self.add_environment(environment_name, environment_builder_params, **run_params)

        for agent_name, agent_params in zip(agent_names_list, agent_builders_params):
            self.add_agent(environment_name, agent_name, agent_params)

    def add_environment(self, environment_name, environment_builder_params, **run_params):
        """
        Add an environment to the benchmarking suite.

        Args:
            environment_name (str): name of the environment for the experiment (E.g. Gym.Pendulum-v0);
            environment_builder_params (dict): parameters for the environment builder;
            run_params: Parameters that are passed to the run method of the experiment.

        """
        if environment_name in self._environment_dict:
            raise AttributeError('The environment {} has been already added to the benchmark'.format(environment_name))

        self._environment_dict[environment_name] = dict(
            build_params=environment_builder_params,
            run_params=run_params
        )

        self._experiment_structure[environment_name] = dict()

    def add_agent(self, environment_name, agent_name, agent_params):
        """
        Add an agent to the benchmarking suite.

        Args:
            environment_name (str): name of the environment for the experiment (E.g. Gym.Pendulum-v0);
            agent_name (str): name of the agent for the experiments;
            agent_params (list): dictionary containing the parameters for the agent builder.

        """
        assert environment_name in self._environment_dict
        if agent_name in self._experiment_structure[environment_name]:
            raise AttributeError(
                'An experiment for environment {} and builders {} already exists.'.format(environment_name, agent_name)
            )
        self._agent_list.append(agent_name)
        environment_builder_params = self._environment_dict[environment_name]['build_params']
        exp = self._create_experiment(environment_name, environment_builder_params, agent_name, agent_params)
        self._experiment_structure[environment_name][agent_name] = exp

    def run(self, exec_type='sequential'):
        """
        Run all experiments in the suite.

        """
        for environment, agents in self._experiment_structure.items():
            for agent, exp in agents.items():
                self.logger.info('Starting Experiment for {} on {}'.format(agent, environment))
                run_params = self._environment_dict[environment]['run_params']
                exp.run(exec_type=exec_type, parallel=self._parallel, slurm=self._slurm, **run_params)

    def print_experiments(self):
        """
        Print the experiments in the suite.

        """
        first = True
        for env, agents in self._experiment_structure.items():
            if not first:
                self.logger.weak_line()
            first = False
            self.logger.info('Environment: ' + env)
            for agent, _ in agents.items():
                self.logger.info('- ' + agent)

    def save_parameters(self):
        """
        Save the experiment parameters in yaml files inside the parameters folder

        """
        for env, params in self._parameters_dict.items():
            self.logger.save_params(env, params)

    def save_plots(self, **plot_params):
        """
        Save the result plots to the log directory.

        Args:
            **plot_params: parameters to be passed to the suite visualizer.

        """
        visualizer = BenchmarkSuiteVisualizer(self.logger, **plot_params)
        visualizer.save_reports()

    def show_plots(self, **plot_params):
        """
        Display the result plots.

        Args:
            **plot_params: parameters to be passed to the suite visualizer.

        """
        visualizer = BenchmarkSuiteVisualizer(self.logger, **plot_params)
        visualizer.show_report()

    def _create_experiment(self, environment, environment_params, agent_name, agent_builder_params):
        environment_name, environment_id = self._split_env_name(environment)
        environment_params = self._update_environment_params(environment_name, environment_id, environment_params)

        logger = BenchmarkLogger(
            log_dir=self.logger.get_path(),
            log_id='{}/{}'.format(environment_id, agent_name),
            use_timestamp=False
        )

        try:
            builder = getattr(mushroom_rl_benchmark.builders, '{}Builder'.format(agent_name))
        except AttributeError as e: 
            logger.exception(e)

        agent_builder, agent_params = builder.default(get_default_dict=True, **agent_builder_params)
        env_builder = EnvironmentBuilder(environment_name, environment_params)

        self._add_parameters(agent_name, environment_id, agent_params)

        return BenchmarkExperiment(agent_builder, env_builder, logger)

    def _split_env_name(self, environment):
        separator = '.'

        if separator in environment:
            environment_name, environment_id = environment.split(separator)
            return environment_name, environment_id
        else:
            return environment, environment

    def _update_environment_params(self, environment_name, environment_id, environment_params):
        if environment_params is None:
            environment_params = dict()

        if environment_name != environment_id:
            environment_params = dict(
                env_id=environment_id,
                **environment_params)

        return environment_params

    def _add_parameters(self, agent_name, environment_name, params):

        if environment_name not in self._parameters_dict:
            self._parameters_dict[environment_name] = dict()

        del params['cls']
        del params['use_cuda']
        self._parameters_dict[environment_name][agent_name] = params
