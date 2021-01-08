import mushroom_rl_benchmark.builders
from mushroom_rl_benchmark.builders import EnvironmentBuilder
from mushroom_rl_benchmark.core.experiment import BenchmarkExperiment
from mushroom_rl_benchmark.core.logger import BenchmarkLogger
from mushroom_rl_benchmark.core.visualizer import BenchmarkSuiteVisualizer


class BenchmarkSuite:
    """
    Class to orchestrate the execution of multiple experiments.

    """
    def __init__(self, log_dir=None, log_id=None, use_timestamp=True, **suite_params):
        """
        Constructor.

        Args:
            log_dir (str): path to the log directory (Default: ./logs or /work/scratch/$USER)
            log_id (str): log id (Default: benchmark[_YYYY-mm-dd-HH-MM-SS])
            use_timestamp (bool): select if a timestamp should be appended to the log id
            **suite_params: parameters that are passed to the run method of the experiment
        
        """
        self._experiment_structure = dict()
        self._environment_dict = dict()
        self._agent_list = []
        self._suite_params = suite_params
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
        self._experiment_structure[environment_name][agent_name] = \
            self._create_experiment(environment_name, environment_builder_params, agent_name, agent_params)

    def _create_experiment(self, environment, environment_params, agent_name, agent_builder_params):
        separator = '.'
        if environment_params is None:
            environment_params = dict()
        if separator in environment:
            environment_name, environment_id = environment.split(separator)
            environment_params = dict(
                env_id=environment_id,
                **environment_params)
            environment = environment_id
        else:
            environment_name = environment

        logger = BenchmarkLogger(
            log_dir=self.logger.get_path(), 
            log_id='{}/{}'.format(environment, agent_name),
            use_timestamp=False
        )

        try:
            builder = getattr(mushroom_rl_benchmark.builders, '{}Builder'.format(agent_name))
        except AttributeError as e: 
            logger.exception(e)

        agent_builder = builder.default(**agent_builder_params)
        env_builder = EnvironmentBuilder(environment_name, environment_params)

        exp = BenchmarkExperiment(agent_builder, env_builder, logger)

        return exp

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

    def run(self, exec_type='sequential'):
        """
        Run all experiments in the suite.

        """
        for environment, agents in self._experiment_structure.items():
            for agent, exp in agents.items():
                self.logger.info('Starting Experiment for {} on {}'.format(agent, environment))
                run_params = self._environment_dict[environment]['run_params']
                exp.run(exec_type=exec_type, **self._suite_params, **run_params)

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
