import mushroom_rl_benchmark.builders
from mushroom_rl_benchmark.builders import EnvironmentBuilder
from mushroom_rl_benchmark.core.experiment import BenchmarkExperiment
from mushroom_rl_benchmark.core.logger import BenchmarkLogger


class BenchmarkSuite:
    """
    Class to orchestrate the execution of multiple experiments.

    """
    def __init__(self, log_dir=None, log_id=None, use_timestamp=True, **run_params):
        """
        Constructor.

        Args:
            log_dir (str): path to the log directory (Default: ./logs or /work/scratch/$USER)
            log_id (str): log id (Default: benchmark[_YY-mm-ddTHH:MM:SS.zzz])
            use_timestamp (bool): select if a timestamp should be appended to the log id
            **run_params: parameters that are passed to the run method of the experiment
        
        """
        self.experiment_structure = dict()
        self.environment_list = []
        self.agent_list = []
        self.run_params = run_params
        self.logger = BenchmarkLogger(log_dir=log_dir, log_id=log_id, use_timestamp=use_timestamp)

    def add_experiment(self, environment_name, environment_builder_params, agent_name, agent_builder_params):
        """
        Add an experiment to the suite.

        Args:
            environment_name (str): name of the environment for the experiment (E.g. Gym.Pendulum-v0)
            environment_builder_params (dict): parameters for the environment builder
            agent_name (str): name of the agent for the experiment
            agent_builder_params (dict): parameters for the agent builder

        """
        if environment_name in self.environment_list:
            if agent_name in self.experiment_structure[environment_name]:
                raise AttributeError('An experiment for environment {} and builders {} already exists.'.format(environment_name, agent_name))
            else:
                self.experiment_structure[environment_name][agent_name] = self._create_experiment(environment_name, environment_builder_params, agent_name, agent_builder_params)
        else:
            self.environment_list.append(environment_name)
            self.experiment_structure[environment_name] = {agent_name: self._create_experiment(environment_name, environment_builder_params, agent_name, agent_builder_params)}
            
        if agent_name not in self.agent_list:
            self.agent_list.append(agent_name)

    def _create_experiment(self, environment, environment_params, agent_name, agent_builder_params):
        separator = '.'
        if environment_params is None:
            environment_params = dict()
        if separator in environment:
            environment_name, environment_id = environment.split(separator)
            environment_params = dict(
                env_id=environment_id,
                **environment_params)
            environment = environment.replace(separator, '_')
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
        for env, agents in self.experiment_structure.items():
            for agent, _ in agents.items():
                self.logger.info('Environment: {}\tAgent: {}'.format(env, agent))

    def run(self, exec_type='sequential'):
        """
        Run all experiments in the suite.

        """
        for environment, agents in self.experiment_structure.items():
            for agent, exp in agents.items():
                self.logger.info('Starting Experiment for {} on {}'.format(agent, environment))
                exp.run(exec_type=exec_type, **self.run_params)
