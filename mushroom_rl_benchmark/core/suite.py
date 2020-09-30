import mushroom_rl_benchmark.builders
from mushroom_rl_benchmark.builders import EnvironmentBuilder
from mushroom_rl_benchmark.core.experiment import BenchmarkExperiment
from mushroom_rl_benchmark.core.logger import BenchmarkLogger


class BenchmarkSuite:
    def __init__(self, log_dir=None, log_id=None, use_timestamp=True, **run_params):
        self.experiment_structure = dict()
        self.environment_list = []
        self.agent_list = []
        self.run_params = run_params
        self.logger = BenchmarkLogger(log_dir=log_dir, log_id=log_id, use_timestamp=use_timestamp)

    def add_experiment(self, environment, environment_params, agent_name, agent_builder_params):

        if environment in self.environment_list:
            if agent_name in self.experiment_structure[environment]:
                raise AttributeError('An experiment for environment {} and builders {} already exists.'.format(environment, agent_name))
            else:
                self.experiment_structure[environment][agent_name] = self._create_experiment(environment, environment_params, agent_name, agent_builder_params)
        else:
            self.environment_list.append(environment)
            self.experiment_structure[environment] = {agent_name: self._create_experiment(environment, environment_params, agent_name, agent_builder_params)}
            
        if agent_name not in self.agent_list:
            self.agent_list.append(agent_name)

    def _create_experiment(self, environment, environment_params, agent_name, agent_builder_params):
        separator = '.'
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
        for env, agents in self.experiment_structure.items():
            for agent, _ in agents.items():
                self.logger.info('Environment: {}\tAgent: {}'.format(env, agent))

    def run(self, exec_type='sequential'):
        for environment, agents in self.experiment_structure.items():
            for agent, exp in agents.items():
                self.logger.info('Starting Experiment for {} on {}'.format(agent, environment))
                exp.run(exec_type=exec_type, **self.run_params)
