import yaml
from pathlib import Path


class BenchmarkConfiguration:
    def __init__(self, config_path):
        self._config_path = Path(config_path)

        self._env_params = dict()
        env_cfg_dir = self._config_path / 'env'
        for env_config_path in sorted(env_cfg_dir.iterdir()):
            if env_config_path.suffix == '.yaml':
                env_name = env_config_path.stem
                with open(env_config_path, 'r') as config_file:
                    yaml_file = yaml.safe_load(config_file)
                    self._env_params[env_name] = yaml_file

    @property
    def envs(self):
        return self._env_params.keys()

    def get_available_agents(self, env):
        return self._env_params[env]['agent_params'].keys()

    def get_environment_id(self, env):
        return self._env_params[env].get('id', env)

    def get_experiment_params(self, env, agent):
        env_config = self._env_params[env]

        return env_config['env_params'], env_config['run_params'], env_config['agent_params'][agent]

    def select(self, environments, algorithms):
        environments = tuple(self.envs) if 'all' in environments else tuple(environments)
        unknown = [environment for environment in environments if environment not in self.envs]
        if unknown:
            available = ', '.join(self.envs)
            raise ValueError(f'Unknown environment {unknown[0]}. Available environments: {available}')

        selected_experiments = list()
        for environment in environments:
            available = tuple(self.get_available_agents(environment))
            selected = available if 'all' in algorithms else tuple(algorithms)
            invalid = [algorithm for algorithm in selected if algorithm not in available]
            if invalid:
                choices = ', '.join(available)
                raise ValueError(f'{invalid[0]} is not configured for {environment}. '
                                 f'Available algorithms: {choices}')
            selected_experiments.extend((environment, algorithm) for algorithm in selected)

        return tuple(selected_experiments)
