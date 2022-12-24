import yaml
from pathlib import Path


class BenchmarkConfiguration:
    def __init__(self, config_path):
        self._config_path = Path(config_path)

        with open(self._config_path / 'suite.yaml', 'r') as param_file:
            self._suite_params = yaml.safe_load(param_file)['suite_params']

            if 'quiet' in self._suite_params:
                self._quiet = self._suite_params['quiet']
                del self._suite_params['quiet']
            else:
                self._quiet = True

        self._env_params = dict()
        env_cfg_dir = self._config_path / 'env'
        for env_config_path in env_cfg_dir.iterdir():
            if env_config_path.suffix == '.yaml':
                env_name = env_config_path.stem
                with open(env_config_path, 'r') as config_file:
                    yaml_file = yaml.safe_load(config_file)
                    self._env_params[env_name] = yaml_file

    @property
    def quiet(self):
        return self._quiet

    @property
    def suite_params(self):
        return self._suite_params

    @property
    def envs(self):
        return self._env_params.keys()

    def get_available_agents(self, env):
        return self._env_params[env]['agent_params'].keys()

    def get_experiment_params(self, env, agent):
        env_config = self._env_params[env]

        return env_config['env_params'], env_config['run_params'], env_config['agent_params'][agent]


