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

            if 'show_progress_bar' in self._suite_params:
                self._show_progress_bar = self._suite_params['show_progress_bar']
                del self._suite_params['show_progress_bar']
            else:
                self._show_progress_bar = True

        self._env_params = dict()
        env_cfg_dir = self._config_path / 'env'
        for env_config_path in env_cfg_dir.iterdir():
            if env_config_path.suffix == '.yaml':
                env_name = env_config_path.stem
                with open(env_config_path, 'r') as config_file:
                    yaml_file = yaml.safe_load(config_file)
                    self._env_params[env_name] = yaml_file

        self._sweep_params = dict()
        sweep_cfg_dir = self._config_path / 'sweep'
        for sweep_config_path in sweep_cfg_dir.iterdir():
            if sweep_config_path.suffix == '.yaml':
                sweep_name = sweep_config_path.stem
                with open(sweep_config_path, 'r') as sweep_file:
                    yaml_file = yaml.safe_load(sweep_file)
                    self._sweep_params[sweep_name] = yaml_file

    @property
    def quiet(self):
        return self._quiet

    @property
    def show_progress_bar(self):
        return self._show_progress_bar

    @property
    def suite_params(self):
        return self._suite_params

    @property
    def envs(self):
        return self._env_params.keys()

    def get_available_agents(self, env):
        return self._env_params[env]['agent_params'].keys()

    def get_available_sweeps(self):
        return self._sweep_params.keys()

    def get_experiment_params(self, env, agent):
        env_config = self._env_params[env]

        return env_config['env_params'], env_config['run_params'], env_config['agent_params'][agent]

    def get_sweep_params(self, sweep, alg):
        algs_sweep_params = self._sweep_params[sweep]

        if alg in algs_sweep_params:
            return algs_sweep_params[alg]
        else:
            return dict()
