import yaml
from inspect import signature
from pathlib import Path
from collections import OrderedDict

import mushroom_rl_benchmark.builders
from mushroom_rl_benchmark.utils import dictionary_to_primitive


class BenchmarkParams:
    def __init__(self):
        self._params_dict = OrderedDict()

    def add_experiment_params(self, env_id, env_name, env_params, agent_name, agent_params, run_params):
        self._add_env(env_id, env_name, env_params, run_params)

        self._add_agent(env_id, agent_name, agent_params)

    def save_params(self, base_path):
        save_path = Path(base_path) / 'params'
        save_path.mkdir(parents=True, exist_ok=True)

        for env, params in self._params_dict.items():
            file_name = env + '.yaml'
            primitive_params = dictionary_to_primitive(params)
            self._save_yaml(save_path / file_name, primitive_params)

    def _add_env(self, env_id, env_name, env_params, run_params):
        if env_id not in self._params_dict:
            self._params_dict[env_id] = dict()
            self._params_dict[env_id]['run_params'] = run_params
            self._params_dict[env_id]['env_params'] = dict(name=env_name, params=env_params)
            self._params_dict[env_id]['agent_params'] = dict()

    def _add_agent(self, env_id, agent_name, agent_params):
        agent_builder_factory = getattr(mushroom_rl_benchmark.builders, f'{agent_name}Builder')
        default_param_dict = self._extract_default_params(agent_builder_factory.default)
        default_param_dict.update(agent_params)

        self._params_dict[env_id]['agent_params'][agent_name] = default_param_dict

    @staticmethod
    def _extract_default_params(method):
        parameters = signature(method).parameters

        default_param_dict = {p.name: p.default for p in parameters.values()}

        return default_param_dict

    @staticmethod
    def _save_yaml(path, obj):
        with path.open('w') as f:
            yaml.dump(obj, f, version=(1, 2), default_flow_style=False, sort_keys=False)
