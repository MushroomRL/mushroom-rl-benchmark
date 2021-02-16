import os
import pickle
import yaml
from datetime import datetime

import torch
import numpy as np
from pathlib import Path

from mushroom_rl.core import Serializable
from mushroom_rl.core.logger import ConsoleLogger

from mushroom_rl_benchmark.utils import dictionary_to_primitive


class BenchmarkLogger(ConsoleLogger):
    """
    Class to handle all interactions with the log directory.
    """

    def __init__(self, log_dir=None, log_id=None, use_timestamp=True):
        """
        Constructor.

        Args:
            log_dir (str, None): path to the log directory, if not specified defaults to ./logs or to
                /work/scratch/$USER if the second directory exists;
            log_id (str, None): log id, if not specified defaults to: benchmark[_YY-mm-ddTHH:MM:SS.zzz]);
            use_timestamp (bool, True): select if a timestamp should be appended to the log id.

        """
        self._file_J = 'J.pkl'
        self._file_R = 'R.pkl'
        self._file_V = 'V.pkl'
        self._file_entropy = 'entropy.pkl'
        self._file_best_agent = 'best_agent.msh'
        self._file_last_agent = 'last_agent.msh'
        self._file_env_builder = 'environment_builder.pkl'
        self._file_agent_builder = 'agent_builder.pkl'
        self._file_config = 'config.yaml'
        self._file_stats = 'stats.yaml'

        self._log_dir = ''
        self._log_id = ''
        
        # Set and create log directories
        self.set_log_dir(log_dir)
        self.set_log_id(log_id, use_timestamp=use_timestamp)

        super().__init__(self._log_id, Path(self.get_path()), log_file_name='console')

    def set_log_dir(self, log_dir):
        if log_dir is None:
            default_dir = Path('logs')
            scratch_dir = Path('/work', 'scratch', os.getenv('USER'))
            if scratch_dir.is_dir():
                log_dir = scratch_dir / 'logs'
            else:
                log_dir = default_dir
        else:
            log_dir = Path(log_dir)

        if not log_dir.exists():
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        if not log_dir.is_dir():
            raise NotADirectoryError("Path to save builders is not valid")
        
        self._log_dir = log_dir

    def get_log_dir(self):
        return str(self._log_dir)

    def set_log_id(self, log_id, use_timestamp=True):
        if log_id is None:
            log_id = 'benchmark'
        if use_timestamp:
            log_id += '_{}'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        path = self._log_dir / log_id
        if not path.exists():
            Path(path).mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise NotADirectoryError("Path to save builders is not valid")
        self._log_id = log_id

    def get_log_id(self):
        return self._log_id

    def get_path(self, filename=''):
        return self._log_dir / self._log_id / filename

    def get_params_path(self, filename=''):
        params_dir = self._log_dir / self._log_id / 'params'

        if not params_dir.exists():
            params_dir.mkdir(parents=True, exist_ok=True)

        return params_dir / filename

    def get_figure_path(self, filename='', subfolder=None):
        figure_dir = Path(self._log_dir) / self._log_id / 'plots'
        if subfolder is not None:
            figure_dir = figure_dir / subfolder
        if not figure_dir.exists():
            figure_dir.mkdir(parents=True, exist_ok=True)

        return str(figure_dir / filename)

    def save_J(self, J):
        self._save_pickle(self.get_path(self._file_J), J)

    def load_J(self):
        return self._load_pickle(self.get_path(self._file_J))

    def save_R(self, R):
        self._save_pickle(self.get_path(self._file_R), R)

    def load_R(self):
        return self._load_pickle(self.get_path(self._file_R))

    def save_V(self, V):
        self._save_pickle(self.get_path(self._file_V), V)

    def load_V(self):
        return self._load_pickle(self.get_path(self._file_V))

    def save_entropy(self, entropy):
        self._save_pickle(self.get_path(self._file_entropy), entropy)

    def load_entropy(self):
        path = self.get_path(self._file_entropy)
        if path.exists():
            return self._load_pickle(path)
        else:
            return None

    def exists_policy_entropy(self):
        return Path(self.get_path(self._file_entropy)).exists()

    def save_best_agent(self, agent):
        agent.save(self.get_path(self._file_best_agent))

    def save_last_agent(self, agent):
        agent.save(self.get_path(self._file_last_agent))

    def exists_best_agent(self):
        return Path(self.get_path(self._file_best_agent)).exists()

    def load_best_agent(self):
        return Serializable.load(self.get_path(self._file_best_agent))

    def load_last_agent(self):
        return Serializable.load(self.get_path(self._file_last_agent))

    def save_environment_builder(self, env_builder):
        self._save_pickle(self.get_path(self._file_env_builder), env_builder)

    def load_environment_builder(self):
        return self._load_pickle(self.get_path(self._file_env_builder))

    def save_agent_builder(self, agent_builder):
        self._save_pickle(self.get_path(self._file_agent_builder), agent_builder)

    def load_agent_builder(self):
        return self._load_pickle(self.get_path(self._file_agent_builder))

    def save_config(self, config):
        self._save_yaml(self.get_path(self._file_config), config)

    def load_config(self):
        return self._load_yaml(self.get_path(self._file_config))

    def exists_stats(self):
        return Path(self.get_path(self._file_stats)).exists()

    def save_stats(self, stats):
        self._save_yaml(self.get_path(self._file_stats), stats)

    def load_stats(self):
        return self._load_yaml(self.get_path(self._file_stats))

    def save_params(self, env, params):
        file_name = env + '.yaml'
        primitive_params = dictionary_to_primitive(params)
        self._save_yaml(self.get_params_path(file_name), primitive_params)

    def save_figure(self, figure, figname, subfolder=None, as_pdf=False, transparent=True):
        extension = '.pdf' if as_pdf else '.png'
        figure.savefig(self.get_figure_path(figname + extension, subfolder), transparent=transparent)

    @staticmethod
    def _save_pickle(path, obj):
        with Path(path).open('wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def _save_numpy(path, obj):
        with Path(path).open('wb') as f:
            np.save(f, obj)
    
    @staticmethod
    def _save_torch(path, obj):
        torch.save(obj, path)
    
    @staticmethod
    def _save_yaml(path, obj):
        with Path(path).open('w') as f:
            yaml.dump(obj, f, version=(1, 2), default_flow_style=False)

    @staticmethod
    def _load_pickle(path):
        with Path(path).open('rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def _load_numpy(path):
        with Path(path).open('rb') as f:
            return np.load(f)
    
    @staticmethod
    def _load_torch(path):
        return torch.load(path)
    
    @staticmethod
    def _load_yaml(path):
        with Path(path).open('r') as f:
            return yaml.load(f)

    @classmethod
    def from_path(cls, path):
        """
        Method to create a BenchmarkLogger from a path.

        """
        path = Path(path)
        return cls(path.parent, path.name, False)
