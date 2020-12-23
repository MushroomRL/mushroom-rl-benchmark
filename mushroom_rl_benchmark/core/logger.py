import os
import logging
import json
import pickle
from datetime import datetime

import torch
import numpy as np
from pathlib import Path
from mushroom_rl.algorithms import Agent


class BenchmarkLogger:
    """
    Class to handle all interactions with the log directory.
    """

    file_Js = 'Js.pkl'
    file_Rs = 'Rs.pkl'
    file_Qs = 'Qs.pkl'
    file_policy_entropies = 'policy_entropies.pkl'
    best_agent_dir = 'best_agent.msh'
    last_agent_dir = 'last_agent.msh'
    file_env_builder = 'environment_builder.pkl'
    file_agent_builder = 'agent_builder.pkl'
    file_config = 'config.json'
    file_stats = 'stats.json'

    def __init__(self, log_dir=None, log_id=None, use_timestamp=True):
        """
        Constructor.

        Args:
            log_dir (str, None): path to the log directory, if not specified defaults to ./logs or to
                /work/scratch/$USER if the second directory exists;
            log_id (str, None): log id, if not specified defaults to: benchmark[_YY-mm-ddTHH:MM:SS.zzz]);
            use_timestamp (bool, True): select if a timestamp should be appended to the log id.

        """

        self.log_dir = ''
        self.log_id = ''
        self.log = None
        
        # Set and create log directories
        self.set_log_dir(log_dir)
        self.set_log_id(log_id, use_timestamp=use_timestamp)
        
        # Get get logger for benchmark
        self.log = logging.getLogger(log_id) # (self.get_log_id())
        self.log.setLevel(logging.DEBUG)
        
        # Create handlers for console and file
        fh = logging.FileHandler(self.get_path('console.log'))
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Add formatter to handlers 
        formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # add the handlers to logger
        self.log.addHandler(ch)
        self.log.addHandler(fh)

    def set_log_dir(self, log_dir):
        if log_dir is None:
            default_dir = './logs'
            scratch_dir = os.path.join('/work', 'scratch', os.getenv('USER'))
            if Path(scratch_dir).is_dir():
                log_dir = os.path.join(scratch_dir, 'logs')
            else:
                log_dir = default_dir
        if not os.path.exists(log_dir):
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        if not os.path.isdir(log_dir):
            raise NotADirectoryError("Path to save builders is not valid")
        self.log_dir = log_dir

    def get_log_dir(self):
        return self.log_dir

    def set_log_id(self, log_id, use_timestamp=True):
        if log_id is None:
            log_id = 'benchmark'
        if use_timestamp:
            log_id += '_{}'.format(datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%z"))
        path = os.path.join(self.log_dir, log_id, '')
        if not os.path.exists(path):
            Path(path).mkdir(parents=True, exist_ok=True)
        if not os.path.isdir(path):
            raise NotADirectoryError("Path to save builders is not valid")
        self.log_id = log_id

    def get_log_id(self):
        return self.log_id

    def get_path(self, filename=''):
        return os.path.join(self.log_dir, self.log_id, filename)

    def info(self, message):
        """
        Log info message.

        Args:
            message (str): message string

        """
        self.log.info(message)
    
    def warning(self, message):
        """
        Log warning message.

        Args:
            message (str): message string

        """
        self.log.warning(message)
    
    def exception(self, message):
        """
        Log exception message.

        Args:
            message (str): message string

        """
        self.log.exception(message)
    
    def critical(self, message):
        """
        Log critical message.

        Args:
            message (str): message string

        """
        self.log.critical(message)

    def save_Js(self, Js):
        self._save_pickle(self.get_path(self.file_Js), Js)

    def load_Js(self):
        return self._load_pickle(self.get_path(self.file_Js))

    def save_Rs(self, Rs):
        self._save_pickle(self.get_path(self.file_Rs), Rs)

    def load_Rs(self):
        return self._load_pickle(self.get_path(self.file_Rs))

    def save_Qs(self, Qs):
        self._save_pickle(self.get_path(self.file_Qs), Qs)

    def load_Qs(self):
        return self._load_pickle(self.get_path(self.file_Qs))

    def save_policy_entropies(self, policy_entropies):
        self._save_pickle(self.get_path(self.file_policy_entropies), policy_entropies)

    def load_policy_entropies(self):
        return self._load_pickle(self.get_path(self.file_policy_entropies))

    def exists_policy_entropy(self):
        return Path(self.get_path(self.file_policy_entropies)).exists()

    def save_best_agent(self, agent):
        self._save_agent(self.get_path(self.best_agent_dir), agent)

    def save_last_agent(self, agent):
        self._save_agent(self.get_path(self.last_agent_dir), agent)
        
    def _save_agent(self, path, agent):
        agent.save(path)

    def load_best_agent(self):
        return self._load_agent(self.get_path(self.best_agent_dir))

    def load_last_agent(self):
        return self._load_agent(self.get_path(self.last_agent_dir))
        
    def _load_agent(self, path):
        return Agent.load(path)

    def save_environment_builder(self, env_builder):
        self._save_pickle(self.get_path(self.file_env_builder), env_builder)

    def load_environment_builder(self):
        return self._load_pickle(self.get_path(self.file_env_builder))

    def save_agent_builder(self, agent_builder):
        self._save_pickle(self.get_path(self.file_agent_builder), agent_builder)

    def load_agent_builder(self):
        return self._load_pickle(self.get_path(self.file_agent_builder))

    def save_config(self, config):
        self._save_json(self.get_path(self.file_config), config)

    def load_config(self):
        return self._load_json(self.get_path(self.file_config))

    def save_stats(self, stats):
        self._save_json(self.get_path(self.file_stats), stats)

    def load_stats(self):
        return self._load_json(self.get_path(self.file_stats))

    def save_figure(self, figure, figname):
        figure.savefig(self.get_path(figname) + ".png")

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
    def _save_json(path, obj):
        with Path(path).open('w') as f:
            json.dump(obj, f, indent=2)

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
    def _load_json(path):
        with Path(path).open('r') as f:
            return json.load(f)

    @classmethod
    def from_path(cls, path):
        """
        Method to create a BenchmarkLogger from a path.

        """
        path = Path(path)
        return cls(path.parent, path.name, False)
