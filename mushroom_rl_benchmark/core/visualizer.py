import numpy as np
import matplotlib
default_backend = matplotlib.rcParams['backend']
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from mushroom_rl_benchmark.utils import get_mean_and_confidence
from mushroom_rl.core import Core

from mushroom_rl_benchmark.core.logger import BenchmarkLogger

import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='scipy')


def plot_mean_conf(data, ax, color='blue', facecolor=None, alpha=0.4, label=None):
    """
    Method to plot mean and confidence interval for data on pyplot axes.

    """
    facecolor = color if facecolor is None else facecolor

    mean, conf = get_mean_and_confidence(np.array(data))
    upper_bound = mean + conf
    lower_bound = mean - conf

    ax.plot(mean, color=color, label=label)
    ax.fill_between(np.arange(np.size(mean)), upper_bound, lower_bound, facecolor=facecolor, alpha=alpha)


class BenchmarkVisualizer(object):
    """
    Class to handle all visualizations of the experiment.

    """
    plot_counter = 0

    def __init__(self, logger, data=None, has_entropy=None, id=1):
        """
        Constructor.

        Args:
            logger (BenchmarkLogger): logger to be used;
            data (dict, None): dictionary with data points for visualization;
            has_entropy (bool, None): select if entropy is available for the algorithm.

        """
        self.logger = logger
        self.data = data
        self.id = id

        if has_entropy is None:
            if self.is_data_persisted:
                self.has_entropy = self.logger.exists_policy_entropy()
            else:
                self.has_entropy = 'Es' in self.data
        else:
            self.has_entropy = has_entropy

    @property
    def is_data_persisted(self):
        """
        Check if data was passed as dictionary or should be read from log directory.

        """
        return self.data is None
    
    def get_Js(self):
        """
        Get Js from dictionary or log directory.

        """
        if self.is_data_persisted:
            return self.logger.load_Js()
        else:
            return self.data['Js']
    
    def get_Rs(self):
        """
        Get Rs from dictionary or log directory.

        """
        if self.is_data_persisted:
            return self.logger.load_Rs()
        else:
            return self.data['Rs']
    
    def get_Qs(self):
        """
        Get Qs from dictionary or log directory.

        """
        if self.is_data_persisted:
            return self.logger.load_Qs()
        else:
            return self.data['Qs']
    
    def get_Es(self):
        """
        Get Es from dictionary or log directory.

        """
        if self.is_data_persisted:
            return self.logger.load_policy_entropies()
        else:
            return self.data['Es']

    def get_report(self):
        """
        Create report plot with matplotlib.

        """

        plot_cnt = self.plot_counter
        self.plot_counter += 1

        j_pos, r_pos, q_pos, e_pos = 131, 132, 133, 144

        if self.has_entropy:
            j_pos += 10
            r_pos += 10
            q_pos += 10

        fig = plt.figure(plot_cnt * 10 + self.id, figsize=(24,6), dpi=80)
        j_ax = fig.add_subplot(j_pos, 
            ylabel='J', 
            xlabel='epochs')
        plot_mean_conf(self.get_Js(), j_ax)

        r_ax = fig.add_subplot(r_pos,
            ylabel='R', 
            xlabel='epochs')
        plot_mean_conf(self.get_Rs(), r_ax)

        q_ax = fig.add_subplot(q_pos,
            ylabel='V', 
            xlabel='epochs')
        plot_mean_conf(self.get_Qs(), q_ax)

        if self.has_entropy:
            e_ax = fig.add_subplot(e_pos,
                ylabel='policy_entropy', 
                xlabel='epochs')
            plot_mean_conf(self.get_Es(), e_ax)

        fig.tight_layout()

        return fig

    def save_report(self, file_name='report_plot'):
        """
        Method to save an image of a report of the training metrics from a performend experiment.

        """
        fig = self.get_report()
        self.logger.save_figure(fig, file_name)
        plt.close(fig)

    def show_report(self):
        """
        Method to show a report of the training metrics from a performend experiment.

        """
        matplotlib.use(default_backend)
        fig = self.get_report()
        plt.show()
        plt.close(fig)

    def show_agent(self, episodes=5, mdp_render=False):
        """
        Method to run and visualize the best builders in the environment.

        """
        matplotlib.use(default_backend)
        mdp = self.logger.load_environment_builder().build()
        if mdp_render:
            mdp.render()
        agent = self.logger.load_best_agent()
        core = Core(agent, mdp)
        core.evaluate(n_episodes=episodes, render=True)

    @classmethod
    def from_path(cls, path):
        """
        Method to create a BenchmarkVisualizer from a path.

        """
        path = Path(path)
        return cls(BenchmarkLogger(path.parent, path.name, False))


class BenchmarkSuiteVisualizer(object):
    plot_counter = 0

    def __init__(self, logger, color_cycle=None):
        self._logger = logger

        path = Path(self._logger.get_path())

        self._logger_dict = {}
        self._color_cycle = dict() if color_cycle is None else color_cycle

        alg_count = 0
        for env_dir in path.iterdir():
            if env_dir.is_dir() and env_dir.name != 'plots':
                env = env_dir.name
                self._logger_dict[env] = dict()

                for alg_dir in env_dir.iterdir():
                    if alg_dir.is_dir():
                        alg = alg_dir.name

                        if alg not in self._color_cycle:
                            self._color_cycle[alg] = 'C' + str(alg_count)

                        alg_logger = BenchmarkLogger.from_path(alg_dir)
                        self._logger_dict[env][alg] = alg_logger
                        alg_count += 1

    def get_report(self, env, data_type):
        """
        Create report plot with matplotlib.

        """
        self.plot_counter += 1

        plot_id = self.plot_counter * 1000
        fig = plt.figure(plot_id, figsize=(8, 6), dpi=80)
        ax = plt.axes(ylabel=data_type, xlabel='# Epochs')

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize('x-large')

        for alg, logger in self._logger_dict[env].items():
            color = self._color_cycle[alg]
            data = getattr(logger, 'load_' + data_type + 's')()

            plot_mean_conf(data, ax, color=color, label=alg)

            # if logger.exists_policy_entropy():
            #     plot_mean_conf(logger.load_policy_entropies(), ax, color=color, label=alg)

        ax.grid()
        ax.legend(fontsize='medium', ncol=len(self._logger_dict[env]), frameon=False,
                  loc='upper center', bbox_to_anchor=(0.5, 0.05))
        fig.tight_layout()

        return fig

    def save_reports(self):
        """
        Method to save an image of a report of the training metrics from a performend experiment.

        """
        for env in self._logger_dict.keys():
            for data_type in ['J', 'R', 'Q']:
                fig = self.get_report(env, data_type)
                self._logger.save_figure(fig, env + '_' + data_type)
                plt.close(fig)

    def show_reports(self):
        """
        Method to show a report of the training metrics from a performend experiment.

        """
        matplotlib.use(default_backend)
        for env in self._logger_dict.keys():
            for data_type in ['J', 'R', 'Q']:
                self.get_report(env, data_type)
        plt.show()