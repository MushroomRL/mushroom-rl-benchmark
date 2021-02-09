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
                self.has_entropy = 'E' in self.data
        else:
            self.has_entropy = has_entropy

    @property
    def is_data_persisted(self):
        """
        Check if data was passed as dictionary or should be read from log directory.

        """
        return self.data is None
    
    def get_J(self):
        """
        Get J from dictionary or log directory.

        """
        if self.is_data_persisted:
            return self.logger.load_J()
        else:
            return self.data['J']
    
    def get_R(self):
        """
        Get R from dictionary or log directory.

        """
        if self.is_data_persisted:
            return self.logger.load_R()
        else:
            return self.data['R']
    
    def get_V(self):
        """
        Get V from dictionary or log directory.

        """
        if self.is_data_persisted:
            return self.logger.load_V()
        else:
            return self.data['V']
    
    def get_entropy(self):
        """
        Get entropy from dictionary or log directory.

        """
        if self.is_data_persisted:
            return self.logger.load_entropy()
        else:
            return self.data['E']

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
        j_ax.grid()
        plot_mean_conf(self.get_J(), j_ax)

        r_ax = fig.add_subplot(r_pos,
            ylabel='R', 
            xlabel='epochs')
        r_ax.grid()
        plot_mean_conf(self.get_R(), r_ax)

        v_ax = fig.add_subplot(q_pos,
            ylabel='V', 
            xlabel='epochs')
        v_ax.grid()
        plot_mean_conf(self.get_V(), v_ax)

        if self.has_entropy:
            e_ax = fig.add_subplot(e_pos,
                ylabel='policy_entropy', 
                xlabel='epochs')
            e_ax.grid()
            plot_mean_conf(self.get_entropy(), e_ax)

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
    """
    Class to handle visualization of a benchmark suite.

    """
    plot_counter = 0

    def __init__(self, logger, color_cycle=None, y_limit=None, legend=None):
        """
        Constructor.

        Args:
            logger (BenchmarkLogger): logger to be used;
            color_cycle (dict, None): dictionary with colors to be used for each algorithm;
            y_limit (dict, None): dictionary with environment specific plot limits.
            legend (dict, None): dictionary with environment specific legend parameters.

        """
        self._logger = logger

        path = Path(self._logger.get_path())

        self._logger_dict = {}
        self._color_cycle = dict() if color_cycle is None else color_cycle
        self._y_limit = dict() if y_limit is None else y_limit
        self._legend_dict = dict() if legend is None else legend

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

    def _legend(self, ax, env, data_type):
        if env in self._legend_dict and data_type in self._legend_dict[env]:
            legend_dict = self._legend_dict[env][data_type]
        else:
            legend_dict = dict()

        fontsize = legend_dict.pop('fontsize', 'x-large')
        frameon = legend_dict.pop('frameon', False)
        loc = legend_dict.pop('loc', 'lower right')
        default_bbox = (0.7, 1.0) if data_type == 'entropy' else (0.7, 0.15)
        bbox_to_anchor = legend_dict.pop('bbox_to_anchor', default_bbox)
        ax.legend(fontsize=fontsize, ncol=len(self._logger_dict[env]) // 2, frameon=frameon,
                  loc=loc, bbox_to_anchor=bbox_to_anchor, **legend_dict)

    def get_report(self, env, data_type):
        """
        Create report plot with matplotlib.

        """
        self.plot_counter += 1

        plot_id = self.plot_counter * 1000
        fig = plt.figure(plot_id, figsize=(8, 6), dpi=80)
        ax = plt.axes()
        ax.set_xlabel('# Epochs', fontweight='bold')
        ax.set_ylabel(data_type, fontweight='bold', rotation=0 if len(data_type) == 1 else 90)

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize('x-large')

        max_epochs = 1
        for alg, logger in self._logger_dict[env].items():
            color = self._color_cycle[alg]
            data = getattr(logger, 'load_' + data_type)()

            if data is not None:
                plot_mean_conf(data, ax, color=color, label=alg)
                max_epochs = max(max_epochs, len(data[0]))

        if env in self._y_limit and data_type in self._y_limit[env]:
            ax.set_ylim(**self._y_limit[env][data_type])

        ax.set_xlim(xmin=0, xmax=max_epochs-1)
        ax.grid()
        self._legend(ax, env, data_type)
        fig.tight_layout()

        return fig

    def save_reports(self, as_pdf=True, transparent=True):
        """
        Method to save an image of a report of the training metrics from a performend experiment.

        Args:
            as_pdf (bool, True): whether to save the reports as pdf files or png;
            transparent (bool, True): If true, the figure background is transparent and not white;

        """
        for env in self._logger_dict.keys():
            for data_type in ['J', 'R', 'V', 'entropy']:
                fig = self.get_report(env, data_type)
                self._logger.save_figure(fig, data_type, env, as_pdf=as_pdf, transparent=transparent)
                plt.close(fig)

    def show_reports(self):
        """
        Method to show a report of the training metrics from a performend experiment.

        """
        matplotlib.use(default_backend)
        for env in self._logger_dict.keys():
            for data_type in ['J', 'R', 'V', 'entropy']:
                self.get_report(env, data_type)
        plt.show()
