import matplotlib
default_backend = matplotlib.rcParams['backend']
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from mushroom_rl.core import Core

from mushroom_rl_benchmark.utils import plot_mean_conf
from mushroom_rl_benchmark.core.logger import BenchmarkLogger

import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='scipy')


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
