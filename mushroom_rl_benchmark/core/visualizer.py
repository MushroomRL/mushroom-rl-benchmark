import numpy as np
import matplotlib
default_backend = matplotlib.rcParams['backend']
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from mushroom_rl_benchmark.utils import get_mean_and_confidence
from mushroom_rl.core import Core

from mushroom_rl_benchmark.core.logger import BenchmarkLogger


class BenchmarkVisualizer:

    plot_counter = 0

    def __init__(self, logger, data=None, has_entropy=None, id=1):
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
        return self.data is None
    
    def get_Js(self):
        if self.is_data_persisted:
            return self.logger.load_Js()
        else:
            return self.data['Js']
    
    def get_Rs(self):
        if self.is_data_persisted:
            return self.logger.load_Rs()
        else:
            return self.data['Rs']
    
    def get_Qs(self):
        if self.is_data_persisted:
            return self.logger.load_Qs()
        else:
            return self.data['Qs']
    
    def get_Es(self):
        if self.is_data_persisted:
            return self.logger.load_policy_entropies()
        else:
            return self.data['Es']

    def get_report(self, color='red', facecolor='blue', alpha=0.4, grid=True):

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
        self.plot_mean_conf(self.get_Js(), j_ax, color=color, facecolor=facecolor, alpha=alpha, grid=grid)

        r_ax = fig.add_subplot(r_pos,
            ylabel='R', 
            xlabel='epochs')
        self.plot_mean_conf(self.get_Rs(), r_ax, color=color, facecolor=facecolor, alpha=alpha, grid=grid)

        q_ax = fig.add_subplot(q_pos,
            ylabel='V', 
            xlabel='epochs')
        self.plot_mean_conf(self.get_Qs(), q_ax, color=color, facecolor=facecolor, alpha=alpha, grid=grid)

        if self.has_entropy:
            e_ax = fig.add_subplot(e_pos,
                ylabel='policy_entropy', 
                xlabel='epochs')
            self.plot_mean_conf(self.get_Es(), e_ax, color=color, facecolor=facecolor, alpha=alpha, grid=grid)

        fig.tight_layout()

        return fig

    def save_report(self, file_name='report_plot', color='red', facecolor='blue', alpha=0.4, grid=True):
        """
        Method to save an image of a report of the training metrics from a performend experiment.
        """
        fig = self.get_report(color=color, facecolor=facecolor, alpha=alpha, grid=grid)
        self.logger.save_figure(fig, file_name)
        plt.close(fig)
    
    def show_report(self, color='red', facecolor='blue', alpha=0.4, grid=True):
        """
        Method to show a report of the training metrics from a performend experiment.
        """
        matplotlib.use(default_backend)
        fig = self.get_report(color=color, facecolor=facecolor, alpha=alpha, grid=grid)
        plt.show()
        plt.close(fig)

    @staticmethod
    def plot_mean_conf(data, ax, color='red', facecolor='blue', alpha=0.4, grid=True):
        """
        Method to plot mean and confidence interval for data on pyplot axes.
        """
        mean, conf = get_mean_and_confidence(np.array(data))
        upper_bound = mean + conf
        lower_bound = mean - conf

        if grid:
            ax.grid()
        ax.plot(mean, color=color)
        ax.fill_between(np.arange(np.size(mean)), upper_bound, lower_bound, facecolor=facecolor, alpha=alpha)

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