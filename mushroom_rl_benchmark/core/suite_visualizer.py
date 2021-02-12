import matplotlib
default_backend = matplotlib.rcParams['backend']
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from mushroom_rl_benchmark.utils import plot_mean_conf

from mushroom_rl_benchmark.core.logger import BenchmarkLogger

import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='scipy')


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
        loc = legend_dict.pop('loc', 'center')
        default_bbox = (0.5, 0.9) if data_type == 'entropy' else (0.5, 0.1)
        bbox_to_anchor = legend_dict.pop('bbox_to_anchor', default_bbox)
        ax.legend(fontsize=fontsize, ncol=len(self._logger_dict[env]) // 2, frameon=frameon,
                  loc=loc, bbox_to_anchor=bbox_to_anchor, **legend_dict)

    def get_report(self, env, data_type):
        """
        Create report plot with matplotlib.

        """

        if data_type == 'entropy':
            has_entropy = False
            for logger in self._logger_dict[env].values():
                if logger.exists_policy_entropy():
                    has_entropy = True
                    break
            if not has_entropy:
                return None

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

                if fig is not None:
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
