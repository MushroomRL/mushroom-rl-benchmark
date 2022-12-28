import matplotlib
default_backend = matplotlib.get_backend()
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from itertools import cycle

from mushroom_rl.core import Logger
from mushroom_rl.utils.plot import plot_mean_conf

from mushroom_rl_benchmark.core import BenchmarkDataLoader
import mushroom_rl_benchmark.utils.metrics as metrics

import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='scipy')


class BenchmarkSuiteVisualizer(object):
    """
    Class to handle visualization of a benchmark suite.

    """
    plot_counter = 0

    def __init__(self, path, is_sweep, color_cycle=None, y_limit=None, legend=None):
        """
        Constructor.

        Args:
            path (str, Path): path to the directory to visualize;
            is_sweep (bool): whether the benchmark is a parameter sweep;
            color_cycle (dict, None): dictionary with colors to be used for each algorithm;
            y_limit (dict, None): dictionary with environment specific plot limits.
            legend (dict, None): dictionary with environment specific legend parameters.

        """
        assert is_sweep is not None
        self._logger = Logger('plots', path)
        self._is_sweep = is_sweep

        path = self._logger.path.parent

        self._loader_dict = {}
        self._color_cycle = dict() if color_cycle is None else color_cycle
        self._line_cycle = dict()
        self._lines = ["-", "--", "-.", ":"]
        self._y_limit = dict() if y_limit is None else y_limit
        self._legend_dict = dict() if legend is None else legend

        if is_sweep:
            self._load_sweep(path)
        else:
            self._load_benchmark(path)

    def _load_benchmark(self, path):
        alg_count = 0
        for env_dir in path.iterdir():
            if env_dir.is_dir() and env_dir.name not in ['plots', 'params']:
                env = env_dir.name
                self._loader_dict[env] = dict()

                for alg_dir in env_dir.iterdir():
                    if alg_dir.is_dir():
                        alg = alg_dir.name

                        if alg not in self._color_cycle:
                            self._color_cycle[alg] = 'C' + str(alg_count)

                        alg_loader = BenchmarkDataLoader(alg_dir)
                        self._loader_dict[env][alg] = alg_loader
                        alg_count += 1

    def _load_sweep(self, path):
        alg_count = 0
        for env_dir in path.iterdir():
            if env_dir.is_dir() and env_dir.name not in ['plots', 'params']:
                env = env_dir.name
                self._loader_dict[env] = dict()

                for alg_dir in env_dir.iterdir():
                    if alg_dir.is_dir():
                        alg = alg_dir.name

                        line_cycler = cycle(self._lines)
                        for sweep_dir in alg_dir.iterdir():
                            if sweep_dir.is_dir():
                                sweep_name = alg + '_' + sweep_dir.name

                                if sweep_name not in self._color_cycle:
                                    self._color_cycle[sweep_name] = 'C' + str(alg_count)
                                    self._line_cycle[sweep_name] = next(line_cycler)

                                sweep_logger = BenchmarkDataLoader(sweep_dir)
                                self._loader_dict[env][sweep_name] = sweep_logger
                        alg_count += 1

    def _legend(self, ax, env, data_type):
        if env in self._legend_dict and data_type in self._legend_dict[env]:
            legend_dict = self._legend_dict[env][data_type]
        else:
            legend_dict = dict()

        fontsize = legend_dict.pop('fontsize', 'x-large')
        frameon = legend_dict.pop('frameon', False)
        loc = legend_dict.pop('loc', 'center')
        default_bbox = (0.5, -0.25) if data_type == 'E' else (0.5, -0.25)
        bbox_to_anchor = legend_dict.pop('bbox_to_anchor', default_bbox)
        ncol = legend_dict.pop('ncol', len(self._loader_dict[env]) // 2)
        ax.legend(fontsize=fontsize, ncol=ncol, frameon=frameon,
                  loc=loc, bbox_to_anchor=bbox_to_anchor, **legend_dict)

    def _save_figure(self, figure, figname, subfolder=None, as_pdf=False, transparent=True):
        """
        Save the figure file using the respective path.

        Args:
            figure (object): the figure to save;
            figname (str): the name of the figure;
            subfolder (str, None): optional subfolder where to save the figure;
            as_pdf (bool, False): whether to save the figure in PDF or not;
            transparent (bool, True): whether the figure should be transparent or not.

        """
        extension = '.pdf' if as_pdf else '.png'
        path = self._logger.path
        if subfolder:
            path = path / subfolder
        path.mkdir(exist_ok=True)

        file_path = path / (figname + extension)

        figure.savefig(file_path, transparent=transparent)

    def get_report(self, env, data_type, selected_alg=None):
        """
        Create report plot with matplotlib.

        """
        if data_type == 'V':
            has_value = False
            for alg, loader in self._loader_dict[env].items():
                if (selected_alg is None or alg.startswith(selected_alg + '_')) and loader.value_function_found:
                    has_value = True
                    break
            if not has_value:
                return None

        if data_type == 'E':
            has_entropy = False
            for alg, loader in self._loader_dict[env].items():
                if (selected_alg is None or alg.startswith(selected_alg + '_')) and loader.entropy_found:
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
        default_color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        for alg, loader in self._loader_dict[env].items():
            if selected_alg is None or alg.startswith(selected_alg + '_'):
                if selected_alg is None:
                    color = self._color_cycle[alg]
                    line = '-' if alg not in self._line_cycle else self._line_cycle[alg]
                else:
                    color = next(default_color_cycle)
                    line = '-'
                data = loader.load_aggregate_file(data_type)

                if data is not None:
                    plot_mean_conf(data, ax, color=color, line=line, label=alg)
                    max_epochs = max(max_epochs, len(data[0]))

        if env in self._y_limit and data_type in self._y_limit[env]:
            ax.set_ylim(**self._y_limit[env][data_type])

        ax.set_xlim(xmin=0, xmax=max_epochs-1)
        ax.grid()
        self._legend(ax, env, data_type)
        fig.tight_layout()

        return fig

    def get_boxplot(self, env, metric_type, data_type, selected_alg=None):
        """
        Create boxplot with matplotlib for a given metric.

        Args:
            env (str): The environment name;
            metric_type (str): The metric to compute.

        Returns:
            A figure with the desired boxplot of the given metric.

        """
        if data_type == 'V':
            has_value = False
            for alg, loader in self._loader_dict[env].items():
                if (selected_alg is None or alg.startswith(selected_alg + '_')) and loader.value_function_found:
                    has_value = True
                    break
            if not has_value:
                return None

        self.plot_counter += 1

        plot_id = self.plot_counter * 1000
        fig = plt.figure(plot_id, figsize=(8, 6), dpi=80)
        ax = plt.axes()
        ax.set_title(f'{metric_type} {data_type}', fontweight='bold')

        metric_function = getattr(metrics, f'{metric_type}_metric')

        boxplot_data = list()
        boxplot_labels = list()
        for alg, loader in self._loader_dict[env].items():
            if selected_alg is None or alg.startswith(selected_alg + '_'):
                data = loader.load_aggregate_file(data_type)
                if data is not None:
                    boxplot_data.append(metric_function(data))
                    boxplot_labels.append(alg)

        if len(boxplot_data) == 0:
            return None

        ax.boxplot(boxplot_data, showfliers=False, labels=boxplot_labels)
        ax.grid()
        fig.tight_layout()

        return fig

    def save_reports(self, as_pdf=True, transparent=True, alg_sweep=False):
        """
        Method to save an image of a report of the training metrics from a performed experiment.

        Args:
            as_pdf (bool, True): whether to save the reports as pdf files or png;
            transparent (bool, True): If true, the figure background is transparent and not white;
            alg_sweep (bool, False): If true, the method will generate a separate figure for each algorithm sweep.

        """
        for env in self._loader_dict.keys():
            for data_type in ['J', 'R', 'V', 'E']:
                if alg_sweep:
                    env_dir = self._logger.path / env
                    for alg_dir in env_dir.iterdir():
                        alg = alg_dir.name
                        fig = self.get_report(env, data_type, alg)

                        if fig is not None:
                            self._save_figure(fig, data_type, env + '/' + alg, as_pdf=as_pdf, transparent=transparent)
                            plt.close(fig)
                else:
                    fig = self.get_report(env, data_type)

                    if fig is not None:
                        self._save_figure(fig, data_type, env, as_pdf=as_pdf, transparent=transparent)
                        plt.close(fig)

    def save_boxplots(self, as_pdf=True, transparent=True, alg_sweep=False):
        """
        Method to save an image of a report of the training metrics from a performed experiment.

        Args:
            as_pdf (bool, True): whether to save the reports as pdf files or png;
            transparent (bool, True): If true, the figure background is transparent and not white;
            alg_sweep (bool, False): If true, the method will generate a separate figure for each algorithm sweep.

        """
        for env in self._loader_dict.keys():
            for data_type in ['J', 'R', 'V']:
                for metric in ['max', 'convergence']:
                    if alg_sweep:
                        env_dir = self._logger.path / env
                        for alg_dir in env_dir.iterdir():
                            alg = alg_dir.name
                            fig = self.get_boxplot(env, metric, data_type, alg)

                            if fig is not None:
                                self._save_figure(fig, f'{metric}_{data_type}', env + '/' + alg,
                                                  as_pdf=as_pdf, transparent=transparent)
                                plt.close(fig)
                    else:
                        fig = self.get_boxplot(env, metric, data_type)

                        if fig is not None:
                            self._save_figure(fig, f'{metric}_{data_type}', env, as_pdf=as_pdf, transparent=transparent)
                            plt.close(fig)

    def show_reports(self, boxplots=False, alg_sweep=False):
        """
        Method to show a report of the training metrics from a performend experiment.

        Args:
            alg_sweep (bool, False): If true, the method will generate a separate figure for each algorithm sweep.

        """
        matplotlib.use(default_backend)
        for env in self._loader_dict.keys():
            for data_type in ['J', 'R', 'V', 'E']:
                if alg_sweep:
                    for alg in self._loader_dict[env].keys():
                        self.get_report(env, data_type, alg)
                else:
                    self.get_report(env, data_type)

            plt.show()

        if boxplots:
            for env in self._loader_dict.keys():
                for metric in ['max', 'convergence']:
                    for data_type in ['J', 'R', 'V']:
                        if alg_sweep:
                            for alg in self._loader_dict[env].keys():
                                self.get_boxplot(env, metric, data_type, alg)
                        else:
                            self.get_boxplot(env, metric, data_type)

                plt.show()
