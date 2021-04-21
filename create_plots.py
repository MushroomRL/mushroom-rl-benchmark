#!/usr/bin/env python3

import yaml
from pathlib import Path
from argparse import ArgumentParser

from mushroom_rl_benchmark import BenchmarkSuiteVisualizer, BenchmarkLogger


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, required=True,
                          help='Benchmark directory where the plots generation is needed')
    parser.add_argument("-p", "--parameter-sweep", action='store_true',
                        help='Flag to consider the benchmark as a parameter sweep.')
    parser.add_argument("-s", "--show", action='store_true',
                        help='Flag to show the plots and not only save them.')

    args = vars(parser.parse_args())
    return args.values()


if __name__ == '__main__':
    path, sweep, show = get_args()

    plots_file = Path('cfg') / 'plots.yaml'
    logger = BenchmarkLogger.from_path(path)

    with open(plots_file, 'r') as plots_file:
        plot_params = yaml.safe_load(plots_file)

    visualizer = BenchmarkSuiteVisualizer(logger, sweep, **plot_params)
    if show:
        visualizer.show_reports()
    visualizer.save_reports(as_pdf=False)
