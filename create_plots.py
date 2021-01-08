#!/usr/bin/env python3

import yaml
from pathlib import Path
from argparse import ArgumentParser
from mushroom_rl_benchmark import BenchmarkSuiteVisualizer, BenchmarkLogger


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, required=True,
                          help='Benchmark directory where the plots generation is needed')
    args = vars(parser.parse_args())
    return args.values()


if __name__ == '__main__':
    path, = get_args()

    plots_file = Path('cfg') / 'plots.yaml'
    logger = BenchmarkLogger.from_path(path)

    with open(plots_file, 'r') as plots_file:
        plot_params = yaml.safe_load(plots_file)

    visualizer = BenchmarkSuiteVisualizer(logger, **plot_params)
    visualizer.show_reports()
    visualizer.save_reports()
