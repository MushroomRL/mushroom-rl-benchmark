#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

from mushroom_rl_benchmark.core import aggregate_results


def get_args():
    parser = ArgumentParser()
    arg_test = parser.add_argument_group('benchmark parameters')
    arg_test.add_argument("-d", "--log-dir", type=str, required=True,
                          help='Path of the top level folder dir')
    arg_test.add_argument("-s", "--is-sweep", action='store_true',
                          help='If the logs are from a parameter sweep or not')
    arg_test.add_argument("-i", "--ignore", type=str, nargs='*', default=[],
                          help='Folders to ignore')

    args = parser.parse_args()

    return Path(args.log_dir), args.is_sweep, args.ignore


if __name__ == '__main__':
    path, sweep, ignore = get_args()

    for env_dir in path.iterdir():
        if env_dir.is_dir() \
                and env_dir.name not in ['plots', 'params'] \
                and env_dir.name not in ignore:
            for alg_dir in env_dir.iterdir():
                if alg_dir.is_dir() and alg_dir.name != 'plots' and alg_dir.name not in ignore:
                    if sweep:
                        for sweep_dir in alg_dir.iterdir():
                            if sweep_dir.is_dir() and sweep_dir.name not in ignore:
                                aggregate_results(sweep_dir)
                    else:
                        aggregate_results(alg_dir)
