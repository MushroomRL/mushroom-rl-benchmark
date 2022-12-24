#!/usr/bin/env python3

from pathlib import Path
from argparse import ArgumentParser

from mushroom_rl_benchmark.utils import aggregate_results


def get_args():
    parser = ArgumentParser()
    arg_test = parser.add_argument_group('benchmark parameters')
    arg_test.add_argument("-d", "--log-dir", type=str, required=True,
                          help='Path of the top level folder dir')
    arg_test.add_argument("-i", "--ignore", type=str, nargs='*', default=[],
                          help='Path of the top level folder dir')

    args = parser.parse_args()

    return Path(args.log_dir), args.ignore


if __name__ == '__main__':
    path, ignore = get_args()

    for env_dir in path.iterdir():
        if env_dir.is_dir() and env_dir.name != 'plots' and env_dir.name not in ignore:
            for alg_dir in env_dir.iterdir():
                if alg_dir.is_dir() and alg_dir.name != 'plots'and alg_dir.name not in ignore:
                    aggregate_results(env_dir, alg_dir.name)
