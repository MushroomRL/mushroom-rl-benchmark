#!/usr/bin/env python3

import json
from pathlib import Path
from argparse import ArgumentParser

from mushroom_rl.core import Logger

from mushroom_rl_benchmark import BenchmarkSuite


def get_args(argv=None):
    parser = ArgumentParser()
    arg_test = parser.add_argument_group('benchmark parameters')
    arg_test.add_argument("-e", "--env", type=str, nargs='+', required=True,
                          help='Environments to be used by the benchmark. '
                               'Use \'all\' to select all the available environments.')
    arg_test.add_argument("-a", "--algorithm", type=str, nargs='+', default=['all'],
                          help='Algorithms to be used by the benchmark. '
                               'Use \'all\' to select all the algorithms defined in the config file.')
    arg_test.add_argument("-s", "--seeds", type=int,
                          help='Number of seeds per experiment. By default, use the environment configuration.')
    arg_test.add_argument("-x", "--execution-type",
                          choices=['sequential', 'parallel', 'slurm'],
                          help='Execution type for the benchmark. By default, use the Hydra configuration.')
    arg_test.add_argument("-t", "--test", action='store_true',
                          help='Flag to test the script and NOT execute the benchmark.')
    arg_test.add_argument("-d", "--demo", action='store_true',
                          help='Flag to run a reduced version of the benchmark.')
    arg_test.add_argument("-o", "--output-dir", type=str,
                          help='Result directory. By default, use a timestamped output directory.')
    arg_test.add_argument('--quiet', action='store_true',
                          help='Disable experiment logs and progress bars.')
    arg_test.add_argument('--override', action='append', default=[], metavar='HYDRA_OVERRIDE',
                          help='Additional Hydra override. Repeat for multiple overrides.')

    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    if args.seeds is not None and args.seeds < 1:
        raise SystemExit('--seeds must be positive')

    n_seeds = args.seeds
    if args.demo:
        n_seeds = 2

    logger = Logger(results_dir=None)

    logger.info('Starting benchmarking script')
    logger.info('Execution type: ' + (args.execution_type or 'configured default'))
    logger.info(f'Running full benchmark: {not args.demo}')
    if n_seeds is None:
        logger.info('Using the configured number of seeds for each environment')
    else:
        logger.info(f'Number of seeds per experiment: {n_seeds}')

    config_dir = Path(__file__).parent / 'cfg'
    suite = BenchmarkSuite(config_dir=config_dir, logger=logger)

    if args.demo:
        suite.set_demo_run_params()

    try:
        suite.add_selected(args.env, args.algorithm)
    except (KeyError, ValueError) as error:
        raise SystemExit(str(error)) from error

    logger.info('Running the benchmarks')
    logger.weak_line()
    overrides = args.override.copy()
    if n_seeds is not None:
        overrides.append(f'+n_seeds={n_seeds}')
    if args.output_dir is not None:
        overrides.append(f'output_root={json.dumps(args.output_dir)}')
    if args.quiet:
        overrides.append('log_console=false')
    suite.run(exec_type=args.execution_type, test=args.test, overrides=overrides)


if __name__ == '__main__':
    main()
