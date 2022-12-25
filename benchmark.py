#!/usr/bin/env python3

from pathlib import Path
from argparse import ArgumentParser

from mushroom_rl.core import Logger

from mushroom_rl_benchmark import BenchmarkSuite


def get_args():
    parser = ArgumentParser()
    arg_test = parser.add_argument_group('benchmark parameters')
    arg_test.add_argument("-e", "--env", type=str, nargs='+', required=True,
                          help='Environments to be used by the benchmark. '
                               'Use \'all\' to select all the available environments.')
    arg_test.add_argument("-a", "--algorithm",  type=str, nargs='+', default=['all'],
                          help='Algorithms to be used by the benchmark. '
                               'Use \'all\' to select all the algorithms defined in the config file.')
    arg_test.add_argument("-s", "--seeds", type=int, default=25,
                          help='Number of seed per experiment')
    # arg_test.add_argument("-s", "--sweep", type=str, required=False,
    #                       help='Sweep configuration file to be used by the benchmark.')
    arg_test.add_argument("-x", "--execution_type",
                          choices=['sequential', 'parallel', 'slurm'],
                          default='parallel',
                          help='Execution type for the benchmark.')
    arg_test.add_argument("-t", "--test", action='store_true',
                          help='Flag to test the script and NOT execute the benchmark.')
    arg_test.add_argument("-d", "--demo", action='store_true',
                          help='Flag to run a reduced version of the benchmark.')

    args = vars(parser.parse_args())
    return args.values()


if __name__ == '__main__':
    env_ids, algs, n_seeds, exec_type, test, demo = get_args()
    cfg_dir = Path(__file__).parent / 'cfg'

    if test:
        exec_type += '_test'

    if demo:
        n_seeds = 2

    logger = Logger(results_dir=None)

    logger.info('Starting benchmarking script')
    logger.info('Execution type: ' + exec_type)
    logger.info('Running full benchmark: ' + str(not demo))
    logger.info('Number of seeds per experiment: ' + str(n_seeds))

    suite = BenchmarkSuite(config_dir=cfg_dir, n_seeds=n_seeds)

    if demo:
        suite.set_demo_run_params()

    if 'all' in env_ids:
        logger.info('Running benchmark on all available environments')
        suite.add_full_benchmark()
    else:
        for env in env_ids:
            if 'all' in algs:
                logger.info(f'Adding all algorithms for the {env} environment')
                suite.add_environment(env)
            else:
                logger.info(f'Adding the following algorithms for the {env} environment:')
                for alg in algs:
                    logger.info(f'- {alg}')
                suite.add_experiments(env, algs)

    logger.info('Running the benchmarks')
    logger.weak_line()
    suite.run(exec_type=exec_type)

