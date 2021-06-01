#!/usr/bin/env python3

import yaml
from pathlib import Path
from argparse import ArgumentParser

from mushroom_rl.core import Logger

from mushroom_rl_benchmark import BenchmarkSuite
from mushroom_rl_benchmark.utils import build_sweep_list


def get_args():
    parser = ArgumentParser()
    arg_test = parser.add_argument_group('benchmark parameters')
    arg_test.add_argument("-e", "--env", type=str, nargs='+', required=True,
                          help='Environments to be used by the benchmark. '
                               'Use \'all\' to select all the available environments.')
    arg_test.add_argument("-a", "--algorithm",  type=str, nargs='+', default=['all'],
                          help='Algorithms to be used by the benchmark. '
                               'Use \'all\' to select all the algorithms defined in the config file.')
    arg_test.add_argument("-s", "--sweep", type=str, required=False,
                          help='Sweep configuration file to be used by the benchmark.')
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
    env_ids, algs, sweep, exec_type, test, demo = get_args()
    cfg_dir = Path(__file__).parent / 'cfg'
    env_cfg_dir = cfg_dir / 'env'
    sweep_cfg_dir = cfg_dir / 'sweep'

    param_path = 'suite.yaml'
    plots_path = 'plots.yaml'

    logger = Logger(results_dir=None)

    logger.info('Starting benchmarking script')

    if 'all' in env_ids:
        logger.info('Running benchmark on all available environments')
        assert len(env_ids) == 1
        env_ids = list()
        for env_id in env_cfg_dir.iterdir():
            if env_id.suffix == '.yaml':
                env_ids.append(env_id.stem)

    logger.info('Execution type: ' + exec_type)
    logger.info('Running FULL: ' + str(not demo))

    if sweep is not None:
        logger.info(f'Sweep configuration file: {sweep}')
    logger.strong_line()

    with open(cfg_dir / param_path, 'r') as param_file:
        suite_params = yaml.safe_load(param_file)['suite_params']

    with open(cfg_dir / plots_path, 'r') as plots_file:
        plot_params = yaml.safe_load(plots_file)

    if sweep is not None:
        sweep_file_name = sweep + '.yaml'
        with open(sweep_cfg_dir / sweep_file_name, 'r') as sweep_file:
            sweep_params = yaml.safe_load(sweep_file)

    suite = BenchmarkSuite(**suite_params)

    for env_id in env_ids:
        config_path = cfg_dir / 'env' / (env_id + '.yaml')

        if not config_path.exists():
            logger.error('The environment configuration file ' + config_path.name + ' does not exists')
            exit()

        with open(config_path, 'r') as config_file:
            yaml_file = yaml.safe_load(config_file)
        run_params = yaml_file['run_params']
        env_data = yaml_file['env_params']
        agent_data = yaml_file['agent_params']

        if 'all' not in algs:
            agent_data = {k: agent_data[k] for k in algs}

        agents = agent_data.keys()
        agents_params = agent_data.values()

        env = env_data['name']
        env_params = env_data['params']

        if demo:
            run_params['n_runs'] = 4
            run_params['n_epochs'] = 10
            if 'n_steps' in run_params:
                run_params['n_steps'] = 15000
            else:
                run_params['n_episodes'] = 10
            if 'n_episodes_test' in run_params:
                run_params['n_episodes_test'] = 5
            else:
                run_params['n_steps_test'] = 1000

        if sweep is None:
            suite.add_experiments(env, env_params, agents, agents_params, **run_params)
        else:
            sweep_list = build_sweep_list(agents, sweep_params)
            suite.add_experiments_sweeps(env, env_params, agents, agents_params, sweep_list, **run_params)

    suite.print_experiments()
    logger.strong_line()

    suite.save_parameters()

    if not test:
        logger.info('Running the benchmarks')
        logger.weak_line()
        suite.run(exec_type=exec_type)

        if exec_type != 'slurm':
            logger.info('Saving the plots on disk')
            suite.save_plots(**plot_params)
