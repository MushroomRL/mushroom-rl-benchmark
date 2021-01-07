#!/usr/bin/env python3

import yaml
from pathlib import Path
from argparse import ArgumentParser
from mushroom_rl_benchmark import BenchmarkSuite
from mushroom_rl.core import Logger


def get_args():
    parser = ArgumentParser()
    arg_test = parser.add_argument_group('benchmark parameters')
    arg_test.add_argument("-e", "--env", type=str, nargs='+',
                          help='Environments to be used by the benchmark. '
                               'Use all, to select all environments available.')
    arg_test.add_argument("-s", "--slurm", action='store_true',
                          help='Flag to use of SLURM.')
    arg_test.add_argument("-q", "--sequential", action='store_true',
                          help='Flag to run a the benchmark sequentially.')
    arg_test.add_argument("-t", "--test", action='store_true',
                          help='Flag to test the script and NOT execute the benchmark.')
    arg_test.add_argument("-d", "--demo", action='store_true',
                          help='Flag to run a reduced version of the benchmark.')

    args = vars(parser.parse_args())
    return args.values()


if __name__ == '__main__':
    env_ids, use_slurm, sequential, test, demo = get_args()
    cfg_dir = Path(__file__).parent / 'cfg'
    env_cfg_dir = cfg_dir / 'env'

    logger = Logger(results_dir=None)

    logger.info('Starting benchmarking script')

    if 'all' in env_ids:
        logger.info('Running benchmark on all available environments')
        assert len(env_ids) == 1
        env_ids = list()
        for env_id in env_cfg_dir.iterdir():
            if env_id.suffix == '.yaml':
                env_ids.append(env_id.stem)

    type_msg = 'parallel'
    type_msg = 'slurm' if use_slurm else type_msg
    type_msg = 'sequential' if sequential else type_msg
    logger.info('Execution type:' + type_msg)
    logger.info('Runing FULL:' +str(not demo))
    logger.strong_line()

    exec_type = 'slurm' if use_slurm else 'parallel'
    exec_type = 'sequential' if sequential else 'parallel'
    param_file = 'params_slurm.yaml' if use_slurm else 'params_local.yaml'

    suite_params = yaml.safe_load(open(cfg_dir / param_file, 'r'))['suite_params']

    suite = BenchmarkSuite(**suite_params)

    for env_id in env_ids:
        config_file = cfg_dir / 'env' / (env_id + '.yaml')

        assert not (use_slurm and sequential)

        yaml_file = yaml.safe_load(open(config_file, 'r'))
        run_params = yaml_file['run_params']
        env_data = yaml_file['env_params']
        agent_data = yaml_file['agent_params']

        agents = agent_data.keys()
        agents_params = agent_data.values()

        env = env_data['name']
        env_params = env_data['params']

        logger.info('Adding environment:' + env)
        logger.info('Adding agents:' + str(list(agents)))

        if demo:
            run_params['n_runs'] = 4
            run_params['n_epochs'] = 10
            run_params['n_steps'] = 15000
            run_params['n_episodes_test'] = 5

        suite.add_experiments(env, env_params, agents, agents_params, **run_params)

    suite.print_experiments()
    logger.strong_line()

    if not test:
        logger.info('Running the benchmarks')
        logger.weak_line()
        suite.run(exec_type=exec_type)

        if exec_type != 'slurm':
            logger.info('Saving the plots on disk')
            suite.save_plots()
