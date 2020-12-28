#!/usr/bin/env python3

import yaml
from pathlib import Path
from argparse import ArgumentParser
from mushroom_rl_benchmark import BenchmarkSuite


def get_args():
    parser = ArgumentParser()
    arg_test = parser.add_argument_group('benchmark parameters')
    arg_test.add_argument("-e", "--env", type=str, required=True,
                          help='Environment to benchmark.')
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
    env_id, use_slurm, sequential, test, demo = get_args()
    cfg_dir = Path(__file__).parent / 'cfg'
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

    print('Environment:', env)
    print('Agents:', str(list(agents)))

    type_msg = 'parallel'
    type_msg = 'slurm' if use_slurm else type_msg
    type_msg = 'sequential' if sequential else type_msg
    print('Execution type:', type_msg)
    print('Runing FULL:', not demo)
    print()

    if demo:
        run_params['n_runs'] = 4
        run_params['n_epochs'] = 10
        run_params['n_steps'] = 15000
        run_params['n_episodes_test'] = 5

    exec_type = 'slurm' if use_slurm else 'parallel'
    exec_type = 'sequential' if sequential else 'parallel'
    param_file = 'params_slurm.yaml' if use_slurm else 'params_local.yaml'

    suite_params = yaml.safe_load(open(cfg_dir / param_file, 'r'))['suite_params']
    
    suite = BenchmarkSuite(
        **suite_params,
        **run_params)

    for agent, agent_params in zip(agents, agents_params):
        suite.add_experiment(env, env_params, agent, agent_params)

    suite.print_experiments()

    if not test:
        suite.run(exec_type=exec_type)
