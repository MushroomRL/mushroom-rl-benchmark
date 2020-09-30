#!/usr/bin/env python3

import yaml
from pathlib import Path
from argparse import ArgumentParser
from mushroom_rl_benchmark import BenchmarkSuite


def get_args():
    parser = ArgumentParser()
    arg_test = parser.add_argument_group('benchmark_script')
    arg_test.add_argument("-e", "--env", type=str, required=True,
                          help='Select an environment for the benchmark.')
    arg_test.add_argument("-s", "--slurm", action='store_true',
                          help='Flag to signalize the usage of SLURM.')
    arg_test.add_argument("-t", "--test", action='store_true',
                          help='Flag go signalize that you want to test the script and NOT execute the benchmark.')
    arg_test.add_argument("-f", "--full", action='store_true',
                          help='Flag to indicate that you want to run the full benchmark.')

    args = vars(parser.parse_args())
    return args.values()


if __name__ == '__main__':
    env_id, use_slurm, test, full_experiment = get_args()
    script_dir = Path(__file__).parent
    config_file = script_dir / 'env' / (env_id + '.yaml')

    agent_data, env_data = yaml.safe_load(open(config_file, 'r')).values()

    agents = agent_data.keys()
    agents_params = agent_data.values()

    env = env_data['name']
    env_params = env_data['params']

    print('Environment:', env)
    print('Agents:', str(list(agents)))
    print('Using SLURM:', use_slurm)
    print('Runing FULL:', full_experiment)
    print()

    exec_type = 'slurm' if use_slurm else 'parallel'
    slurm_conf = 'params_slurm.yaml' if not full_experiment else 'params_slurm_full.yaml'
    local_conf = 'params_local.yaml' if not full_experiment else 'params_local_full.yaml'
    param_file = slurm_conf if use_slurm else local_conf

    run_params, suite_params = yaml.safe_load(open(script_dir / param_file, 'r')).values()

    suite = BenchmarkSuite(
        **suite_params,
        **run_params)

    for agent, agent_params in zip(agents, agents_params):
        suite.add_experiment(env, env_params, agent, agent_params)

    suite.print_experiments()

    if not test:
        suite.run(exec_type=exec_type)
