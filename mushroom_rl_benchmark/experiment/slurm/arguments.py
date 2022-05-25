import argparse


def make_arguments(**params):
    """
    Create a script argument string from dictionary

    """
    param_strings = ["--{} '{}'".format(key, params[key]) for key in params.keys()]
    return ' '.join(param_strings)


def read_arguments_run(arg_string=None):
    """
    Parse the arguments for the run script.

    Args:
        arg_string (str, None): pass the argument string.

    """
    parser = argparse.ArgumentParser()

    exp_args = parser.add_argument_group('mushroom_rl_benchmark')
    exp_args.add_argument("--log_dir", type=str, required=True)
    exp_args.add_argument("--n_epochs", type=int, required=True)
    exp_args.add_argument("--n_steps", type=int, default=None)
    exp_args.add_argument("--quiet", type=bool, default=True)
    exp_args.add_argument("--save_agent", type=bool, default=False)
    exp_args.add_argument('--n_steps_test', type=int, default=None)
    exp_args.add_argument("--n_episodes_test", type=int, default=None)
    exp_args.add_argument('--seed', type=int, default=None)

    if arg_string is not None:
        args = vars(parser.parse_args(arg_string))
    else:
        args = vars(parser.parse_args())

    log_dir = args['log_dir']
    del args['log_dir']

    return log_dir, args


def read_arguments_aggregate(arg_string=None):
    """
    Parse the arguments for the aggregate script.

    Args:
        arg_string (str, None): pass the argument string.

    """
    parser = argparse.ArgumentParser()

    arg_test = parser.add_argument_group('mushroom_rl_benchmark')
    arg_test.add_argument("--log_dir", type=str, required=True)
    arg_test.add_argument("--log_id", type=str, required=True)
    arg_test.add_argument('--seed', type=int, default=None)

    if arg_string is not None:
        args = vars(parser.parse_args(arg_string))
    else:
        args = vars(parser.parse_args())

    log_dir = args['log_dir']
    log_id = args['log_id']

    return log_dir, log_id
