def build_sweep_list(algs, sweep_conf, base_name='c_'):
    """
    Build the sweep list, from a compact dictionary specification, for every considered algorithm.

    Args:
        algs (list): list of algorithms to be considered;
        sweep_conf (dict): dictionary with a compact sweep configuration for every algorithm;
        base_name (str, 'c_'): base name for the sweep configuiration.

    Returns:
        The sweep list to be used with the suite.

    """
    sweep_list = list()

    for alg in algs:
        raw_sweep_dict = sweep_conf[alg]
        sweep_dict = build_sweep_dict(base_name=base_name, **raw_sweep_dict)
        sweep_list.append(sweep_dict)

    return sweep_list


def build_sweep_dict(base_name='c_', **kwargs):
    """
    Build the sweep dictionary, from a set of variable specifications.

    Args:
        base_name (str, 'c_'): base name for the sweep configuiration;
        **kwargs: the parameter specifications for the sweep.

    Returns:
        The sweep dictionary, where the key is the sweep name and the value is a dictionary with the sweep parameters.

    """

    sweep_dict = dict()

    for sweep_name, sweep_params in generate_sweep(base_name=base_name, **kwargs):
        sweep_dict[sweep_name] = sweep_params

    return sweep_dict


def generate_sweep(base_name='c_', **kwargs):
    """
    Generator that returns tuples with sweep name and parameters

    Args:
        base_name (str, 'c_'): base name for the sweep configuiration;
        **kwargs: the parameter specifications for the sweep.

    """
    for i, sweep_params in enumerate(generate_sweep_params(**kwargs)):
        sweep_name = base_name + str(i)
        yield sweep_name, sweep_params


def generate_sweep_params(**kwargs):
    """
    Generator that returns sweep parameters

    Args:
        **kwargs: the parameter specifications for the sweep.

    """
    assert len(kwargs) > 0
    current_dict = dict()
    items = list(kwargs.items())
    yield from _generate_sweep_params_recursive(current_dict, items)


def _generate_sweep_params_recursive(current_dict, items):
    if len(items) > 0:
        key, values = items[0]

        for value in values:
            current_dict[key] = value
            yield from _generate_sweep_params_recursive(current_dict, items[1:])
    else:
        yield current_dict


if __name__ == '__main__':
    a_ppo = [1, 2]
    b_ppo = ['a', 'b']

    a_trpo = [3, 4]
    b_trpo = ['c', 'd']

    sweep_conf = {
        'PPO': dict(a=a_ppo, b=b_ppo),
        'TRPO': dict(a=a_trpo, b=b_trpo),
    }
    algs = list(sweep_conf.keys())

    print('ppo sweep')
    for p in generate_sweep_params(a=a_ppo, b=b_ppo):
        print(p)

    print('trpo sweep')
    print(list(generate_sweep(a=a_trpo, b=b_trpo)))

    print('sweep_list')
    print(algs)
    print(build_sweep_list(algs, sweep_conf))

