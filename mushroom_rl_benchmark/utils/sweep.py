def generate_sweep(sweep_config, base_name='c_'):
    """
    Generator that returns tuples with sweep name and parameters

    Args:
        sweep_config (dict): the parameter specifications for the sweep;
        base_name (str, 'c\_'): base name for the sweep configuiration.

    """
    for i, sweep_params in enumerate(generate_sweep_params(**sweep_config)):
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
        yield current_dict.copy()
