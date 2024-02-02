from copy import deepcopy

__env_param_prefix__ = 'env_params_'


def mask_env_parameters(env_parameters):
    masked_env_parameters = dict(env_name=env_parameters['name'])

    if env_parameters['params'] is None:
        env_parameters['params'] = dict()
        
    for key, value in env_parameters['params'].items():
        masked_key = __env_param_prefix__ + key
        masked_env_parameters[masked_key] = value

    return masked_env_parameters


def extract_env_parameters(kwargs):
    env_parameters = dict()
    agent_parameters = deepcopy(kwargs)
    offset = len(__env_param_prefix__)

    delete_list = []

    for key, value in kwargs.items():
        if key.startswith(__env_param_prefix__):
            delete_list.append(key)
            env_parameters[key[offset:]] = value

    for key in delete_list:
        del agent_parameters[key]

    return env_parameters, agent_parameters
