import numpy as np
from inspect import signature

from mushroom_rl.utils.frames import LazyFrames


def get_init_states(dataset):
    """
    Get the initial states of a MushroomRL dataset

    Args:
        dataset (Dataset): a MushroomRL dataset.

    """
    pick = True
    x_0 = list()
    for d in dataset:
        if pick:
            if isinstance(d[0], LazyFrames):
                x_0.append(np.array(d[0]))
            else:
                x_0.append(d[0])
        pick = d[-1]
    return np.array(x_0)


def extract_arguments(args, method):
    """
    Extract the arguments from a dictionary that fit to a methods parameters.

    Args:
        args (dict): dictionary of arguments;
        method (function): method for which the arguments should be extracted.

    """
    intersection = lambda list1, list2: [x for x in list1 if x in list2]
    filterByKey = lambda keys, data: {x: data[x] for x in keys if x in data }
    keys = intersection(signature(method).parameters.keys(), args.keys())
    params = filterByKey(keys, args)
    return params