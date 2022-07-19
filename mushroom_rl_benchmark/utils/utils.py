from inspect import signature


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