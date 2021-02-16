def _is_primitive(obj):
    """
    Check if the given object is a primitive type

    Args:
        obj: the object to test
    Returns:
        True if the object is primitive i.e. hasn't a __dict__ attribute

    """
    return not hasattr(obj, '__dict__')


def object_to_primitive(obj):
    """
    Converts an object into a string using the class name

    Args:
        obj: the object to convert.

    Returns:
        A string representing the object.

    """
    if isinstance(obj, type):
        return obj.__name__
    else:
        return type(obj).__name__


def dictionary_to_primitive(data):
    """
    Function that converts a dictionary by transforming any objects inside into strings

    Args:
        data (dict): the dictionary to convert.

    Returns:
        The converted dictionary.

    """
    primitive_data = dict()

    for key, value in data.items():

        if _is_primitive(value):
            if isinstance(value, dict):
                primitive_value = dictionary_to_primitive(value)
            else:
                primitive_value = value
        else:
            primitive_value = object_to_primitive(value)

        primitive_data[key] = primitive_value

    return primitive_data
