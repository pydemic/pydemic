from functools import singledispatch

import numpy as np
import pandas as pd


@singledispatch
def to_json(obj):
    """
    Convert object to JSON.
    """
    try:
        method = obj.to_json
    except AttributeError:
        name = type(obj).__name__
        raise TypeError(f"'{name}' objects cannot be converted to JSON")
    else:
        return method()


@to_json.register(int)
@to_json.register(float)
@to_json.register(str)
@to_json.register(bool)
@to_json.register(type(None))
def _(x):
    return x


@to_json.register(list)
@to_json.register(tuple)
@to_json.register(set)
@to_json.register(frozenset)
@to_json.register(np.ndarray)
def _(xs):
    return [to_json(x) for x in xs]


@to_json.register(dict)
@to_json.register(pd.Series)
def _(xs):
    return {check_string(k): to_json(v) for k, v in xs.items()}


def check_string(x):
    if isinstance(x, str):
        return x
    elif isinstance(x, bool):
        return str(x).lower()
    elif isinstance(x, (int, float)):
        return str(x)
    elif x is None:
        return "null"
    elif x is ...:
        return "..."
    else:
        name = type(x).__name__
        raise TypeError(f"invalid type for object key: {name}")
