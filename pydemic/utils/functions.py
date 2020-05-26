from functools import lru_cache, wraps

import numpy as np


def interpolant(x, y):
    """
    Creates a linear interpolant for the given function passed as sequences of
    x, y points.
    """
    x = np.array(x)
    y = np.array(y)

    def fn(t):
        return np.interp(t, x, y)

    return fn


def lru_safe_cache(size):
    """
    A safe LRU cache that returns a copy of the cached element to prevent
    mutations from cached values.
    """

    def decorator(func):
        cached = lru_cache(size)(func)

        @wraps(func)
        def fn(*args, **kwargs):
            return cached(*args, **kwargs).copy()

        fn.unsafe = cached
        return fn

    return decorator


def coalesce(*args, raises=False):
    """
    Return the first non-null value.

    If raises=True and no non-null value is found, raise a ValueError.

    Examples:
        >>> coalesce(None, "first", None, "second")
        'first'
    """
    for arg in args:
        if arg is not None:
            return arg
    if raises:
        raise ValueError("All values are null.")
    return None


def maybe_run(*args, **kwargs):
    """
    Return None if argument after the input function is null, otherwise
    execute fn(*args, **kwargs)

    Examples:
        >>> from math import sqrt
        >>> maybe_run(sqrt, 4)
        2
        >>> maybe_run(sqrt, None)
        None

    """
    if args[1] is None:
        return None
    args = iter(args)
    return next(args)(*args, **kwargs)
