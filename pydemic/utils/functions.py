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