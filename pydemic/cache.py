from functools import wraps
from time import time

from .config import memory
from .types import Result


def ttl_cache(key, fn=None, *, timeout=6 * 3600, **cache_kwargs):
    """
    Decorator that creates a cached version of function that stores results
    in disk for the given timeout (in seconds).

    Args:
        timeout:
            Maximum time the item is kept in cache (in seconds).

    Returns:
        A decorated function that stores items in the given cache for the given
        timeout.

    Examples:
        >>> @ttl_cache("my-cache", timeout=3600)
        ... def expensive_function(url):
        ...     # Some expensive function, possibly touching the internet...
        ...     response = requests.get(url)
        ...     ...
        ...     return pd.DataFrame(response.json())

    Notes:
        The each pair of (cache name, function name) must be unique. It cannot
        decorate multiple lambda functions or callable objects with no __name__
        attribute.
    """
    if not fn:
        return lambda f: ttl_cache(key, f, timeout=timeout, **cache_kwargs)

    mem = memory(key)

    # We need to wrap fn into another decorator to preserve its name and avoid
    # confusion with joblib's cache. This function just wraps the result of fn
    # int a Result() instance with the timestamp as info.
    @mem.cache(**cache_kwargs)
    @wraps(fn)
    def cached(*args, **kwargs):
        return Result(fn(*args, **kwargs), time())

    # Now the decorated function asks for the result in the cache, checks
    # if it is within the given timeout and return or recompute the value
    @wraps(fn)
    def decorated(*args, **kwargs):
        mem_item = cached.call_and_shelve(*args, **kwargs)
        result = mem_item.get()
        if result.info + timeout < time():
            mem_item.clear()
            result = cached(*args, **kwargs)
        return result.value

    decorated.clear = mem.clear
    decorated.prune = mem.reduce_size

    return decorated


def simple_cache(*fn_or_key):
    """
    A simple in-disk cache.

    Can be called as ``simple_cache(key, fn)``, to decorate a function or as as
    decorator in ``@simple_cache(key)``.
    """
    if len(fn_or_key) == 2:
        fn, key = fn_or_key
    else:
        fn = None
        (key,) = fn_or_key

    if not fn:
        return lambda f: ttl_cache(f, key)

    return memory(key).cache(fn)
