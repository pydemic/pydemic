import datetime
from functools import wraps, lru_cache
from time import time
from typing import Union, Type, Sequence

import sidekick as sk

from pydemic.utils import as_seq
from . import config
from .types import Result
from .utils import today

PERIOD_ALIASES = {
    "day": datetime.timedelta(days=1),
    "week": datetime.timedelta(days=7),
    **{"{n}h": datetime.timedelta(hours=n) for n in range(1, 25)},
}

# TODO: abstract the many available Python caching libs and move this
# functionality to sidekick
#
# References and similar projects
# - http://joblib.readthedocs.io/
# - https://cachetools.readthedocs.io/
# - https://github.com/lonelyenvoy/python-memoization


@sk.fn.curry(2)
def ttl_cache(key, fn, *, timeout=6 * 3600, **cache_kwargs):
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
    mem = config.memory(key)

    # We need to wrap fn into another decorator to preserve its name and avoid
    # confusion with joblib's cache. This function just wraps the result of fn
    # int a Result() instance with the timestamp as info.
    @mem.cache(**cache_kwargs)
    @wraps(fn)
    def cached(*args, **kwargs):
        return Result(fn(*args, **kwargs), time())

    # Now the decorated function asks for the result in the cache, checks
    # if it is within the given timeout and return or recompute the value
    @wraps_with_cache(fn, cached)
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


@sk.fn.curry(2)
def disk_cache(key, fn):
    """
    A simple in-disk cache.

    Can be called as ``simple_cache(key, fn)``, to decorate a function or as as
    decorator in ``@simple_cache(key)``.
    """
    return config.memory(key).cache(fn)


@sk.fn.curry(2)
def period_cache(
    period: Union[str, int, datetime.timedelta],
    fn: callable,
    memory=None,
    maxsize=2048,
    fallback: Sequence[Type[Exception]] = None,
):
    """
    Keeps value in cache within n intervals of the given time delta.

    Args:
        period:
            Time period in which the cache expires. Can be given as a timedelta,
            a integer (in seconds) or a string in the set {'day', 'week', '1h',
            '2h', ..., '24h'}.

            Other named periods cah be registered using the :func:`register_period`
            function.
        fn:
            The decorated function.
        memory (str):
            If given, corresponds to a memory object returned by :func:`memory`.
        maxsize:
            If no memory object is given, corresponds to the maxsize used by
            the lru_cache mechanism.
        fallback:
            If an exception or list of exceptions, correspond to the kinds of
            errors that triggers the cache to check previously stored responses.
            There is nothing that guarantees that the old values will still
            be present, but it gives a second attempt that may hit the cache
            or call the function again.

    Examples:
        >>> @period_cache("day")
        ... def fn(x):
        ...     print('Doing really expensive computation...')
        ...     return ...
    """

    # Select the main method to decorate the cached function
    if memory:
        mem = config.memory(memory)
        decorator = mem.cache
    else:
        decorator = lru_cache(maxsize)

    # Reads a period and return a function that return increments of the period
    # according to the current time. This logic is encapsulated into the key()
    # function.
    date = today()
    ref_time = datetime.datetime(date.year, date.month, date.day).timestamp()
    if isinstance(period, str):
        period = PERIOD_ALIASES[period].seconds
    period = int(period)
    get_time = time
    key = lambda: int(get_time() - ref_time) // period

    # The main cached function. This is stored only internally and the function
    # exposed to the user fixes the _cache_bust and _recur parameters to the
    # correct values.
    fallback = tuple(as_seq(fallback)) if fallback else ImpossibleError

    @decorator
    def cached(_cache_bust, _recur, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except fallback:
            if _recur > 0:
                return cached(_cache_bust - 1, _recur - 1, *args, **kwargs)
            raise

    # Save function
    @wraps_with_cache(fn, cached)
    def decorated(*args, **kwargs):
        return cached(key(), 1, *args, **kwargs)

    return decorated


class ImpossibleError(Exception):
    """
    It is an error to raise this exception, do not use it!
    """


def wraps_with_cache(fn, cache=None):
    """
    Like functools.wraps, but also copy the cache methods created either
    by lru_cache or by joblib.Memory.cache.
    """
    cache = cache or fn
    wrapped = wraps(fn)
    for attr in ("cache_info", "clear_cache"):
        if hasattr(cache, attr):
            setattr(wrapped, attr, getattr(cache, attr))
    return wrapped
