import functools
import sys
import time
from functools import wraps
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
import sidekick as sk

from .base import Property


def method_as_function(method, wrapper=None):
    """
    Take a method from a property object and transform it in a independent
    function that takes the owner object as first input.
    """

    if wrapper is None:
        try:
            mod = sys.modules[method.__module__]
            path = method.__qualname__.partition(".")[0]
            wrapper = getattr(mod, path)
        except AttributeError:
            pass
        else:
            wrapper = lambda x: Property(x)

    @wraps(method)
    def fn(_object, *args, **kwargs):
        return method(wrapper(_object), *args, **kwargs)

    return fn


def function_as_method(fn):
    """
    Convert function that receives value to a method of a property accessor
    for that value.
    """

    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        return fn(self._object, *args, **kwargs)

    return wrapped


@sk.fn.curry(1)
def cached(fn, ttl=None, small=None):
    """
    Tips toward a caching mechanism.

    Currently, it is a no-op, but we plan to make an actual caching
    implementation in the future.
    """
    return fn


@sk.fn.curry(1)
def cached_small_value(fn, ttl=False, maxsize=None):
    """
    Small cached values are stored in memory with an optional ttl.
    """

    cache = {}
    clean_size = max(maxsize // 2, 1)

    @wraps(fn)
    def decorated(self, *args, **kwargs):
        key = Args.from_signature(*args, **kwargs)

        try:
            result = cache[key]
        except KeyError:
            value = get_cached_small_value(self, fn, *args, **kwargs)
            result = value, (time.time() + ttl if ttl else None)

        if maxsize and len(cache) >= maxsize:
            for k in list(sk.islice(cache, clean_size)):
                del cache[k]
        cache[key] = result
        return result[0]

    return decorated


def get_cached_small_value(*args, **kwargs):
    """
    Non-decorated base implementation of a method that fetches a property from
    a method of a class.

    Method receives
    """
    prop, method, *args = args
    if isinstance(method, str):
        fn = getattr(prop, method)
    else:
        fn = method.__get__(prop)
    return fn(*args, **kwargs)


class Args(NamedTuple):
    args: tuple
    kwargs: MappingProxyType
    hash: int

    @classmethod
    def from_signature(cls, *args, **kwargs):
        hash_key = hash(tuple(iter_hash(args, kwargs)))
        return Args(args, MappingProxyType(kwargs), hash_key)

    def __hash__(self):
        return self.hash

    def __iter__(self):
        yield self.args
        yield self.kwargs


@functools.singledispatch
def value_hash(x):
    return hash(x)


@value_hash.register(np.ndarray)
def _(x):
    return hash(x.tostring())


@value_hash.register(list)
def _hash_sequence(x):
    return hash(tuple(map(value_hash, x)))


def iter_hash(args, kwargs, value_hash=value_hash):
    for i, arg in enumerate(args):
        yield (i, value_hash(arg))
    for k, v in sorted(kwargs.items()):
        yield (k, value_hash(v))
