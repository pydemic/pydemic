from collections import MutableMapping, ChainMap
from functools import partial
from operator import itemgetter
from typing import Sequence

import sidekick.api as sk

from .computed_dict_utils import (
    initial_values,
    get_computed_key_declarations,
    get_argnames,
    partial_args,
)


class MissingDict(dict):
    """
    Dictionary that automatically compute values for missing keys.
    """

    __slots__ = ("missing",)

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            self.missing = args[0]
            super().__init__(**kwargs)
        elif len(args) == 2:
            self.missing, data = args
            super().__init__(data, **kwargs)
        else:
            raise TypeError("expect between 1 and 2 positional arguments")

    def __missing__(self, key):
        return self.missing(key)

    def __repr__(self):
        cls = type(self).__name__
        return f"{cls}({dict(self)})"


class ComputedDict(MutableMapping):
    """
    Mapping in which values can be computed from functions.
    """

    __slots__ = ("_dependent", "_independent")
    _initial = None
    _is_callable = staticmethod(callable)

    def __init_subclass__(cls):
        super().__init_subclass__()
        hints = get_computed_key_declarations(cls)
        if hints:
            cls._initial = initial_values(cls)
            cls._initial.update({k: getattr(cls, k, None) for k in hints})
            for k in hints:
                setattr(cls, k, property(itemgetter(k)))

    def __init__(self, *args, **kwargs):
        self._independent: dict = {}
        self._dependent: dict = {}
        if self._initial:
            self.update(self._initial)
        self.update(*args, **kwargs)

    def __len__(self):
        return len(self._independent)

    def __iter__(self):
        yield from self._independent

    def __getitem__(self, key):
        try:
            return self._independent[key]
        except KeyError:
            pass

        try:
            return self.eval(key, {})
        except KeyError as ex:
            msg = f"error raised when evaluating map[{key!r}]: {ex}"
            raise KeyError(msg)

    def __setitem__(self, key, value):
        try:
            if self._is_callable(value):
                self.update({key: value})
            elif key not in self._dependent:
                self._independent[key] = value
            else:
                self._set_computed(key, value)
        except Exception as ex:
            cls = type(ex).__name__
            msg = f"error setting map[{key!r}] = {value!r}; {cls}: {ex}"
            raise KeyError(msg)

    def __delitem__(self, key):
        if any(key in args for args, _ in self._dependent.values()):
            raise ValueError("cannot delete variable with dependents")
        elif key in self._dependent:
            del self._dependent[key]
        else:
            del self._independent[key]

    def __repr__(self):
        cls = type(self).__name__
        return f"{cls}({self._independent})"

    def __getstate__(self):
        return self._independent, self._dependent

    def __setstate__(self, st):
        self._independent, self._dependent = st

    def _set_computed(self, key, value):
        """
        Set a computed value.
        """
        args, fn = self._dependent[key]
        if hasattr(fn, "_func_inverse_"):
            inv = fn._func_inverse_
            _, *inv_args = sk.signature(inv).argnames()
            if inv_args:
                env = self.get_keys(inv_args, {key: value}, dependents=True)
                self[args[0]] = inv(value, *(env[k] for k in inv_args))
            else:
                self[args[0]] = inv(value)
        else:
            self._independent[key] = value
            self._dependent.pop(key, None)

    def _get_computed(self, key, env):
        """
        Get computed key from environment.
        """
        args, fn = self._dependent[key]
        for arg in args:
            if arg not in env:
                env[arg] = self.eval(arg, env)
        return fn(*(env[k] for k in args))

    def update(self, *args, **kwargs):
        """
        Update mapping with new associations.
        """
        if len(args) == 0:
            data = kwargs
        elif len(args) == 1:
            data = dict(args[0])
            data.update(kwargs)
        else:
            raise TypeError("update receive 0 or 1 positional arguments")

        pred = self._is_callable
        is_dep = lambda pair: pred(pair[1])
        deps, values = sk.separate(is_dep, data.items(), consume=True)

        values = dict(values)
        self._independent.update(values)
        for k in values:
            self._dependent.pop(k, None)

        deps = dict(deps)
        for k, fn in deps.items():
            self._dependent[k] = get_argnames(fn), fn
            self._independent.pop(k, None)

    def eval(self, key, env: MutableMapping):
        """
        Eval key function within the given environment.

        Args:
            key:
                The desired key.
            env:
                A mapping from key to evaluated values. Initialize with an empty
                dictionary to compute values from scratch.
        """
        try:
            return ChainMap(env, self._independent)[key]
        except KeyError:
            return self._get_computed(key, env)

    def get_keys(self, keys: Sequence[str], env=None, *, dependents: bool = False) -> dict:
        """
        Get several keys at once.

        Args:
            keys:
                A sequence of keys to be included.
            env:
                An optional evaluation environment.
            dependents:
                If true, also include dependent variables that were computed as
                intermediate steps during evaluation of keys. This is slightly
                faster that returning the dictionary with only the desired keys.
        """
        env = {} if env is None else env
        for k in keys:
            env[k] = self.eval(k, env)

        if dependents or len(env) == len(keys):
            return env
        return {k: env[k] for k in keys}

    def items(self, dependent=False):
        """
        Iterator over pairs of (key, value).
        """
        yield from self._independent.items()
        if dependent:
            env = {}
            for k in self._dependent:
                yield k, self.eval(k, env)

    def tagged_items(self):
        """
        Iterator over triples of (independent, key, value), in which dependent
        is a boolean telling if the key is independent or not.
        """
        for k, v in self._independent.items():
            yield True, k, v
        env = {}
        for k in self._dependent:
            yield False, k, self.eval(k, env)

    def values(self, dependent=False):
        """
        Iterator over values.
        """
        yield from self._independent.values()
        if dependent:
            env = {}
            for k in self._dependent:
                yield self.eval(k, env)

    def keys(self, dependent=False):
        """
        Iterator over keys. If dependent=True, also includes computed keys.
        """
        yield from self
        if dependent:
            yield from self._dependent

    def caching(self):
        """
        Return a caching view of the computed dict.

        The result is a mapping that stores all values in dict.
        """
        return MissingDict(self.__getitem__)

    def copy(self, **kwargs):
        """
        Return a new copy of computed dict, possibly overriding some keys.
        """
        new = object.__new__(type(self))
        new.__setstate__(self.__getstate__())
        new._dependent = self._dependent.copy()
        new._independent = self._independent.copy()
        if kwargs:
            new.update(kwargs)
        return new

    def partial(self, *args, **kwargs):
        """
        Create another computed dict that partially apply arguments to all
        computed keys in the dictionary.
        """
        new = self.copy()
        dependent = {}
        for k, (argnames, fn) in new._dependent.items():
            kwargs_ = {k: v for k, v in kwargs.items() if k in argnames}
            args_ = partial_args(argnames, args, kwargs_)
            fn_ = partial(fn, *args, **kwargs_)
            dependent[k] = args_, fn_
        return new


class BoundComputedDict(ComputedDict):
    """
    A computed dict in which computed keys are bound to the first argument.
    """

    __slots__ = ("object",)

    def __init__(self, obj, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object = obj

    def _get_computed(self, key, env):
        args, fn = self._dependent[key]
        for arg in args[1:]:
            if arg not in env:
                env[arg] = self.eval(arg, env)
        return fn(self.object, *(env[k] for k in args))


class DelayedArgsComputedDict(ComputedDict):
    """
    Like computed dict, but accepts functions that takes some specific
    parameters. It does not accept functions that mixes accepted with
    non-accepted arguments.
    """

    __slots__ = ("_delayed_arguments",)

    def __init__(self, *args, **kwargs):
        if not args:
            cls = type(self).__name__
            raise TypeError(f"{cls} requires at least one argument")
        delayed, *args = args
        self._delayed_arguments = frozenset(delayed)
        super().__init__(*args, **kwargs)

    def _is_callable(self, fn):
        if not callable(fn):
            return False

        args = set(sk.signature(fn).argnames())
        if self._delayed_arguments.issuperset(args):
            return False
        elif self._delayed_arguments.intersection(args):
            msg = f"function {fn} mixes delayed and non-delayed arguments: {args}"
            raise ValueError(msg)
        else:
            return True


class ProxyDict(MutableMapping):
    """
    Wraps an object into a dict and associate keys to method calls.
    """

    def __init__(self, obj, args=(), kwargs=None, fields=()):
        self._data = {}
        self._wrapped = obj
        self._args = tuple(args)
        self._kwargs = dict(kwargs or {})
        self._fields = set(fields)

    def __iter__(self):
        fields = self._fields
        yield from fields
        for k in self._data:
            if k not in fields:
                yield k

    def __getitem__(self, key):
        try:
            return self._data[key]
        except KeyError:
            if key in self._fields:
                fn = getattr(self._wrapped, key)
                value = self._data[key] = fn(*self._args, **self._kwargs)
                return value
            raise

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]
        self._fields.discard(key)

    def __len__(self):
        return len(self._fields.union(self._data))


class ComputedProxyDict(ComputedDict):
    """
    A computed dict subclass that falls back to a ProxyDict wrapping an
    object for missing keys. ProxyDict keys are treated as independent keys.
    """

    def __init__(self, obj, args=(), kwargs=None, fields=()):
        super().__init__()
        self._independent = ProxyDict(obj, args, kwargs, fields)
