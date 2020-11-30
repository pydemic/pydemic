from collections import MutableMapping, ChainMap
from operator import itemgetter
from typing import Sequence

import sidekick.api as sk

from ..expr_compiler import compile_expr


def inverse(attr):
    """
    Declare field as an inverse transform of another field.
    """
    fn = eval(f"lambda {attr}: 1 / {attr}")
    fn._func_inverse_ = lambda x: 1 / x
    return fn


def transform(fn, inv=None, args=None):
    """
    Declare field as an inverse transform of another field.
    """
    if isinstance(fn, str):
        fn = compile_expr(fn)
    if isinstance(inv, str):
        inv = compile_expr(inv)

    if inv is not None:
        fn._func_inverse_ = inv
    if args is not None:
        fn._argnames_ = args
    return fn


def alias(attr):
    """
    Declare field as an alias of another field.
    """
    fn = eval(f"lambda {attr}: {attr}")
    fn._func_inverse_ = lambda x: x
    return fn


def _initial_values(cls):
    initial = {}
    for b in reversed(cls.__bases__):
        initial.update(getattr(b, "_initial", None) or ())
    return initial


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
        hints = getattr(cls, "__annotations__", None)
        if hints is not None:
            cls._initial = _initial_values(cls)
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
        return self._dependent, self._independent

    def __setstate__(self, state):
        self._dependent, self._independent = state

    def __getinitargs__(self):
        data = self._independent.copy()
        for k, (_, fn) in self._dependent.items():
            data[k] = fn
        return data

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

    def eval(self, key, env):
        """
        Eval key function within the given environment.

        Independent keys must be called with no argument.
        """
        try:
            return ChainMap(env, self._independent)[key]
        except KeyError:
            pass

        args, fn = self._dependent[key]
        for arg in args:
            if arg not in env:
                env[arg] = self.eval(arg, env)
        return fn(*(env[k] for k in args))

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

    def values(self, dependent=False):
        """
        Iterator over values.
        """
        yield from self._independent.values()
        if dependent:
            env = {}
            for k in self._dependent:
                yield self.eval(k, env)

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


#
# Utility functions
#
def get_argnames(fn):
    try:
        return fn._argnames_
    except AttributeError:
        return sk.signature(fn).argnames()
