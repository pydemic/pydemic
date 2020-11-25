from typing import Mapping, MutableMapping

import sidekick.api as sk


class Params(Mapping):
    """
    Represent a dictionary of parameters.
    """

    __slots__ = ("_data", "_initializing")

    _keys = ()
    _all_keys = ()

    def __init_subclass__(cls):
        bases = (t for t in cls.__bases__ if issubclass(t, Params))
        base_params = sk.concat((b._all_keys for b in bases))
        annotations = getattr(cls, "__annotations__", {})
        all_keys = {*annotations, *base_params}
        keys = [*filter(cls._is_main_param, all_keys)]

        cls._all_keys = tuple(all_keys)
        cls._keys = tuple(keys)

    @classmethod
    def _is_main_param(cls, name):
        return getattr(getattr(cls, name), "is_param", False)

    def __init__(self, data=(), **kwargs):
        self._data = {}
        self._mutable = True
        for k, v in dict(data, **kwargs).items():
            setattr(self, k, v)
        self._mutable = False

    def __getitem__(self, key):
        try:
            return self._data[key]
        except KeyError:
            if key not in self._all_keys:
                raise
        try:
            self._data[key] = value = getattr(self, key)
            return value
        except AttributeError:
            raise KeyError(key)

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __contains__(self, key):
        return key in self._all_keys

    def __repr__(self):
        name = type(self).__name__
        return f"{name}({dict(self)})"

    def copy(self, **kwargs):
        """
        Create copy possibly changing the values of some parameters
        """
        return type(self)(self, **kwargs)


class MutableParams(Params, MutableMapping):
    """
    A mutable version of params.
    """

    __slots__ = ()

    def __init__(self, data=(), **kwargs):
        super().__init__(data, **kwargs)
        self._mutable = True

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        if key not in self._all_keys:
            raise KeyError
        elif key in self._data:
            del self._data[key]


class Proxy(Params):
    """
    Wraps an object and treats its methods as lazy initializers for param
    fields.
    """

    def __init__(self, obj, data=(), args=(), kwargs=None, extra=()):
        super().__init__(data)
        self._wrapped = obj
        self._args = tuple(args)
        self._kwargs = dict(kwargs or {})
        self._extra = set(extra)
        self._keys = {*type(self)._keys, *self._extra}
        self._all_keys = {*type(self)._all_keys, *self._extra}

    def __getitem__(self, key):
        try:
            return self._data[key]
        except KeyError:
            if key not in self._all_keys:
                raise

        if key in self._extra:
            return self._get_value(key)

        try:
            self._data[key] = value = getattr(self, key)
            return value
        except AttributeError:
            raise KeyError(key)

    def __getattr__(self, attr):
        if attr in self._extra:
            return self._get_value(attr)
        raise AttributeError(attr)

    def _get_value(self, key):
        fn = get_attr_or_item(self._wrapped, key)
        value = fn(*self._args, **self._kwargs) if callable(fn) else fn
        self._data[key] = value
        return value


def get_attr_or_item(obj, key):
    if hasattr(obj, "__getitem__"):
        try:
            return obj[key]
        except KeyError:
            pass

    return getattr(obj, key)
