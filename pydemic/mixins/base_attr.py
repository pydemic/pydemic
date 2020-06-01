import weakref
from collections import MutableMapping
from functools import lru_cache
from typing import TYPE_CHECKING, Iterator, Set

import pandas as pd

if TYPE_CHECKING:
    from ..models import Model


class BaseAttr(MutableMapping):
    """
    Base class for Info and Results objects.
    """

    model: "Model"
    _method_namespace: str = None
    __slots__ = ("_model_ref",)

    @property
    def _cache(self):
        raise NotImplementedError

    @property
    def model(self):
        return self._model_ref()

    def __init__(self, obj):
        self._model_ref = weakref.ref(obj)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[str]:
        seen = set()
        prefix = f"get_{self._method_namespace}_keys_"
        n = len(prefix)

        for method in dir(self.model):
            if method.startswith(prefix) and "__" not in method:
                key = method[n:]
                if key not in seen:
                    yield key
                seen.add(key)

        for key in self._cache:
            if key not in seen:
                yield key

    def __getitem__(self, item):
        group, key = normalize_key(item)
        cache = self._cache

        namespace = self._method_namespace
        model = self.model

        if key:
            cache = cache[group]
            try:
                return cache[key]
            except KeyError:
                pass
            result = get_scalar_item(model, group, key, namespace)

            # Save in cache only explicit keys
            # This avoid problems with aliases
            if key in get_dict_keys(model, group, namespace):
                cache[key] = result
            return result

        else:
            out = cache[group]
            try:
                out.update(get_dict_item(model, group, namespace, exclude=set(out)))
            except KeyError:
                if out:
                    return out
                raise
            return out

    def __setitem__(self, item, value):
        group, key = normalize_key(item)
        if key:
            self._set_item(group, key, value)
        else:
            for k, v in value.items():
                self._set_item(group, k, v)

    def __delitem__(self, item):
        group, key = normalize_key(item)
        if key:
            del self._cache[group][key]
        else:
            del self._cache[group]

    def _set_item(self, group, key, value):
        # Can be overridden by subclasses
        self._cache[group][key] = value

    def to_dict(self, flat=False):
        """
        Convert data to nested dict.
        """
        data = {}
        for k, v in self.items():
            if flat:
                data.update({f"{k}.{ki}": vi for ki, vi in v.items()})
            else:
                data[k] = v
        return data

    def to_frame(self) -> pd.DataFrame:
        """
        Expose information about object as a dataframe.
        """
        data = self.to_series()
        return pd.DataFrame({self._method_namespace: data})

    def to_series(self) -> pd.Series:
        """
        Expose information about object as a dataframe.
        """
        return pd.Series(self.to_dict(flat=True))

    def _html_repr_(self):
        return self.to_frame()._html_repr_()


def normalize_key(key):
    """
    Return tuple of (group, key) from key.
    """

    if isinstance(key, str):
        group, _, key = key.partition(".")
    elif isinstance(key, tuple):
        group, key = key
    else:
        raise TypeError(f"invalid key type: {type(key).__class__}")
    return group, key or None


def extra_keys(model, name, namespace):
    """
    Collect extra keys implemented as get_{namespace}_{group}__{name} methods.
    """

    prefix = f"get_{namespace}_value_{name}__"
    n = len(prefix)

    for attr in dir(model):
        if attr.startswith(prefix):
            yield attr[n:].replace("__", ".")


def get_scalar_item(model: "Model", group, key, namespace):
    """
    Fetch scalar value from Model instance attribute.
    """
    assert isinstance(namespace, str)

    name = f"get_{namespace}_value_{group}"
    full_name = f"{name}__{key}"

    if hasattr(model, full_name):
        fn = getattr(model, full_name)
        return fn()

    try:
        fn = getattr(model, name)
    except AttributeError:
        cls = type(model).__name__
        raise KeyError(f"{cls} instance has no '{group}.{key}' {namespace} key.")
    else:
        return fn(key)


def get_dict_item(model, group, namespace, exclude: Set[str] = frozenset()):
    """
    Fetch values from Model instance.
    """

    keys = get_dict_keys(model, group, namespace)
    prop = getattr(model, namespace)
    return {key: prop[group, key] for key in keys - exclude}


@lru_cache(512)
def get_dict_keys(model, group, namespace):
    """
    Get unique group keys for model attribute.
    """

    name = f"get_{namespace}_keys_{group}"

    try:
        fn = getattr(model, name)
    except AttributeError:
        cls = type(model).__name__
        raise KeyError(f"{cls} instance has no '{group}' {namespace} group")
    else:
        keys = fn()
        extra = extra_keys(model, group, namespace)
        return {*keys, *extra}
