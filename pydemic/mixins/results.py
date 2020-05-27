from collections import MutableMapping
from functools import lru_cache
from typing import TYPE_CHECKING, Iterator, Set

import pandas as pd

if TYPE_CHECKING:
    from .with_results import WithResultsMixin  # noqa: F401
    from ..models import Model  # noqa: F401


class Results(MutableMapping):
    """
    Results objects store dynamic information about a simulation.

    Results data is not necessarily static throughout the simulation. It can
    include values like total death toll, number of infected people, etc. It also
    distinguishes from information stored in the model mapping interface in
    that it does not include time series.

    Most information available as ``m[<param>]`` will also be available as
    ``m.results[<param>]``. While the first typically include the whole time
    series for the object, the second typically correspond to the last value
    in the time series.
    """

    __slots__ = ("owner",)
    owner: "Model"

    @property
    def _cache(self):
        model: "WithResultsMixin" = self.owner
        if model.iter != model._results_dirty_check:
            model._results_cache.clear()
            model._results_dirty_check = model.iter
        return model._results_cache

    def __init__(self, obj):
        self.owner = obj

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[str]:
        seen = set()

        prefix = "get_result_keys_"
        n = len(prefix)

        for method in dir(self.owner):
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

        if key:
            cache = cache[group]
            try:
                return cache[key]
            except KeyError:
                pass
            result = get_scalar_item(self.owner, group, key)

            # Save in cache only explicit keys
            # This avoid problems with aliases
            if key in get_dict_keys(self.owner, group):
                cache[key] = result
            return result

        else:
            out = cache[group]
            try:
                out.update(get_dict_item(self.owner, group, exclude=set(out)))
            except KeyError:
                if out:
                    return out
                raise
            return out

    def __setitem__(self, item, value):
        group, key = normalize_key(item)
        if key:
            self._cache[group][key] = value
        else:
            self._cache[group].update(value)

    def __delitem__(self, item):
        group, key = normalize_key(item)
        if key:
            del self._cache[group][key]
        else:
            del self._cache[group]

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
        raise NotImplementedError

    def _html_repr_(self):
        return self.to_frame()._html_repr_()


def extra_keys(model, name):
    """
    Collect extra keys implemented as _get_result_<name>__<key> methods.
    """

    prefix = f"get_result_value_{name}__"
    n = len(prefix)

    for attr in dir(model):
        if attr.startswith(prefix):
            yield attr[n:].replace("__", ".")


def get_scalar_item(model: "Model", group, key):
    """
    Fetch scalar result values from Model instance.
    """

    name = f"get_result_value_{group}"
    full_name = f"{name}__{key}"

    if hasattr(model, full_name):
        fn = getattr(model, full_name)
        return fn()

    try:
        fn = getattr(model, name)
    except AttributeError:
        cls = type(model).__name__
        raise KeyError(f"{cls} instance has no '{group}.{key}' result key.")
    else:
        return fn(key)


def get_dict_item(model, group, exclude: Set[str] = frozenset()):
    """
    Fetch dict result values from Model instance.
    """

    keys = get_dict_keys(model, group)
    results = model.results
    return {key: results[group, key] for key in keys - exclude}


@lru_cache(512)
def get_dict_keys(model, group):
    """
    Get unique result group keys for model.
    """

    name = f"get_result_keys_{group}"

    try:
        fn = getattr(model, name)
    except AttributeError:
        cls = type(model).__name__
        raise KeyError(f"{cls} instance has no '{group}' result group")
    else:
        keys = fn()
        extra = extra_keys(model, group)
        return {*keys, *extra}


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
