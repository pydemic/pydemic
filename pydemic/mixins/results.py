from collections import Mapping
from typing import TYPE_CHECKING, Iterator, Sequence, Set

import pandas as pd

if TYPE_CHECKING:
    from .with_results import WithResultsMixin  # noqa: F421
    from ..models import Model  # noqa: F421


class Results(Mapping):
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
        owner: "WithResultsMixin" = self.owner
        if owner.iter != owner._results_dirty_check:
            owner._results_cache.clear()
        return owner._results_cache

    def __init__(self, obj):
        self.owner = obj

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[str]:
        prefix = "get_result_keys_"
        n = len(prefix)
        for method in dir(self.owner):
            if method.startswith(prefix) and "__" not in method:
                yield method[n:]

    def __getitem__(self, item):
        if isinstance(item, str):
            prefix, _, suffix = item.partition(".")
        elif isinstance(item, tuple):
            prefix, suffix = item
        else:
            cls_name = type(item).__name__
            raise TypeError(f"invalid key type: {cls_name}")

        cache = self._cache
        if suffix:
            cache = cache[prefix]
            try:
                return cache[suffix]
            except KeyError:
                pass
            cache[suffix] = result = get_scalar_item(self.owner, prefix, suffix)
            return result

        else:
            out = cache[prefix]
            out.update(get_dict_item(self.owner, prefix, exclude=set(out)))
            return out

    def to_dict(self, flat=False):
        """
        Convert data to nested dict.
        """
        data = {}
        for k, v in self.items():
            if flat:
                data.update({f"{k}.{ki}": vi for ki, vi in v})
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
    name = f"get_result_keys_{group}"

    try:
        fn = getattr(model, name)
    except AttributeError:
        cls = type(model).__name__
        raise KeyError(f"{cls} instance has no {group} result group")
    else:
        keys = fn()
        extra = extra_keys(model, group)
        keys = {*keys, *extra} - exclude

    results = model.results
    return {key: results[group, key] for key in keys}
