from collections import Mapping
from typing import TYPE_CHECKING, Iterator

import pandas as pd

if TYPE_CHECKING:
    from .with_results import WithResultsMixin  # noqa: F421


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
        for method in dir(self.owner):
            if method.startswith("get_result_"):
                if "__" not in method:
                    yield method[11:]

    def __getitem__(self, item):
        cache = self._cache
        try:
            return cache[item]
        except KeyError:
            pass

        prefix, _, tail = item.partition(".")
        tail = tail.replace(".", "__")
        name = f"get_result_{prefix}"
        full_name = f"{name}__{tail}"
        try:
            method = getattr(self.owner, full_name)
        except AttributeError:
            try:
                method = getattr(self.owner, name)
            except AttributeError:
                cls = type(self.owner).__name__
                raise KeyError(f"{cls} object has no {item!r} result key.")
            else:
                result = method(tail or None)
        else:
            result = method()

        cache[item] = result
        return result

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
