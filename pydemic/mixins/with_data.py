from abc import ABC
from typing import Callable, TYPE_CHECKING

import numpy as np
import pandas as pd
import sidekick as sk

from .data_transforms import DATA_TRANSFORMS, MODEL_TRANSFORMS

if TYPE_CHECKING:
    from .meta_info import Meta

ListLike = (list, np.ndarray, pd.Index)


class WithDataMixin(ABC):
    """
    Subclasses have a ".data" dataframe attribute that hold time-series
    information about a simulation.
    """

    meta: "Meta"
    data: pd.DataFrame
    times: pd.Index
    get_param: Callable[[str], float]

    def __init__(self, data=None):
        if data is not None:
            self.data = data

    def __getitem__(self, item):
        if isinstance(item, tuple):
            first, idx = item

            if isinstance(first, str):
                col, _, transform = first.rpartition(":")
                if col:
                    fn = self.data_transformer(transform)
                    return fn(col, idx)
                else:
                    col = transform
                    try:
                        return self.get_column(col, idx)
                    except ValueError:
                        raise KeyError(item)

            elif isinstance(first, ListLike):
                data = [self[col, idx] for col in first]
                return pd.concat(data, axis=1)

        elif isinstance(item, (str, *ListLike)):
            return self.__getitem__((item, None))

        elif isinstance(item, slice):
            raise NotImplementedError
        else:
            raise TypeError(f"invalid item: {item!r}")

    def get_column(self, name, idx):
        """
        Return a data column with the given name.
        """

        name = name.replace("-", "_")

        # Specialized data getter methods take precedence
        try:
            method = getattr(self, f"get_data_{name}")
        except AttributeError:
            pass
        else:
            return method(idx)

        # The next step is to check if requested column is in the state space
        name = self.meta.data_aliases.get(name, name)
        try:
            if idx is None:
                data = self.data[name]
            else:
                data = self.data[name].iloc[idx]
        except KeyError:
            pass
        else:
            return data.copy()

        # Finally, it may correspond to a parameter. We have two options. It may
        # be explicitly stored as a time-series or it may be computed implicitly
        # from the other parameters.
        if name in self.meta.params.all:
            x = self.get_param(name)
            times = self.times[idx or slice(None, None)]
            return pd.Series([x] * len(times), index=times, name=name)
        raise ValueError(f"invalid column: {name!r}")

    def data_transformer(self, name):
        """
        Return a transformer function that reads a column represented by name
        (usually calling self.get_data(col) and then transform the result by
        some operation.

        Default operations:
            dates, days, weeks, months, years:
                Convert index of the series or dataframe to days/weeks/months or
                years instead of dates.
            df, np:
                Force output to be a dataframe/numpy array, even when it has a
                single column.
            pp, ppc, p1k, p10k, p100k, p1m:
                Divide data and display it per population. The different names
                represent simple conversions: pp = data / population,
                pc = 100 * pp, p1k = 1000 pp, and so on.
            int, float, str:
                Coercion to integers, floats, or strings.
            round, round1, round2, round3:
                Rounding to 0, 1, 2, or 3 decimal places.
            human, pcfmt, p1kfmt:
                Human-friendly numeric formats.
        """

        if name in MODEL_TRANSFORMS:
            fn = MODEL_TRANSFORMS[name]
            return lambda col, idx: fn(self, col, idx)

        try:
            fn = DATA_TRANSFORMS[name]
            return lambda col, idx: fn(self[col], idx)

        except KeyError:
            raise ValueError(f"Invalid transform: {name}")


class WithDataModelMixin(WithDataMixin, ABC):
    """
    Mixin for simulation model subclasses with a data attribute.
    """

    def initialize(self):
        raise NotImplementedError

    @sk.lazy
    def data(self):
        self.initialize()
        return self.__dict__["data"]

    def get_data_population(self, idx):
        """
        Return current population.
        """
        return self.data.iloc[idx or slice(None, None)].sum(1)

    def get_data_N(self, idx):
        """
        Shorthand for get_data_population.
        """
        return self.get_data_population(idx)
