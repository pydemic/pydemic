from abc import ABC

import pandas as pd
import sidekick as sk

from .data_transforms import DATA_TRANSFORMS, MODEL_TRANSFORMS


class WithDataMixin(ABC):
    """
    Subclasses have a ".data" dataframe attribute that hold time-series
    information about a simulation.
    """

    data: pd.DataFrame
    DATA_ALIASES: dict  # FIXME: remove this!

    def __init__(self, data=None):
        if data is not None:
            self.data = data

    def __getitem__(self, item):
        if isinstance(item, str):
            col, _, transform = item.rpartition(":")
            if col:
                fn = self.get_data_transformer(transform)
                return fn(col)
            else:
                try:
                    return self.get_data(transform)
                except ValueError:
                    raise KeyError(item)

        elif isinstance(item, list):
            df = pd.DataFrame()
            for col in item:
                series = self[col]
                name = col.partition(":")[0]
                df[name] = series
            return df

        elif isinstance(item, tuple):
            col, idx = item
            if ":" not in col:
                col = self.DATA_ALIASES.get(col, col)
            return self[col].iloc[idx]

        elif isinstance(item, slice):
            raise NotImplementedError
        else:
            raise TypeError(f"invalid item: {item!r}")

    def get_data(self, name):
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
            return method()

        # The next step is to check if requested column is in the state space
        name = self.DATA_ALIASES.get(name, name)
        try:
            data = self.data[name]
        except KeyError:
            pass
        else:
            return data.copy()

        # Finally, it may correspond to a parameter. We have two options. It may
        # be explicitly stored as a time-series or it may be computed implicitly
        # from the other parameters.
        if name in self._meta.params.all:
            x = self.get_param(name)
            return pd.Series([x] * self.iter, index=self.times, name=name)
        raise ValueError(f"invalid column: {name!r}")

    def get_data_transformer(self, name):
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
            return lambda col: fn(self, col)

        try:
            fn = DATA_TRANSFORMS[name]
            return lambda col: fn(self[col])

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

    def get_data_population(self):
        """
        Return current population.
        """
        return self.data.sum(1)
