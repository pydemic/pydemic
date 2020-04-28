import datetime
from abc import ABCMeta
from numbers import Number
from typing import Sequence, Callable, Union

import numpy as np
import pandas as pd
import sidekick as sk
from sidekick import placeholder as _

from .clinical_acessor import Clinical
from .. import utils
from ..packages import plt
from ..params import Params, Param, param as _param
from ..utils import today

NOW = datetime.datetime.now()
TODAY = datetime.date(NOW.year, NOW.month, NOW.day)
DAY = datetime.timedelta(days=1)
TIME_CONVERSIONS = {"days": 1, "weeks": 7, "months": 365.25 / 12, "years": 365.25}
RATIO_CONVERSIONS = {"pp": 1, "ppc": 100, "p1k": 1e3, "p10k": 1e4, "p100k": 1e5, "p1m": 1e6}
DATA_CONVERSIONS = {"int": int, "float": float, "str": str}
ELEMENTWISE_TRANSFORMS = {
    "round": lambda x: round(x),
    "round1": lambda x: round(x, 1),
    "round2": lambda x: round(x, 2),
    "round3": lambda x: round(x, 3),
    "human": utils.fmt,
    "pcfmt": utils.pc,
    "p1kfmt": utils.pm,
    "p10kfmt": utils.p10k,
    "p100kfmt": utils.p100k,
}
not_implemented = lambda *args: sk.error(NotImplementedError)
pplt = sk.import_later("..plot", package=__package__)


class ModelMeta(ABCMeta):
    """
    Metaclass for model classes.
    """

    DATA_ALIASES: dict
    _meta: "Meta"

    def __init__(cls, name, bases, ns):
        super().__init__(name, bases, ns)
        cls._meta = Meta(cls)


class Meta:
    """
    Meta information about model
    """

    cls: ModelMeta

    def __init__(self, cls):
        self.cls = cls

    @sk.lazy
    def component_index(self):
        cls = self.cls
        if hasattr(cls, "DATA_COLUMNS"):
            items = zip(cls.DATA_COLUMNS, cls.DATA_COLUMNS)
        else:
            items = cls.DATA_ALIASES.items()

        idx_map = {}
        for i, (k, v) in enumerate(items):
            idx_map[k] = idx_map[v] = i
        return idx_map

    @sk.lazy
    def data_columns(self):
        cls = self.cls
        try:
            return tuple(getattr(cls, "DATA_COLUMNS"))
        except AttributeError:
            return tuple(cls.DATA_ALIASES.values())

    @sk.lazy
    def primary_params(self):
        return [k for k, v in self._params() if not v.is_derived]

    @sk.lazy
    def derived_params(self):
        return [k for k, v in self._params() if v.is_derived]

    @sk.lazy
    def params(self):
        return self.primary_params + self.derived_params

    def _params(self):
        cls = self.cls
        for k in dir(cls):
            v = getattr(cls, k, None)
            if hasattr(v, "__get__") and getattr(v, "is_param", False):
                yield k, v


class Model(metaclass=ModelMeta):
    """
    Base class for all models.
    """

    # Constants
    DATA_ALIASES = {}

    # Initial values
    params: Params = sk.lazy(not_implemented)
    state: np.ndarray = sk.lazy(not_implemented)

    # Initial time
    date: datetime.date = None
    time: float = 0.0
    iter: int = sk.property(lambda m: len(m.data))
    dates: pd.DatetimeIndex = sk.property(lambda m: m.to_dates(m.times))
    times: pd.Index = sk.property(lambda m: m.data.index)

    # Common epidemiological parameters
    K = sk.property(not_implemented)
    duplication_time = sk.property(np.log(2) / _.K)

    # Cached properties
    initial_population = sk.property(lambda m: m.data.iloc[0].sum())

    # Special accessors
    clinical = property(Clinical)

    @classmethod
    def create(cls, params=None):
        new = object.__new__(cls)
        new.set_params(params)
        return new

    def __init__(self, params=None, *, run=None, name=None, date=None, **kwargs):
        self._params = {}
        self.set_params(self.params)
        if params:
            self.set_params(params)
        s1 = set(self._params)
        s2 = set(self._meta.primary_params)
        assert s1 == s2, f"Different param set: {s1} != {s2}"

        self.name = name or f"{type(self).__name__} model"
        self.date = pd.to_datetime(date or today())
        self.set_ic()
        self.data = make_dataframe(self)

        for k, v in kwargs.items():
            if k in self._meta.params:
                self.set_param(k, v)
            elif hasattr(self, k):
                setattr(self, k, v)
            else:
                raise TypeError(f"invalid argument: {k}")

        if run is not None:
            self.run(run)

    def __str__(self):
        return self.name

    #
    # Parameters
    #
    def set_params(self, params=None, **kwargs):
        """
        Set a collection of params.
        """
        if params:
            for p in self._meta.primary_params:
                self._params[p] = kwargs.pop(p) if p in kwargs else params.param(p)

        for k, v in kwargs:
            self.set_param(k, v)

        name = type(self).__name__
        self.params = Params(name, **self._params)

    def set_param(self, name, value, *, pdf=None, ref=None):
        """
        Sets a parameter in the model, possibly assigning a distribution and
        reference.
        """
        if name in self._meta.primary_params:
            self._params[name] = _param(value, pdf=pdf, ref=ref)
        elif name in self._meta.derived_params:
            setattr(self, name, _param(value).value)
        else:
            raise ValueError(f"{name} is an invalid param name")

    def get_param(self, name, param=False) -> Union[Number, Param]:
        """
        Return the parameter with given name.

        Args:
            name:
                Parameter name.
            param:
                If True, return a :cls:`Param` instance instead of a value.
        """
        if param:
            try:
                return self._params[name]
            except KeyError:
                param = self.get_param(name)
                return _param(param)
        try:
            return self._params[name].value
        except KeyError:
            pass
        if name in self._meta.derived_params:
            return getattr(self, name)
        else:
            raise ValueError(f"invalid parameter name: {name!r}")

    #
    # Initial conditions
    #
    def set_ic(self, state=None, **kwargs):
        """
        Set initial conditions.
        """
        if state is None:
            state = (0,) * max((0, *self._meta.component_index.values()))
        self.state = st = np.array(state, dtype=float)
        for k, v in kwargs.items():
            idx = self._meta.component_index(k)
            st[idx] = v

    #
    # Retrieving columns
    #
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
        name = self.DATA_ALIASES.get(name, name)
        try:
            data = self.data[name]
        except KeyError:
            pass
        else:
            return data.copy()

        try:
            method = getattr(self, f"get_data_{name}")
        except AttributeError:
            raise ValueError(f"invalid column: {name!r}")
        else:
            return method()

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

        # Convert index to days instead of dates
        if name == "dates":

            def dates_t(col):
                data = self.get_data(col)
                data.index = self.to_dates(data.index)
                return data

            return dates_t

        elif name in TIME_CONVERSIONS:

            def time_t(col):
                factor = TIME_CONVERSIONS[name]
                data = self.get_data(col)
                data.index = data.index / factor
                return data

            return time_t

        # Force columns to be data frames, even when results are vectors
        elif name == "df":

            def df_t(col):
                name_, _, _ = col.partition(":")
                series = self.get_data(col)
                if isinstance(series, pd.DataFrame):
                    return series
                return pd.DataFrame({name_: series.values}, index=series.index)

            return df_t

        # Force result to be numpy arrays
        elif name == "np":

            def np_t(col):
                return self.get_data(col).values

            return np_t

        # Per population
        if name in RATIO_CONVERSIONS:

            def ratio_t(col):
                factor = RATIO_CONVERSIONS[name]
                return factor * self.get_data(col) / self.initial_population

            return ratio_t

        # Data and rounding conversions
        elif name in DATA_CONVERSIONS:

            def data_t(col):
                kind = DATA_CONVERSIONS[name]
                return self.get_data(col).astype(kind)

            return data_t

        # Elementwise transforms
        elif name in ELEMENTWISE_TRANSFORMS:

            def rounding_t(col):
                fn = ELEMENTWISE_TRANSFORMS[name]
                return self.get_data(col).apply(fn)

            return rounding_t
        else:
            raise ValueError(f"Invalid transform: {name}")

    #
    # Running simulation
    #
    def run(self, time):
        """
        Runs the model for the given duration.
        """
        steps = int(time)
        _, *shape = self.data.shape

        ts = self.time + 1.0 + np.arange(steps)

        data = np.zeros((steps, *shape))
        date = self.date
        self.run_to_fill(data, ts)
        extra = pd.DataFrame(data, columns=self.data.columns, index=ts)

        self.data = pd.concat([self.data, extra])
        self.date = date + time * DAY
        self.time = ts[-1]
        self.state = data[-1]

    def run_to_fill(self, data, times):
        """
        Run simulation to fill pre-allocated array of data.
        """
        raise NotImplementedError

    def run_until(self, condition: Callable[["Model"], bool]):
        """
        Run until stop condition is satisfied.

        Args:
            condition:
                A function that receives a model and return True if stop
                criteria is satisfied.
        """
        raise NotImplementedError

    #
    # Utility methods
    #
    def to_dates(self, times: Sequence, start_date=None) -> pd.DatetimeIndex:
        """
        Convert an array of numerical times to dates.

        Args:
            times:
                Sequence of times.
            start_date:
                Starting date. If not given, uses the starting date for
                simulation.
        """
        dates: pd.DatetimeIndex

        if isinstance(times, pd.DatetimeIndex):
            return times
        if start_date is None:
            start_date = self.date - self.time * DAY

        dates = pd.to_datetime(times, unit="D", origin=start_date)
        return dates

    def to_days(self, dates: Sequence, start_date=None) -> np.ndarray:
        """
        Convert an array of numerical times to dates.

        Args:
            dates:
                Sequence of dates.
            start_date:
                Starting date. If not given, uses the starting date for
                simulation.
        """
        if start_date is None:
            start_date = self.date - self.time * DAY
        data = [(date - start_date).days for date in dates]
        return np.array(data) if data else np.array([], dtype=int)

    #
    # Plotting and showing information
    #
    def plot(self, components=None, *, ax=None, log=True, show=False, dates=False):
        """
        Plot the result of simulation.
        """
        ax = ax or plt.gca()
        kwargs = {"logy": log, "ax": ax}

        def get_column(col):
            if dates:
                col += ":dates"
            data = self[col]
            return data

        components = self.DATA_ALIASES.values() if components is None else components
        for col in components:
            data = get_column(col)
            data.plot(label=col.title(), **kwargs)

        if show:
            plt.show()


def make_dataframe(model: Model):
    """
    Create the initial dataframe for the given model.
    """
    data = [model.state]
    cols = model._meta.data_columns
    index = [model.time]
    return pd.DataFrame(data, columns=cols, index=index)
