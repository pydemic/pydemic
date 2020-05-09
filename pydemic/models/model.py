import datetime
from types import MappingProxyType
from typing import Sequence, Callable

import mundi_demography
import numpy as np
import pandas as pd
import sidekick as sk
from sidekick import placeholder as _

from .clinical_acessor import Clinical
from .data_transforms import DATA_TRANSFORMS, MODEL_TRANSFORMS
from .model_meta import ModelMeta
from ..packages import plt
from ..params import WithParams
from ..utils import today, not_implemented

NOW = datetime.datetime.now()
TODAY = datetime.date(NOW.year, NOW.month, NOW.day)
DAY = datetime.timedelta(days=1)
pplt = sk.import_later("..plot", package=__package__)


class Model(WithParams, metaclass=ModelMeta):
    """
    Base class for all models.
    """

    # Constants
    DATA_ALIASES = {}

    # Initial values
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
    clinical_model = None
    clinical_params = MappingProxyType({})

    @sk.lazy
    def age_distribution(self):
        if "age_pyramid" in self.__dict__:
            return self.age_pyramid.sum(1)
        try:
            region = self.region
        except RuntimeError:
            raise RuntimeError(
                "Could not determine the age_distribution for population.\n"
                "Model must be initialized with either an explicit age_distribution\n"
                "or with some specific region code."
            )
        else:
            return mundi_demography.age_distribution(region)

    @sk.lazy
    def age_pyramid(self):
        try:
            region = self.region
        except RuntimeError:
            raise RuntimeError(
                "Could not determine the age_distribution for population.\n"
                "Model must be initialized with either an explicit age_pyramid\n"
                "or with some specific region code."
            )
        else:
            data = mundi_demography.age_pyramid(region)
        if data.isna().values.all():
            col = self.age_distribution
            males = col // 2
            females = col - males
            return pd.DataFrame({"male": males, "female": females})
        else:
            return data

    @sk.lazy
    def region(self):
        msg = "Model must be initialized with an explicit region code."
        raise RuntimeError(msg)

    @classmethod
    def create(cls, params=None):
        new = object.__new__(cls)
        new.set_params(params)
        return new

    def __init__(self, params=None, *, run=None, name=None, date=None, clinical=None, **kwargs):
        WithParams.__init__(self, params)
        self.name = name or f"{type(self).__name__} model"
        self.date = pd.to_datetime(date or today())
        self.set_ic()
        self.data = make_dataframe(self)

        if clinical:
            clinical = dict(clinical)
            self.clinical_model = clinical.pop("model", None)
            self.clinical_params = clinical

        for k, v in kwargs.items():
            cls = type(self)
            if k in self._meta.params.all:
                self.set_param(k, v)
            elif hasattr(cls, k):
                setattr(self, k, v)
            else:
                raise TypeError(f"invalid argument: {k}")

        if run is not None:
            self.run(run)

    def __str__(self):
        return self.name

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

    def get_data_population(self):
        """
        Return current population.
        """
        return self.data.sum(1)

    def get_info(self, info):
        """
        Query information about model.
        """
        name, _, args = info.partition(":")
        try:
            method = getattr(self, f"get_info_{name}")
        except AttributeError:
            raise ValueError(f"invalid info name: {name}")
        return method(args)

    def get_info_demography(self, arg):
        """
        Retrieve demographic parameters about the population.
        """
        if arg == "population":
            return self.data.iloc[0].sum()
        elif arg == "age_distribution":
            return self.age_distribution
        elif arg == "age_pyramid":
            return self.age_pyramid
        elif arg == "seniors":
            return self.age_distribution.loc[60:].sum()
        else:
            raise ValueError(f"unknown argument: {arg!r}")

    def get_info_demography(self, arg):
        """
        Retrieve demographic parameters about the population.
        """
        if arg == "population":
            return self.data.iloc[0].sum()
        elif arg == "age_distribution":
            return self.age_distribution
        elif arg == "age_pyramid":
            return self.age_pyramid
        elif arg == "seniors":
            return self.age_distribution.loc[60:].sum()
        else:
            raise ValueError(f"unknown argument: {arg!r}")

    def get_info_healthcare(self, arg):
        """
        Return info about the healthcare system.
        """
        if arg == "icu_total_capacity":
            return 4 * self.icu_capacity
        elif arg == "hospital_total_capacity":
            return 4 * self.hospital_capacity
        raise NotImplementedError

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

        return pd.to_datetime(times, unit="D", origin=start_date)

    def to_date(self, time: float) -> pd.Timestamp:
        """
        Convert a single instant to the corresponding datetime
        """
        return pd.to_datetime(time - self.time, unit="D", origin=self.date)

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
    def plot(
        self,
        components=None,
        *,
        ax=None,
        log=False,
        show=False,
        dates=False,
        legend=True,
        grid=True,
    ):
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
            data.plot(label=col.title().replace("-", " ").replace("_", " "), **kwargs)
        if legend:
            ax.legend()
        if grid:
            ax.grid()

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
