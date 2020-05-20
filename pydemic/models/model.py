import datetime
from copy import copy
from types import MappingProxyType
from typing import Sequence, Callable, Mapping

import numpy as np
import pandas as pd
import sidekick as sk
from sidekick import placeholder as _

from .clinical_acessor import Clinical
from .model_meta import ModelMeta
from .. import formulas
from ..diseases import Disease
from ..mixins import (
    WithParamsMixin,
    WithDataModelMixin,
    WithInfoMixin,
    WithSummaryMixin,
    WithRegionDemography,
)
from ..packages import plt
from ..utils import today, not_implemented, extract_keys

NOW = datetime.datetime.now()
TODAY = datetime.date(NOW.year, NOW.month, NOW.day)
DAY = datetime.timedelta(days=1)
pplt = sk.import_later("..plot", package=__package__)


class Model(
    WithDataModelMixin,
    WithInfoMixin,
    WithSummaryMixin,
    WithParamsMixin,
    WithRegionDemography,
    metaclass=ModelMeta,
):
    """
    Base class for all models.
    """

    # Constants
    DATA_ALIASES = {}
    model_name = "Model"

    # Initial values
    state: np.ndarray = None

    # Initial time
    date: datetime.date = None
    time: float = 0.0
    iter: int = sk.property(lambda m: len(m.data))
    dates: pd.DatetimeIndex = sk.property(lambda m: m.to_dates(m.times))
    times: pd.Index = sk.property(lambda m: m.data.index)

    # Common epidemiological parameters
    K = sk.property(not_implemented)
    duplication_time = sk.property(np.log(2) / _.K)

    # Special accessors
    clinical: Clinical = property(lambda self: Clinical(self))
    clinical_model: type = None
    clinical_params: Mapping = MappingProxyType({})
    disease: Disease = None

    @classmethod
    def create(cls, params=None):
        new = object.__new__(cls)
        new.set_params(params)
        return new

    def __init__(
        self, params=None, *, run=None, name=None, date=None, clinical=None, disease=None, **kwargs
    ):
        self.name = name or f"{type(self).__name__} model"
        self.date = pd.to_datetime(date or today())
        self.disease = disease
        self._initialized = False

        demography_opts = WithRegionDemography._init_from_dict(self, kwargs)
        WithParamsMixin.__init__(self, params, disease, keywords=kwargs)
        WithInfoMixin.__init__(self)
        WithSummaryMixin.__init__(self)
        WithDataModelMixin.__init__(self)

        if disease is None:
            self.disease_params = sk.record({})
        else:
            self.disease_params = self.disease.params(**demography_opts)

        if clinical:
            clinical = dict(clinical)
            self.clinical_model = clinical.pop("model", None)
            self.clinical_params = clinical

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise TypeError(f"invalid arguments: {k}")

        if run is not None:
            self.run(run)

    def __str__(self):
        return self.name

    #
    # Pickling and copying
    #
    def copy(self, **kwargs):
        """
        Copy instance possibly setting new values for attributes.

        Keyword Args:
            All keyword arguments are used to reset attributes in the copy.

        Examples:
            >>> m.copy(R0=1.0, name="Stable")
            <SIR(name="Stable")>
        """

        cls = type(self)
        data = self.__dict__.copy()
        params = data.pop("_params")
        results_data = data.pop("_results_data")

        new = object.__new__(cls)
        for k in list(kwargs):
            if k in data:
                data[k] = kwargs.pop(k)

        new._params = copy(params)
        new._results_data = {k: copy(v) for k, v in results_data}
        new.__dict__.update(copy(data))

        for k, v in kwargs.items():
            setattr(new, k, v)

        return new

    def reset_data(self, date=None, **kwargs):
        """
        Return a copy of the model setting the state to the final state. If a
        positional "date" argument is given, reset to the state to the one in the
        specified date.

        Args:
            date (float or date):
                An optional float or datetime selecting the desired date.

        Keyword Args:
            Additional keyword arguments are handled the same way as the
            :method:`copy` method.
        """
        if date is None:
            date = self.date
            time = self.time

        if isinstance(date, (float, int)):
            time = date
            date: pd.datetime = self.to_date(date)
        else:
            time: float = self.to_time(date)

        kwargs["data"] = self.data.loc[[time]]
        kwargs["date"] = date
        kwargs["state"] = kwargs["data"].iloc[0].values
        kwargs["time"] = 1
        return self.copy(**kwargs)

    def trim_dates(self, start=0, end=None):
        """
        Trim data in model to the given interval specified by start and end
        dates or times.

        Args:
            start (int or date):
                Starting date. If not given, start at zero.
            end (int or date):
                End date. If not given, select up to the final date.
        """
        start = int(start or 0)
        end = int(end or self.time)
        new = self.copy(
            date=self.to_date(start),
            data=self.data.iloc[start:end].reset_index(drop=True),
            time=end - start,
            state=self.data.iloc[end].values,
        )
        return new

    #
    # Initial conditions
    #
    def set_ic(self, state=None, **kwargs):
        """
        Set initial conditions.
        """
        keys = {*self.DATA_ALIASES, *self.DATA_ALIASES.values()}
        components = extract_keys(keys, kwargs)

        if self.state is None:
            if state is None:
                state = self.initial_state(**kwargs)
            self.state = np.array(state, dtype=float)

        for k, v in components.items():
            idx = self._meta.component_index(k)
            self.state[idx] = v

    def set_data(self, data):
        """
        Force a dataframe into simulation state.
        """
        data = data.copy()
        data.columns = [self.DATA_ALIASES.get(c, c) for c in data.columns]

        self.set_ic(state=data.iloc[0])
        self.data = data.reset_index(drop=True)
        self.time = len(data) - 1
        self.date = data.index[-1]
        self.state[:] = data.iloc[-1]
        self._initialized = True

    def initial_state(self, cases=None, **kwargs):
        """
        Create the default initial vector for model.
        """
        if cases is not None:
            kwargs.setdefault("population", self.population)
            return formulas.initial_state(self.model_name, cases, self, **kwargs)
        return self._initial_state()

    def _initial_state(self):
        raise NotImplementedError

    def initialize(self):
        """
        Force initialization.
        """
        if not self._initialized:
            self.set_ic()
            self.data = make_dataframe(self)
        self._initialized = True

    #
    # Running simulation
    #
    def run(self, time):
        """
        Runs the model for the given duration.
        """
        steps = int(time)
        self.initialize()

        if time == 0:
            return

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
    def to_dates(self, times: Sequence, start_date=None) -> Sequence[datetime.date]:
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

    def to_date(self, time: float) -> datetime.date:
        """
        Convert a single instant to the corresponding datetime
        """
        return pd.to_datetime(time - self.time, unit="D", origin=self.date)

    def to_times(self, dates: Sequence, start_date=None) -> np.ndarray:
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

    def to_time(self, date, start_date=None) -> float:
        """
        Convert date to time.
        """
        if start_date is None:
            return self.to_time(date, self.date) - self.time
        return float((date - start_date).days)

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
