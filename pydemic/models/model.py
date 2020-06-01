import datetime
from copy import copy
from types import MappingProxyType
from typing import Sequence, Callable, Mapping, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import sidekick as sk

from .clinical_acessor import Clinical
from .metaclass import ModelMeta
from .. import fitting as fit
from .. import formulas
from ..diseases import Disease, DiseaseParams, disease as get_disease
from ..logging import log
from ..mixins import (
    Meta,
    WithParamsMixin,
    WithDataModelMixin,
    WithInfoMixin,
    WithResultsMixin,
    WithRegionDemography,
)
from ..packages import plt
from ..utils import today, not_implemented, extract_keys, param_property

NOW = datetime.datetime.now()
TODAY = datetime.date(NOW.year, NOW.month, NOW.day)
DAY = datetime.timedelta(days=1)
pplt = sk.import_later("..plot", package=__package__)

if TYPE_CHECKING:
    from ..model_group import ModelGroup
    from pydemic_ui.model import UIProperty


class Model(
    WithDataModelMixin,
    WithInfoMixin,
    WithResultsMixin,
    WithParamsMixin,
    WithRegionDemography,
    metaclass=ModelMeta,
):
    """
    Base class for all models.
    """

    meta: Meta

    class Meta:
        model_name = "Model"
        data_aliases = {}

    # Initial values
    state: np.ndarray = None
    initial_cases: float = sk.lazy(lambda self: self._initial_cases())
    initial_infected: float = sk.lazy(lambda self: self._initial_infected())

    # Initial time
    date: datetime.date = None
    time: float = 0.0
    iter: int = sk.property(lambda m: len(m.data))
    dates: pd.DatetimeIndex = sk.property(lambda m: m.to_dates(m.times))
    times: pd.Index = sk.property(lambda m: m.data.index)

    # Common epidemiological parameters
    R0: float = param_property("R0", default=2.0)
    K = sk.property(not_implemented)
    duplication_time = property(lambda self: np.log(2) / self.K)

    # Special accessors
    clinical: Clinical = property(lambda self: Clinical(self))
    clinical_model: type = None
    clinical_params: Mapping = MappingProxyType({})
    disease: Disease = None
    disease_params: DiseaseParams

    @property
    def ui(self) -> "UIProperty":
        try:
            from pydemic_ui.model import UIProperty
        except ImportError as ex:
            log.warn(f"Could not import pydemic_ui.model: {ex}")
            msg = (
                "must have pydemic-ui installed to access the model.ui attribute.\n"
                "Please 'pip install pydemic-ui' before proceeding'"
            )
            raise RuntimeError(msg)
        return UIProperty(self)

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
        self.disease = get_disease(disease)
        self._initialized = False

        # Fix demography
        demography_opts = WithRegionDemography._init_from_dict(self, kwargs)
        self.disease_params = self.disease.params(**demography_opts)

        # Init other mixins
        WithParamsMixin.__init__(self, params, keywords=kwargs)
        WithInfoMixin.__init__(self)
        WithResultsMixin.__init__(self)
        WithDataModelMixin.__init__(self)

        if clinical:
            clinical = dict(clinical)
            self.clinical_model = clinical.pop("model", None)
            self.clinical_params = clinical

        for k, v in kwargs.items():
            if hasattr(self, k):
                try:
                    setattr(self, k, v)
                except AttributeError:
                    name = type(self).__name__
                    msg = f"cannot set '{k}' attribute in '{name}' model"
                    raise AttributeError(msg)
            else:
                raise TypeError(f"invalid arguments: {k}")

        if run is not None:
            self.run(run)

    def __str__(self):
        return self.name

    def _initial_cases(self):
        raise NotImplementedError("must be implemented in subclass")

    def _initial_infected(self):
        raise NotImplementedError("must be implemented in subclass")

    #
    # Pickling and copying
    #
    # noinspection PyUnresolvedReferences
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
        data.pop("_results_cache")

        new = object.__new__(cls)
        for k in list(kwargs):
            if k in data:
                data[k] = kwargs.pop(k)

        new._params = copy(params)
        new._results_cache = {}
        new.__dict__.update(copy(data))

        for k, v in kwargs.items():
            setattr(new, k, v)

        return new

    def split(self, n=None, **kwargs) -> "ModelGroup":
        """
        Create n copies of model, each one may override a different set of
        parameters and return a ModelGroup.

        Args:
            n:
                Number of copies in the resulting list. It can also be a sequence
                of dictionaries with arguments to pass to the .copy() constructor.

        Keyword Args:
            Keyword arguments are passed to the `.copy()` method of the model. If
            the keyword is a sequence, it applies the n-th component of the sequence
            to the corresponding n-th model.
        """

        from ..model_group import ModelGroup

        if n is None:
            for k, v in kwargs.items():
                if not isinstance(v, str) and isinstance(v, Sequence):
                    n = len(v)
                    break
            else:
                raise TypeError("cannot determine the group size from arguments")

        if isinstance(n, int):
            options = [{} for _ in range(n)]
        else:
            options = [dict(d) for d in n]
        n: int = len(options)

        # Merge option dicts
        for k, v in kwargs.items():
            if not isinstance(v, str) and isinstance(v, Sequence):
                xs = v
                m = len(xs)
                if m != n:
                    raise ValueError(
                        f"sizes do not match: "
                        f"{k} should be a sequence of {n} "
                        f"items, got {m}"
                    )
                for opt, x in zip(options, xs):
                    opt.setdefault(k, x)
            else:
                for opt in options:
                    opt.setdefault(k, v)

        # Fix name
        for opt in options:
            try:
                name = opt["name"]
            except KeyError:
                pass
            else:
                opt["name"] = name.format(n=n, **opt)

        return ModelGroup(self.copy(**opt) for opt in options)

    def split_children(self, options=MappingProxyType({}), **kwargs) -> "ModelGroup":
        """
        Similar to split, but split into the children of the given class.

        Args:
            options:
                A mapping between region or region id
        """

        from ..model_group import ModelGroup

        if self.region is None:
            raise ValueError("model is not bound to a region")

        for k in self._params:
            if k not in kwargs:
                kwargs[k] = self.get_param(k)

        for attr in ("disease",):
            kwargs.setdefault(attr, getattr(self, attr))

        return ModelGroup.from_children(self.region, type(self), options, **kwargs)

    def reset(self, date: Union[datetime.date, float] = None, **kwargs):
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
        elif isinstance(date, (float, int)):
            time = float(date)
            date = self.to_date(date)
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

    def epidemic_model_name(self):
        """
        Return the epidemic model name.
        """
        return self.meta.model_name

    #
    # Initial conditions
    #
    def set_ic(self, state=None, **kwargs):
        """
        Set initial conditions.
        """
        if self.state is None:
            if state is None:
                state = self.initial_state(**kwargs)
            self.state = np.array(state, dtype=float)

        alias = self.meta.data_aliases
        for k, v in list(kwargs.items()):
            if k in alias:
                del kwargs[k]
                kwargs[alias[k]] = v
        components = extract_keys(self.meta.variables, kwargs)

        for k, v in components.items():
            idx = self.meta.get_variable_index(k)
            self.state[idx] = v

    def set_data(self, data):
        """
        Force a dataframe into simulation state.
        """
        data = data.copy()
        data.columns = [self.meta.data_aliases.get(c, c) for c in data.columns]

        self.set_ic(state=data.iloc[0])
        self.data = data.reset_index(drop=True)
        self.time = len(data) - 1
        self.date = data.index[-1]
        self.state[:] = data.iloc[-1]
        self.info["observed.dates"] = data.index[[0, -1]]
        self._initialized = True

    def set_cases(self, curves=None, adjust_R0=False, save_cases=False):
        """
        Initialize model from a dataframe with the deaths and cases curve.

        This curve is usually the output of disease.epidemic_curve(region), and is
        automatically retrieved if not passed explicitly and the region of the model
        is set.

        Args:
            curves:
                Dataframe with cumulative ["cases", "deaths"] columns. If not given,
                or None, fetches from disease.epidemic_curves(info)
            adjust_R0:
                If true, adjust R0 from the observed cases.
            save_cases:
                If true, save the cases curves into the model.info["observed.cases"] key.
        """

        if curves is None:
            if self.region is None or self.disease is None:
                msg = 'must provide both "region" and "disease" or an explicit cases ' "curve."
                raise ValueError(msg)
            curves = self.region.pydemic.epidemic_curve(self.disease, real=True)

        if adjust_R0:
            method = "RollingOLS" if adjust_R0 is True else adjust_R0
            Re, _ = value = fit.estimate_R0(self, curves, Re=True, method=method)
            assert np.isfinite(Re), f"invalid value for R0: {value}"
            self.R0 = Re

        # Save notification it in the info dictionary for reference
        if "cases_observed" in curves:
            tf = curves.index[-1]
            rate = curves.loc[tf, "cases_observed"] / curves.loc[tf, "cases"]
        else:
            rate = 1.0
        self.info["observed.notification_rate"] = rate

        # Save simulation state from data
        model = self.epidemic_model_name()
        curve = fit.cases(curves)
        data = fit.epidemic_curve(model, curve, self)
        self.set_data(data)
        try:
            pred_cases = self["cases"]
        except KeyError:
            log.warn("Could not determine model cases")
        else:
            self.initial_cases -= pred_cases.iloc[-1] - curve.iloc[-1] + 1000

        if adjust_R0:
            self.R0 /= self["susceptible:final"] / self.population
            self.info["observed.R0"] = self.R0

        # Optionally save cases curves into the info dictionary
        if save_cases:
            key = "observed.cases" if save_cases is True else save_cases
            df = curves.copy()
            df["cases"] = curve
            df["cases_raw"] = curves["cases"]
            self.info[key] = df

    def initial_state(self, cases=None, **kwargs):
        """
        Create the default initial vector for model.
        """
        if cases is not None:
            kwargs.setdefault("population", self.population)
            return formulas.initial_state(self.epidemic_model_name(), cases, self, **kwargs)
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

        if self.info.get("event.simulation_start") is None:
            self.info.save_event("simulation_start")

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
    def to_dates(self, times: Sequence[int], start_date=None) -> pd.DatetimeIndex:
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

        # noinspection PyTypeChecker
        return pd.to_datetime(times, unit="D", origin=start_date)

    def to_date(self, time: Union[float, int]) -> datetime.date:
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

    def get_times(self, idx=None):
        """
        Get times possibly sliced by an index.
        """
        if idx is None:
            return self.times
        else:
            return self.times[idx]

    def get_data_cases(self, idx):
        raise NotImplementedError

    #
    # Plotting and showing information
    #
    def plot(
        self,
        components=None,
        *,
        ax=None,
        logy=False,
        show=False,
        dates=False,
        legend=True,
        grid=True,
    ):
        """
        Plot the result of simulation.
        """
        ax = ax or plt.gca()
        kwargs = {"logy": logy, "ax": ax, "grid": grid, "legend": legend}

        def get_column(col):
            if dates:
                col += ":dates"
            data = self[col]
            return data

        components = self.meta.variables if components is None else components
        for col in components:
            data = get_column(col)
            data.plot(**kwargs)
        if show:
            plt.show()


def make_dataframe(model: Model):
    """
    Create the initial dataframe for the given model.
    """
    data = [model.state]
    cols = model.meta.variables
    index = [model.time]
    return pd.DataFrame(data, columns=cols, index=index)
