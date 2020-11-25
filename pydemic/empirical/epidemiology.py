from abc import ABC
from copy import copy
from numbers import Number
from typing import TypeVar, Iterable

import numpy as np
import pandas as pd

from sidekick import api as sk
from . import functions
from .utils import cached_method
from .utils import from_region
from ..diseases import Disease, disease as get_disease
from ..params import get_param

T = TypeVar("T")
PERIOD_METHODS = frozenset({"incidence_rate", "period_prevalence"})
PARAM_METHODS = frozenset({"point_prevalence", "period_prevalence"})
VOID_METHODS = frozenset({"population", "new_cases", "cumulative_cases", "attack_rate"})


class Epidemiology(ABC):
    """
    Base interface (with default implementations) for classes that expose
    descriptive epidemiological curves and data.

    Epidemic information is exposed via methods, which all return pandas Series
    object with datetime or integer indexes.
    """

    __slots__ = ()

    #
    # Attributes
    #
    @sk.lazy
    def disease(self) -> Disease:
        """
        Disease associated with the given epidemic curves.s
        """
        return get_disease()

    @sk.lazy
    def infectious_period(self) -> float:
        """
        Infectious period used to compute incidence rates.
        """
        return self.disease.infectious_period()

    #
    # Source information
    #
    @cached_method
    def population(self) -> pd.Series:
        """
        Population dataframe evaluated at the same time points as the epidemic
        curves.
        """
        raise NotImplementedError

    @cached_method
    def new_cases(self) -> pd.Series:
        """
        Number of new cases between two successive observations.
        """
        return functions.new(self.cumulative_cases())

    @cached_method
    def cumulative_cases(self) -> pd.Series:
        """
        Cumulative number of cases.

        It can be either raw data, or corrected by ascertainment rate,
        seasonality bias, and other factors.
        """
        if type(self).new_cases is DescriptiveEpidemiology.new_cases:
            msg = "Subclasses must implement either cumulative_cases() or new_cases()"
            raise NotImplementedError(msg)
        new = self.new_cases()
        return pd.Series(np.add.accumulate(new), index=new.index)

    #
    # Incidence and prevalence
    #
    def attack_rate(self) -> pd.Series:
        """
        Return the fraction of the population that contracted the disease during
        the epidemic episode at each point in time.

        This is the (corrected) number of cases divided by population at the
        start of the interval.
        """
        return self.cumulative_cases() / self.population().iloc[0]

    def incidence_rate(self, period: int) -> pd.Series:
        """
        Number of new cases in the specified period as it evolves in time.

        Each date corresponds to the **end** point for each considered interval.
        In other words, each date counts the number of new cases in the interval
        [date - period, date].

        Incomplete periods produces an cumulative list of cases.
        """
        cases = self.new_cases().rolling(period, min_periods=1).sum()
        return cases / self.population()

    def point_prevalence(self, params=None, **kwargs) -> pd.Series:
        """
        Number of active cases in each point in time.

        In the language of compartmental models, this usually corresponds to the
        Infectious compartment.

        Args:
            params:
                Any object that can be passed to get_param to extract the
                `infectious_period` parameter. `infectious_period` can also
                be passed explicitly as a keyword argument.
        """

        new_cases = self.new_cases()
        gamma = 1 / self._get_param("infectious_period", params, **kwargs)
        N = len(new_cases)

        # We assume that new cases decay exponentially to a recovery. This means
        # that for the collection of cases created at day i, each successive day
        # will still persist a fraction of exp(-gamma * n) with n being the number
        # of days after onset.
        infectious = np.zeros(N)
        for i, x in enumerate(new_cases.values):
            infectious[i:] += x * np.exp(-gamma * np.arange(0, N - i))

        return pd.Series(infectious, index=new_cases.index)

    def period_prevalence(self, period, params=None, **kwargs) -> pd.Series:
        """
        Number of active cases in the specified period as it evolves in time.

        This is similar to incidence_rate(), but also counts ongoing cases rather
        than only new cases that appeared in each interval. It returns all cases
        that are/were active in the interval [date - period, date]

        Args:
            period:
                The considered period for prevalence calculation.
            params:
                Any object that can be passed to get_param to extract the
                `infectious_period` parameter. `infectious_period` can also
                be passed explicitly as a keyword argument.
        """

        new_cases = self.new_cases()
        gamma = 1 / self._get_param("infectious_period", params, **kwargs)
        N = len(new_cases)

        # We assume that new cases decay exponentially to a recovery. This means
        # that for the collection of cases created at day i, each successive day
        # will still persist a fraction of exp(-gamma * n) with n being the number
        # of days after onset.
        infectious = np.zeros(N)
        for i, x in enumerate(new_cases.values):
            values = np.maximum(np.arange(0, N - i) - (period - 1), 0)
            infectious[i:] += x * np.exp(-gamma * values)

        return pd.Series(infectious, index=new_cases.index)

    def dataframe(
        self, columns=None, periods: Iterable[Number] = None, params=None, **kwargs
    ) -> pd.DataFrame:
        """
        Return a DataFrame with information about different columns of data.

        Args:
            columns:
                List of columns to be displayed. If no columns are present,
                return all available information.
            periods:
                List or single numeric value with all considered periods.
            params:
                Override params in methods that allow it. Params must be a mapping
                or a namespace compatible with the get_param function.
            **kwargs:
                Override params in methods that allow it.

        Returns:
            A dataframe with the selected information.
        """
        if columns is None:
            columns = list(VOID_METHODS | (PARAM_METHODS - PERIOD_METHODS))
            if periods is not None:
                columns.extend(PERIOD_METHODS)

        data = {}
        for col in columns:
            method = getattr(self, col)
            options = {}
            if col in PARAM_METHODS:
                options["params"] = params
                options.update(kwargs)
            if col in PERIOD_METHODS:
                if isinstance(periods, Number):
                    data[col] = method(periods, **kwargs)
                elif periods is None:
                    continue
                else:
                    for period in periods:
                        data[f"{col}-{period}"] = method(period, **kwargs)
                continue
            data[col] = method(**kwargs)

        return pd.DataFrame(data)

    #
    # Auxiliary methods
    #
    def _get_param(self, name, params, **kwargs):
        """
        Auxiliary method used to fetch a parameter. Parameter can be passed
        explicitly as a keyword argument, in a param object or fallback to an
        instance attribute.
        """
        if name in kwargs:
            value = kwargs[name]
            if value is not None:
                return value

        if params is not None:
            value = get_param(name, params, None)
            if value is not None:
                return value

        return getattr(self, name)

    def _get_param_as_series(self, index, name, params, **kwargs) -> pd.Series:
        """
        Fetch param and convert it to a Series object if an scalar object is
        returned.
        """
        value = self._get_param(name, params, **kwargs)

        if isinstance(value, pd.Series):
            return value.reindex(index)
        elif isinstance(value, pd.DataFrame):
            return value.iloc[:, 0].reindex(index)
        elif isinstance(value, np.ndarray):
            return pd.Series(value, index=index)
        else:
            return pd.Series([value] * len(index), index=index)


class EpidemicBase(Epidemiology, ABC):
    """
    Common implementations for DescriptiveSeries and DescriptiveCurves.
    """

    def __init__(self, data, params=None, population=None):
        self._data = data
        self._population = self._get_param_as_series(
            self._data.index, "population", params, population=population
        )

    @cached_method
    def population(self) -> pd.Series:
        return self._population

    #
    # Data transformations
    #
    def trim_dates(self: T, start=..., end=...) -> T:
        """
        Trim data in the specified interval of dates.

        Ellipsis are interpreted as open intervals in both ends. Dates are
        anything that can be transformed by pd.to_datetime().

        Examples:
            >>> data.trim_dates('2020-05-01', ...)
        """
        dates = self._data.index
        if start is ...:
            start = dates.iloc[0]
        if end is ...:
            end = dates.iloc[-1]

        start, end = map(pd.to_datetime, (start, end))
        dates = dates[(dates >= start) & (dates <= end)]
        new = copy(self)
        new._data = self._data[dates]
        return new

    def moving_average(self: T, window) -> T:
        """
        Smooth data with a moving average with the given window size.

        Args:
            window:
                Number of points in each moving average window.

        Returns:
            Smoothed out moving average data.
        """
        new = copy(self)
        new._data = self._data.rolling(window).mean()
        return new


class EpidemicSeries(EpidemicBase):
    """
    Descriptive epidemiology based on a single series of cumulative cases.
    """

    __slots__ = ("_data", "_population")

    @classmethod
    def from_region(cls, region, params=None, *, deaths=False, **kwargs):
        """
        Initialize data extracting curve from region.
        """
        transform = lambda df: df["deaths" if deaths else "cases"]
        return from_region(cls, transform, region, params, **kwargs)

    def __init__(self, cases, params=None, *, cumulative=True, **kwargs):
        if not isinstance(cases, pd.Series):
            cases = cases["cases"]
        if not cumulative:
            cases = pd.Series(cases, index=cases.index)
        if not cumulative:
            cases = pd.Series(np.add.accumulate(cases), index=cases.index)
        super().__init__(cases, params, **kwargs)

    def cumulative_cases(self) -> pd.Series:
        return self._data


class EpidemicCurves(EpidemicBase):
    """
    Descriptive epidemiology based on a dataframe of cumulative cases/deaths.

    Attributes:
        cases:
            A descriptive curve with only the "cases" component.
        deaths:
            A descriptive curve with only the "deaths" component.
    """

    __slots__ = ("_data", "_population", "_cases", "_deaths")
    component_constructor = EpidemicSeries

    @classmethod
    def from_region(cls, region, params=None, **kwargs):
        """
        Initialize data extracting curve from region.
        """
        return from_region(cls, lambda x: x, region, params, **kwargs)

    @sk.lazy(name="_cases")
    def cases(self):
        return self.component_constructor(self._data["cases"], population=self._population)

    @sk.lazy(name="_deaths")
    def deaths(self):
        return self.component_constructor(self._data["deaths"], population=self._population)

    def __init__(self, cases, params=None, *, cumulative=True, population=None):
        data = cases[["cases", "deaths"]]
        if not cumulative:
            values = np.add.accumulate(data, axis=0)
            data = pd.DataFrame(values, index=data.index, columns=data.columns)

        super().__init__(data, params=params, population=population)

    def cumulative_cases(self) -> pd.Series:
        return self._data["cases"]
