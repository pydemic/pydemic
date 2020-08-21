from types import MappingProxyType
from typing import cast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from statsmodels import api as sm
from statsmodels.tsa.seasonal import seasonal_decompose, STL

from ..plot import mark_x, color
from ..types import ValueStd

id_ = lambda x: x


#
# Inference classes
#
class Trend:
    """
    Infer real cases from observed cases curve.
    """

    start_at: pd.Timestamp = property(lambda self: self.observed.index[0])
    end_at: pd.Timestamp = property(lambda self: self.observed.index[-1])
    is_linear: bool = True

    # Fitted parameters
    trend: pd.Series = None
    seasonal: pd.Series = None

    def __init__(self, method="linear", period=7):
        self.is_linear = normalize_trend_kind(method) == "linear"
        self.period = period

    def copy(self, deep=True) -> "Trend":
        """
        Return a copy of predictor.
        """
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)

        if True and self.trend is not None:
            new.trend = self.trend.copy()
            new.seasonal = self.seasonal.copy()
        return new

    def fit(self, X: pd.Series, y=None) -> "Trend":
        """
        Fit model with the given cases curve.
        """
        if not isinstance(X, pd.Series):
            kind = type(X).__name__
            raise TypeError(f"input data must be a Series, got {kind}")
        observed = X.copy()
        observed.name = "observed"
        observed.index = pd.to_datetime(observed.index)

        # Execute fitting according to method
        method = self._fit_linear if self.is_linear else self._fit_multiplicative
        self.trend, self.seasonal = method(observed)
        return self

    def fit_predict(self, X, future, **kwargs) -> pd.DataFrame:
        """
        Apply fit and predict methods in sequence.
        """
        return self.fit(X).predict(future, **kwargs)

    def _fit_linear(self, data):
        res = seasonal_decompose(data, period=self.period, extrapolate_trend="freq")
        return res.trend, res.seasonal

    def _fit_multiplicative(self, data):
        # total = data.sum()
        data = np.log(data.where(data > 0, float("nan"))).interpolate()
        # data += np.exp(data).sum() / total
        trend, seasonal = self._fit_linear(data)
        return np.exp(trend), np.exp(seasonal)

    def predict(self, future, trend=None) -> pd.DataFrame:
        """
        Return a data frame with "trend", "seasonal", and "pred" columns.

        Args:
            future:
                Specify the desired future dates. Can be an index or or array
                of timestamps, a series or single-column dataframe with a datetime
                index or an integer with the number of days o advance in the
                future.
            trend:
                If given, specify the trend component instead of inferring it
                from data. If "future" is a Series or DataFrame, it interprets
                the data part as the trend component and the index as the
                sequence of requested future dates.

        Returns:
            A DataFrame with "prediction", "trend" and "seasonal" components.
        """

        if isinstance(future, int):
            dates = self.future_dates(future)
        elif isinstance(future, pd.Series):
            trend = future.values
            dates = future.index
        elif isinstance(future, pd.DataFrame):
            trend = future.iloc[:, 0]
            dates = future.index
        elif isinstance(future, pd.Index):
            dates = future
        else:
            raise TypeError(f"invalid future: {future}")

        # Seasonal component
        days = (dates - self.start_at).map(lambda dt: dt.days)
        seasonal = self.seasonal.iloc[: self.period].reset_index(drop=True).values
        seasonal = pd.Series(seasonal[days % self.period], index=dates)

        # Trend component
        if trend is None:
            fn, fn_inv = (id_, id_) if self.is_linear else (np.log, np.exp)
            trend = extrapolate_trend(fn(self.trend), dates, points=2 * self.period)
            trend = fn_inv(trend)

        pred = self._prediction(trend, seasonal)
        return pd.DataFrame({"prediction": pred, "trend": trend, "seasonal": seasonal})

    def _prediction(self, trend, seasonal):
        if self.is_linear:
            return trend + seasonal
        else:
            return trend * seasonal

    def _residuals(self, observed, trend, seasonal):
        pred = self._prediction(trend, seasonal)
        if self.is_linear:
            return observed - pred
        else:
            return observed / pred

    def future_dates(self, periods=60) -> pd.DatetimeIndex:
        """
        Return an index with future dates from periods.
        """

        days = np.arange(1, periods + 1)
        out = pd.to_datetime(days, origin=self.end_at, unit="D")
        return cast(pd.DatetimeIndex, out)

    def plot(self, pred, ax=None, **kwargs):
        """
        Plot predictions against data.
        """

        ax: Axes = ax or plt.gca()
        self.plot_observations(ax=ax)

        pred["prediction"].plot(color=color(1, ax), ax=ax, **kwargs)
        pred["trend"].plot(color="k", ls="--", lw=2, ax=ax, **kwargs)

        mark_x(self.end_at, "k:")
        mark_x(self.start_at, "k:")
        ax.fill_betweenx(ax.get_ylim(), self.start_at, self.end_at, color="0.9")

        return ax.get_figure()

    def plot_components(self, observed=None, legend=True, **kwargs):
        """
        Plot inferred components of time-series.
        """

        fig, (ax, ax_res) = plt.subplots(2, 1, sharex="all")
        ax: Axes

        ax.set_title("Observations & predictions")
        self.plot_observations(observed, ax=ax, legend=legend, **kwargs)

        kwargs.pop("logy")
        if observed is not None:
            res = self._residuals(observed, self.trend, self.seasonal)
            res.plot(label="Residuals", marker="o", ax=ax_res, **kwargs)

        fig.tight_layout()
        return fig

    def plot_observations(self, observed=None, ax=None, **kwargs):
        """
        Plot observations with inferred trend line and seasonality.
        """

        ax = ax or plt.gca()
        pred = self._prediction(self.trend, self.seasonal)
        if observed is not None:
            observed.plot(label="Observed", lw=2, alpha=0.5, ax=ax, **kwargs)
        self.trend.plot(label="Trend", lw=2, color="k", ax=ax, **kwargs)
        pred.plot(label="Prediction", ax=ax, **kwargs)
        return ax


class STLTrend(Trend):
    """
    Uses a Season-Trend decomposition using LOESS based on
    :class:`statsmodels.tsa.seasonal.STL`
    """

    def __init__(self, method="linear", period=7, inner_iter=None, outer_iter=None, **kwargs):
        super().__init__(method, period)
        kwargs.setdefault("robust", True)
        self.params = kwargs
        self.inner_iter = inner_iter
        self.outer_iter = outer_iter

    def _fit_linear(self, data):
        stl = STL(data, self.period, **self.params)
        res = stl.fit()
        return res.trend, res.seasonal


#
# Utility functions
#
def normalize_trend_kind(
    kind,
    _kinds=MappingProxyType(
        {
            "mul": "multiplicative",
            "multiplicative": "multiplicative",
            "add": "linear",
            "linear": "linear",
        }
    ),
):
    try:
        return _kinds[kind.lower()]
    except KeyError:
        raise ValueError(f"invalid kind: {kind}")


# TODO: make this method more robust to different types of indexes and to work
# as some sort of replacement of Series.reindex() that extrapolate a trend to
# both sides of the observation interval
# noinspection PyTypeChecker
def extrapolate_trend(data: pd.Series, dates: pd.DatetimeIndex, points=7):
    """
    Extrapolate trend to the left and right of input time series using a OLS
    estimate of the derivative using the given number of points.

    Args:
        data:
            Input time series. Must be a Pandas Series object with a DatetimeIndex
            index.
        dates:
            An index with the new dates. If any (or all) of those values lies
            outside the observed range, return a linear extrapolation of the
            trend.
        points:
            Number of points used to estimate the derivative to the left and
            right of the observed range.

    Returns:
        A pd.Series re-indexed in all input dates with a linear extrapolation to
        the end points.
    """
    out = pd.Series([float("nan")] * len(dates), index=dates)
    index: pd.DatetimeIndex = data.index
    common_dates = dates & index
    if not common_dates.empty:
        out[common_dates] = data[common_dates]

    def fix_side(side, sel):
        idx = dates[sel]
        m, dm = estimate_derivative(data, side, points)
        loc = -1 if side == "right" else 0
        y0 = data.iloc[loc]
        out[idx] = (idx - index[loc]).map(lambda dt: y0 + dt.days * m)

    is_future: np.ndarray = dates > index[-1]
    if is_future.any():
        fix_side("right", is_future)

    is_past: np.ndarray = dates < index[0]
    if is_past.any():
        fix_side("left", is_future)

    return out


def estimate_derivative(data: pd.Series, where: str, points=10) -> ValueStd:
    """
    Uses OLS to estimate the derivative of a smoothed out version of the input
    time series selecting either 'left', 'right', or 'all' data points.

    Args:
        data (pd.DataFrame):
            Input time series.
        where ('left', 'right', 'all'):
            Which part of the series should be used to infer the derivative.
        points (int):
            If where in ('left', 'right'), how many points should be used to
            estimate the mean derivative.

    Return:
        A ValueStd instance with the (mean, std_error) of the derivative
        estimate.
    """
    if where == "right":
        Y = data.iloc[-points:]
    elif where == "left":
        Y = data.iloc[:points]
    else:
        Y = data

    X = (Y.index - Y.index[0]).map(lambda d: d.days)
    ols = sm.OLS(Y.values, sm.add_constant(X))
    res = ols.fit()
    _, m = res.params

    return ValueStd(m, 0.0)
