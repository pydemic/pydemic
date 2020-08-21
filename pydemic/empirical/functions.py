import pandas as pd
import numpy as np
import sidekick.api as sk
from .trend_inference import Trend, STLTrend


#
# Functional API
#
@sk.curry(1)
def extract_trend(
    data, method="linear", period=7, seasonal=False, stl=False, **kwargs
) -> pd.DataFrame:
    """
    Extract a linear trend of time-series data.

    Args:
        data:
            Input time-series data with a datetime index and a single column
            of real-valued data.
        method ('mul' or 'linear'):
            If mul, expect strictly positive data that increments by
            multiplicative increments.
        period:
            Natural periodicity expected in data. The default value of 7 assumes
            daily data with some weekly periodicity.
        seasonal:
            If True, append a "seasonal" column with the result of the seasonal
            component identified in data.
        stl:
            If True, uses a Season-Trend decomposition using LOESS based on
            :class:`statsmodels.tsa.seasonal.STL`

    Returns:
        A dataframe with a "trend" column with the identified trend.
    """

    cls = STLTrend if stl else Trend
    predictor = cls(method=method, period=period, **kwargs)
    predictor.fit(data)

    if seasonal:
        return pd.DataFrame({"trend": predictor.trend, "seasonal": predictor.seasonal})
    return pd.DataFrame({"trend": predictor.trend})


@sk.fn
def new(data: pd.Series) -> pd.Series:
    """
    Return daily cases from cumulative data.
    """
    new = data.diff()
    new.iloc[0] = data.iloc[0]
    return new


@sk.fn
def accumulate(data: pd.Series):
    """
    Return cumulative data from daily data of new cases.
    """
    return np.add.accumulate(data)


@sk.curry(1)
def fix_ascertaiment_rate(data: pd.Series, rate=None) -> pd.DataFrame:
    """
    Extract infectious curve from cases data.

    Args:
        data:
            Input dataframe with a cases and possibly deaths columns.
        rate:
            Optional fixed ascertaiment rate. If given, simply multiply input
            data to reach the desired ascertainment rate.
    """
    if rate is not None:
        return data["cases"] / rate
    raise NotImplementedError


@sk.curry(1)
def infectious(cases: pd.Series, seasonality=None, new_cases=False) -> pd.Series:
    """
    Extract infectious curve from cases data.
    """


def estimate_real_cases(
    curves: pd.DataFrame, params=None, method="CFR", min_notification=0.02
) -> pd.DataFrame:
    """
    Estimate the real number of cases from the cases and deaths curves.

    Returns:
        A new dataframe with the corrected "cases" and "deaths" columns.
    """

    if params is None:
        from . import disease

        params = disease().params()

    data = curves[["cases", "deaths"]]

    if method == "CFR":
        daily = data.diff().dropna()
        cases, deaths = daily[(daily != 0).all(axis=1)].values.T
        weights = np.sqrt(cases)
        empirical_CFR = (weights * deaths / cases).sum() / weights.sum()
        if np.isnan(empirical_CFR):
            empirical_CFR = 0.0

        try:
            CFR = params.CFR
        except AttributeError:
            CFR = params.disease_params.CFR

        data["cases"] *= min(max(empirical_CFR, CFR) / CFR, 1 / min_notification)
        return data
    else:
        raise ValueError(f"Invalid estimate method: {method!r}")
