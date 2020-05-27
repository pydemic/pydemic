import numpy as np
import pandas as pd

from .. import plot as plt
from ..diseases import disease as get_disease
from .. import fitting as fit
from ..utils import trim_weeks, weekday_name, accumulate_weekly


def weekday_rates(
    data, *, trend=False, trend_style="k--", xtick_rotation=45, translate=None, **kwargs
):
    """
    Plot fractions of weekly values accumulated at each week day.

    Args:
        data:
            Data frame with time series data
        trend:
            If given, show a trend line with the expected value that should be
            encountered in each week day.
            * False (default): omit trend line
            * {True, 'cte'}: show a horizontal trend as if each week day should
              contribute the same amount to the weekly total
            * 'exp': Assumes an exponential growth;
        trend_style:
            If given, overrides the default style ('k--') for the trend line.
        xtick_rotation:
            Control rotation of x-ticks.

    Keyword Args:
        Additional keyword arguments are passed to the plot method of a
        weekly rates dataframe.
    """

    kwargs.setdefault("width", 0.75)

    data = trim_weeks(data)
    rate = fit.weekday_rate(data)
    std = rate[[col + "_std" for col in data.columns]]
    values = rate[data.columns]
    values.index = (weekday_name(dt, translate=translate) for dt in values.index)
    ax = values.plot(kind="bar", yerr=std.values.T, **kwargs)
    ax.tick_params("x", rotation=xtick_rotation)

    if trend == "exp":
        by_week = accumulate_weekly(data)
        growth, growth_std = fit.average_growth(fit.growth_factors(by_week))
        daily_growth = growth ** (1 / 7)

        X = np.arange(-0.5, 7.5)
        trend = daily_growth ** X
        trend /= trend.sum()
        ax.plot(X, trend, trend_style)

    elif trend == "cte" or trend is True:
        plt.mark_x(1 / 7, trend_style)

    elif trend is not False:
        raise ValueError("invalid trending method")

    return ax


def plot_weekday_rate_for_region(region, disease=None, **kwargs):
    """
    Like plot_weekly_rate, but cache results per-region.
    """

    disease = get_disease(disease)
    curves = disease.epidemic_curve(region)
    values = np.diff(curves, prepend=0, axis=0)

    data = pd.DataFrame(values, index=curves.index, columns=curves.columns)
    data = trim_weeks(data)

    return weekday_rates(data, **kwargs)
