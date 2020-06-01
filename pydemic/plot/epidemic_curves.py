from gettext import gettext as _

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pydemic.types import ValueStd
from . import helpers
from .. import fitting as fit
from ..diseases import disease as get_disease
from ..utils import trim_weeks, weekday_name, accumulate_weekly


def cases_and_deaths(
    data: pd.DataFrame,
    dates: bool = False,
    ax: plt.Axes = None,
    smooth: bool = True,
    cases: str = "cases",
    deaths: str = "deaths",
    tight_layout=False,
    **kwargs,
) -> plt.Axes:
    """
    A simple chart showing observed new cases cases as vertical bars and
    a smoothed out prediction of this curve.

    Args:
        data:
            A dataframe with ["cases", "deaths"] columns.
        dates:
            If True, show dates instead of days in the x-axis.
        ax:
            An explicit matplotlib axes.
        smooth:
            If True, superimpose a plot of a smoothed-out version of the cases
            curve.
        cases:
        deaths:
            Name of the cases/deaths columns in the dataframe.
    """

    if not dates:
        data = data.reset_index(drop=True)

    # Smoothed data
    col_names = {cases: _("Cases"), deaths: _("Deaths")}
    if smooth:
        from pydemic import fitting as fit

        smooth = pd.DataFrame(
            {
                _("{} (smooth)").format(col_names[cases]): fit.smoothed_diff(data[cases]),
                _("{} (smooth)").format(col_names[deaths]): fit.smoothed_diff(data[deaths]),
            },
            index=data.index,
        )
        ax = smooth.plot(legend=False, lw=2, ax=ax)

    # Prepare cases dataframe and plot it
    kwargs.setdefault("alpha", 0.5)
    new_cases = data.diff().fillna(0)
    new_cases = new_cases.rename(col_names, axis=1)

    if "ylim" not in kwargs:
        deaths = new_cases.iloc[:, 1]
        exp = np.log10(deaths[deaths > 0]).mean()
        exp = min(10, int(exp / 2))
        kwargs["ylim"] = (10 ** exp, None)
    ax: plt.Axes = new_cases.plot.bar(width=1.0, ax=ax, **kwargs)

    # Fix xticks
    periods = 7 if dates else 10
    xticks = ax.get_xticks()
    labels = ax.get_xticklabels()
    ax.set_xticks(xticks[::periods])
    ax.set_xticklabels(labels[::periods])
    ax.tick_params("x", rotation=0)
    ax.set_ylim(1, None)
    if tight_layout:
        fig = ax.get_figure()
        fig.tight_layout()
    return ax


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
        growth, growth_std = ValueStd.mean(fit.growth_factors(by_week))
        daily_growth = growth ** (1 / 7)

        X = np.arange(-0.5, 7.5)
        trend = daily_growth ** X
        trend /= trend.sum()
        ax.plot(X)

    elif trend == "cte" or trend is True:
        helpers.mark_x(1 / 7, trend_style)

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
