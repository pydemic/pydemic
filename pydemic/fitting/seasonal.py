import pandas as pd

from ..utils import trim_weeks, accumulate_weekly


def weekday_rate(data):
    """
    Read time series data and show the fraction accumulated in each weekday
    from a series of observations.
    """

    data = trim_weeks(data)
    by_week = accumulate_weekly(data, "trim")

    mean = []
    std = []

    for weekday in range(7):
        values = data.iloc[weekday::7].reset_index(drop=True)
        ratios = values / by_week

        mean.append(ratios.mean())
        std.append(ratios.std())

    means = pd.DataFrame(mean)
    std = pd.DataFrame(std)
    std.columns = [col + "_std" for col in std.columns]

    return pd.concat([means, std], axis=1)
