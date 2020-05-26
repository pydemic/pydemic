import operator as op
from typing import Sequence

import numpy as np
import pandas as pd

_ = lambda x: x
WEEKDAY_NAMES = [
    _("Monday"),
    _("Tuesday"),
    _("Wednesday"),
    _("Thursday"),
    _("Friday"),
    _("Saturday"),
    _("Sunday"),
]


def trim_weeks(data: pd.DataFrame, week_start=0) -> pd.DataFrame:
    """
    Given a dataframe indexed with datetime values, trim both ends to correspond
    to whole weeks.

    Args:
        data:
            A dataframe or series object indexed with dates.
        week_start:
            First day of each week: Monday = 0, Tuesday = 1, and so on.
    """

    weekday = day_of_week(data)
    first_day = weekday.iloc[0]
    if first_day != week_start:
        idx = (week_start - first_day) % 7
        data = data.iloc[idx:]

    week_end = (week_start + 6) % 7
    last_day = weekday.iloc[-1]
    if last_day != week_end:
        idx = len(data) - (last_day - week_end) % 7
        data = data.iloc[:idx]

    return data


def accumulate_weekly(data: pd.DataFrame, method="strict", week_start=0, acc=lambda x: x.sum()):
    """
    Accumulate values by week.

    Args:
        data:
            A dataframe with pandas datetimes in the index.
        method:
            Governs how the function handles partial weeks.
            * 'strict' (default): Raise an error.
            * 'trim-start': Remove additional days from the beginning of the time series.
            * 'trim-end': Remove additional days from the end of the time series.
            * 'trim': Trim both sides to make all weeks start in a predictable day.
        week_start:
            If method is 'trim', controls which day is considered to be the first day
            of the week. Monday = 0, Tuesday = 1, and so on.
        acc:
            Accumulator function.
    """
    if method == "trim":
        data = trim_weeks(data, week_start)
    elif method == "strict":
        if len(data) % 7 != 0:
            raise ValueError("series contains partial weeks")
    elif method == "trim-start":
        extra = len(data) % 7
        if extra:
            data = data.iloc[:-extra]
    elif method == "trim-end":
        extra = len(data) % 7
        if extra:
            data = data.iloc[extra:]
    else:
        raise ValueError(f"invalid method: {method!r}")

    by_week = []
    n_weeks = len(data) // 7
    for week in range(n_weeks):
        total = acc(data.iloc[7 * week : 7 * (week + 1)])
        by_week.append(total)
    return pd.DataFrame(by_week)


def day_of_week(df: pd.DataFrame, col=None) -> pd.Series:
    """
    Return a series for weekdays from input.

    Args:
        df:
            Dataframe or series object.
        col:
            Name of the column with datetime values. If not given, assumes to
            be the index.
    """
    if col is None:
        data = df.index
    else:
        data = df[col]
    data = [*map(op.attrgetter("dayofweek"), data)]
    return pd.Series(data, index=df.index)


def weekday_name(idx, translate=str):
    """
    Return name for weekday from index.

    Input can be an scalar or a sequence.

    Notes:
          Monday = 0, Tuesday = 1, and so on.
    """
    _ = translate or str

    if isinstance(idx, Sequence):
        names = WEEKDAY_NAMES
        print(idx)
        return np.array([_(names[i]) for i in idx])
    return _(WEEKDAY_NAMES[idx])
