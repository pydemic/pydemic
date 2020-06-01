from typing import TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")


def diff(data: T, prepend=0, append=None, smooth=False) -> T:
    """
    Compute difference and return an object with the same shape as the
    original.

    Args:
        data (DataFrame or array):
            Input data to take the derivative.
        prepend:
            Value before the first data point.
        append:
            Value after the last data point. If given, take forward differences
            of data.
        smooth (bool):
            If true, apply the :func:`smooth` function to data. If smooth is a
            number, it is treated as the window size in the triangular
            smoothing process.
    """

    if append is not None:
        out = np.diff(data, axis=0, append=append)
    else:
        out = np.diff(data, axis=0, prepend=prepend)

    if isinstance(data, pd.DataFrame):
        data = pd.DataFrame(out, columns=data.columns, index=data.index)
    elif isinstance(data, pd.Series):
        data = pd.Series(out, index=data.index)
    else:
        data = out

    if smooth:
        window = 14 if smooth is True else smooth
        return _smooth(data, window)
    return data


def smooth(data: T, window=14) -> T:
    """
    Smooth data using a triangular window with the given window size.
    """

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        return smooth(pd.DataFrame(data), window).values

    return data.rolling(window, win_type="triang", min_periods=1, center=True).mean()


def cases(curves) -> pd.Series:
    """
    Return the "cases" column if the dataframe has multiple columns.
    """
    if isinstance(curves, pd.Series):
        return curves
    elif len(curves.columns) == 1:
        return curves.iloc[:, 0]
    else:
        return curves["cases"]


_smooth = smooth
