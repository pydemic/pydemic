from typing import Union, Sequence, TypeVar

import numpy as np
import pandas as pd

Data = Union[Sequence, pd.Series, np.ndarray]
T = TypeVar("T", pd.Series, pd.DataFrame, np.ndarray, Sequence)


def trim_zeros(data: T, direction="both") -> T:
    """
    Remove null values from the start and end of series or array.
    """
    # TODO: use numpy's trim_zeros internally.
    if direction == "none":
        return data

    trim_left = direction in ("left", "both", "begin", "start")
    trim_right = direction in ("right", "both", "end", "finish")

    i = j = 0
    if np.ndim(data) > 1:
        values = data.values
        if trim_left:
            for i, x in enumerate(values):
                if not (x == 0).all():
                    break
        if trim_right:
            for j, x in enumerate(reversed(values)):
                if not (x == 0).all():
                    break
    else:
        if trim_left:
            for i, x in enumerate(data):
                if x != 0:
                    break
        if trim_right:
            for j, x in enumerate(reversed(data)):
                if x != 0:
                    break
    j = max(len(data) - j, i)

    if hasattr(data, "iloc"):
        return data.iloc[i:j]
    return data[i:j]


def force_monotonic(data: T, decreasing=False, check=True, copy=True) -> T:
    """
    Force series to be monotonically increasing/decreasing

    Args:
        data (pd.Series):
            Data to become monotonic.
        decreasing (bool):
            If True, force a monotonic decreasing behavior.
        copy:
            If False, make changes inplace.
        check:
            If False, prevent checking if data is monotonic before fixing
            its contents.

    Returns:
        pd.Series
    """

    if copy:
        data = data.copy()

    if isinstance(data, pd.DataFrame):
        for col in data:
            data[col] = force_monotonic(data[col], decreasing, check, copy=False)
        return data

    if check:
        if decreasing and data.is_monotonic_decreasing:
            return data
        elif not decreasing and data.is_monotonic_increasing:
            return data

    indexes = np.arange(len(data))
    idx = indexes[(data.diff() < 0).values]
    for i in reversed(idx):
        data.iloc[i - 1] = data.iloc[i]
    return data
