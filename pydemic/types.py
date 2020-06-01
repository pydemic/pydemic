from collections import namedtuple
from typing import NamedTuple, Tuple, Iterable

import numpy as np
import pandas as pd

Result = namedtuple("Result", ["value", "info"])
ValueCI = namedtuple("ValueCI", ["value", "low", "high"])


class ValueStd(NamedTuple):
    """
    A value that represents a number with its standard deviation.
    """

    value: float
    std: float

    @classmethod
    def mean(cls, iterable: Iterable[Tuple[float, float]], tol=1e-9) -> "ValueStd":
        """
        Merge several independent point estimates of (value, std) into a single
        estimate, weighting results by the inverse variance.

        Args:
            iterable:
                A sequence of (value, std) tuples.
            tol:
                A normalization term to avoid problem with null variances.
        """
        if isinstance(iterable, pd.DataFrame):
            iterable = iterable.values

        weights = 0.0
        cum_var = 0.0
        N = 0

        for (value, std) in iterable:
            var = std * std + tol
            weight = 1 / var
            weights += weight
            cum_var += weight * value
            N += 1

        return ValueStd(cum_var / weights, np.sqrt(cum_var / N))


class ImproperlyConfigured(Exception):
    """
    Exception raised when trying to initialize object with improper configuration.
    """
