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
    is_finite = property(lambda self: np.isfinite(self.value))

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

    def apply(self, func, derivative=None):
        """
        Transform value by function.
        """
        derivative = derivative or numeric_derivative(func)
        return ValueStd(func(self.value), abs(derivative(self.value)) * self.std)


class ImproperlyConfigured(Exception):
    """
    Exception raised when trying to initialize object with improper configuration.
    """


def numeric_derivative(func, epsilon=1e-6):
    """
    Return a function that computes the numeric first-order derivative of func.
    """

    def diff(x):
        return (func(x + epsilon) - func(x)) / epsilon

    return diff
