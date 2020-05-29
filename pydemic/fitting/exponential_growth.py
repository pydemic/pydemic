from typing import Sequence

import numpy as np
import pandas as pd

from ..types import ValueStd
from ..utils import trim_zeros


def growth_factor(ys: Sequence) -> ValueStd:
    """
    Infer the  growth factor from series of data points ys.

    The growth factor "r" is the parameter that best adjust the sequence:

        y[i + 1] = r * y[i].

    Args:
        ys:
            A sequence of data points approximately generated by the growth
            process shown in the description above.

    Returns:
        A ValueStd named tuple of (value, std), with the associated expected
        value and standard deviation.
    """
    ys = np.asarray(ys)
    eta = ys[1:] / ys[:-1]
    N = len(eta)
    mean_eta = eta.mean()
    delta = np.log(mean_eta) - np.log(eta).mean()

    mean_r = (mean_eta + 2 * delta / N) / (1 + 2 * delta / N)
    denom = N / 2 - 2 * delta
    if denom > 0:
        std_r = np.sqrt(delta / denom) * mean_r
    else:
        std_r = float("inf")
    return ValueStd(mean_r, std_r)


def growth_factors(data):
    """
    Compute growth factors for each column of ys in data frame.

    Return a data frame with ["value", "std"] columns with the original columns
    in the index.
    """
    growth_factors = {}
    for key, col in data.items():
        col = trim_zeros(col)
        growth_factors[key] = growth_factor(col)

    return pd.DataFrame(growth_factors, index=["value", "std"]).T


def average_growth(results, tol=1e-9) -> ValueStd:
    """
    Compute average growth factor from sequence of results, weighting
    by the inverse variance.

    Args:
        results:
            A sequence of (value, std) tuples.
        tol:
            A normalization term to avoid problem with null variances.
    """
    if isinstance(results, pd.DataFrame):
        results = results.values

    weights = 0.0
    cum_var = 0.0
    N = 0

    for (value, std) in results:
        var = std * std + tol
        weight = 1 / var
        weights += weight
        cum_var += weight * value
        N += 1

    return ValueStd(cum_var / weights, np.sqrt(cum_var / N))


def exponential_extrapolation(ys: Sequence, n: int, append=False) -> np.ndarray:
    """
    Receive a sequence  and return the next n points of the series
    extrapolating from the input data.

    Args:
        ys:
            Input data.
        n:
            Number of points of the extrapolation.
        append:
            If True, returns a concatenation of the input series with the
            extrapolation.
    """
    ys = np.asarray(ys)
    r, dr = growth_factor(ys)
    K = np.log(r)
    extrapolation = ys[-1] * np.exp(K * np.arange(1, n + 1))
    if append:
        return np.concatenate([ys, extrapolation])
    return extrapolation
