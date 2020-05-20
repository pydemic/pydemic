from typing import Union

import numpy as np
import statsmodels.api as sm

from ..types import Result


def smoothed_diff(
    data: np.ndarray,
    *,
    bias: float = 1e-4,
    damped: bool = True,
    linear: bool = False,
    fast_init: bool = False,
    retall: bool = False,
    **kwargs,
) -> Union[np.ndarray, Result]:
    """
    Load a series of values and use Holt's algorithm to smooth the finite
    differences.

    Args:
        data:
            Input data.
        damped:
            If True (default), enable damping of the exponential trend.
        linear:
            If True, make the trend linear instead of exponential.
        bias:
            Holt's algorithm require strictly positive values (i.e., greater than,
            but not equal to zero). This parameter controls a bias term that is
            added to the baseline to prevent the existence of null values.
            This transformation is not used if linear=True.
        fast_init:
            If True, prevents using the slower brute force search during
            initialization of the model.
        retall:
            If True, return a named tuple (value, info) with both the computed
            smoothed array and the statistical model used to calibrate the
            parameters.
        **kwargs:

    Returns:
        Either a :cls:`np.ndarray` of smoothed differences or a :cls:`result` value.
    """
    data = np.asarray(data)
    maximum_value = data.max()
    data = np.diff(data, prepend=0.0)

    # Guarantee that there are no zero values in the
    # dataset
    if not linear:
        data = np.maximum(data, 0.0)
        S = data.sum()
        data *= 1 - bias
        data += bias * S

    # Choose empirically validated values
    kwargs.setdefault("smoothing_level", 1 / 5)

    # Holt exponential smoothing
    holt = sm.tsa.Holt(data, exponential=not linear, damped=damped)
    res = holt.fit(use_brute=not fast_init, **kwargs)
    out = res.fittedvalues
    out *= maximum_value / out.sum()
    return Result(out, res) if retall else out
