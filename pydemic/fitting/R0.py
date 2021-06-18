from typing import Sequence

import numpy as np
import pandas as pd

from . import K
from .Rt import ARGS, METHODS_R0, method, rolling_OLS_Rt, naive_Rt
from .epidemic_curves import epidemic_curve
from .utils import cases
from ..utils import extract_keys
from .. import formulas
from ..docs import docstring
from ..types import ValueStd

EPICURVE_PARAMS = ["population"]


@docstring(args=ARGS)
def estimate_R0(model, curves: pd.DataFrame, method="OLS", **kwargs) -> ValueStd:
    """
    Estimate R0 from epidemic curves and model.

    {args}

    Returns:
        A ValueStd with R0 and its associated standard deviation.

    See Also:
        naive_R0
        OLS_R0
    """
    return METHODS_R0[method](model, curves, **kwargs)


@docstring(args=ARGS)
@method("R0", "naive")
def naive_R0(model, curves: pd.DataFrame, window=14, **kwargs) -> ValueStd:
    """
    Naive inference of R(t) from Epidemic curves using the naive_K() function.

    {args}

    See Also:
        :func:`pydemic.fitting.K.naive_K`
    """

    k = K.naive_K(curves, window=window)
    return R0_from_K(model, curves, k, **kwargs)


@docstring(args=ARGS)
@method("R0", "OLS")
def ordinary_least_squares_R0(model: str, curves: pd.DataFrame, window=14, **kwargs) -> ValueStd:
    """
    Compute R0 from K using the K.ordinary_least_squares_K() function.

    {args}

    See Also:
        :func:`pydemic.fitting.K.ordinary_least_squares_K`
    """

    k = K.ordinary_least_squares_K(curves, window=window)
    return R0_from_K(model, curves, k, **kwargs)


@docstring(args=ARGS)
@method("R0", "RollingOLS")
def rolling_OLS_R0(model: str, curves: pd.DataFrame, window=14, **kwargs) -> ValueStd:
    """
    Compute R0 from K using the rolling_OLS_Rt() function and take the
    average value, weighting the most recent days with an exponential decay.

    {args}

    See Also:
        :func:`pydemic.fitting.K.ordinary_least_squares_K`
    """
    a, b = window if isinstance(window, Sequence) else (window, window)
    Rt = rolling_OLS_Rt(model, curves, window=window, **kwargs)
    weights = np.exp(np.arange(-len(Rt), 0) / b)
    R0, low, high = np.average(Rt.values, weights=weights, axis=0)
    return ValueStd(R0, (high - low) / 2)


@docstring(args=ARGS)
@method("R0", "naive_decay")
def naive_decay(model: str, curves: pd.DataFrame, window=14, **kwargs) -> ValueStd:
    """
    Compute R0 from Rt using the naive_Rt() function and take the
    average value, weighting the most recent days with an exponential decay.

    {args}

    See Also:
        :func:`pydemic.fitting.Rt.naive_Rt`
    """
    a, b = window if isinstance(window, Sequence) else (window, window)
    Rt = naive_Rt(model, curves, window=window, **kwargs)
    weights = np.exp(np.arange(-len(Rt), 0) / b)
    R0 = np.average(Rt.values, weights=weights, axis=0).mean()
    return ValueStd(R0, 0)


#
# Auxiliary functions
#
def R0_from_K(model, curves, K, Re=False, **kwargs) -> ValueStd:
    """
    Return (R0, std) from K for model.

    Wraps a common logic for many functions in this module.
    """

    epi_kwargs = extract_keys(EPICURVE_PARAMS, **kwargs)

    if isinstance(model, str):
        params = None
    else:
        params = model
        model = model.epidemic_model_name()

    params = kwargs.pop("params", params)
    if params is None:
        from ..diseases import disease as get_disease

        disease = get_disease()
        params = disease.params()

    mean, std = K
    K_mean = formulas.R0_from_K(model, params, K=mean, **kwargs)
    K_high = formulas.R0_from_K(model, params, K=mean + std, **kwargs)
    K_low = formulas.R0_from_K(model, params, K=mean - std, **kwargs)
    K_std = np.sqrt(((K_mean - K_high) ** 2 + (K_mean - K_low) ** 2) / 2)

    if Re:
        return ValueStd(K_mean, K_std)

    data = epidemic_curve(model, cases(curves), params, **epi_kwargs)
    depletion = data["susceptible"] / data.sum(1)
    factor = 1 / depletion.iloc[-1]
    return ValueStd(K_mean * factor, K_std * factor)
