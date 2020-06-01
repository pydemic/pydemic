from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from mundi.pandas import Pandas
from .utils import diff, smooth, cases
from ..docs import docstring
from ..types import ValueStd

KT_METHODS = {}
K_METHODS = {}
METHODS = {"K": K_METHODS, "Kt": KT_METHODS}
ARGS = """Args:
    curves:
        Cumulative series of cases or deaths.
    window:
        Window size of triangular smoothing. Can be a single number or
        a 2-tuple with the window used to compute daily cases and the window
        used to smooth out the log-derivative.
"""


def method(which, names):
    """
    Decorator that register function as a Kt or K inference method.
    """

    names = [names] if isinstance(names, str) else names
    db = METHODS[which]

    def decorator(fn):
        for name in names:
            if name in db:
                raise RuntimeError(f"method already exists: {name}")
            db[name] = fn
        return fn

    return decorator


#
# Main interface
#
@docstring(args=ARGS)
def estimate_Kt(curves: Pandas, window=14, method="RollingOLS", **kwargs) -> pd.DataFrame:
    """
    Compute K(t) from the epidemic curves. This is function is just a façade to
    the concrete implementations listed bellow.

    {args}
        method ('ROLS', 'naive'):
            The method name:
            * 'ROLS'/'RollingOLS': Uses a Rolling Ordinary Least Squares
              implementation from :func:`rolling_OLS_Kt`.
            * 'naive': Obtain K(t) from the derivative of a smoothed out curve
               of log-new-cases. Implemented by :func:`naive_Kt`.

    Returns:
        pd.DataFrame with a Kt column and possibly other information (e.g.,
        Kt_low, Kt_high) depending on the method.

    See Also:
        naive_Kt
        rolling_OLS_Kt
    """
    return KT_METHODS[method](curves, window=window, **kwargs)


@docstring(args=ARGS)
def estimate_K(curves: Pandas, method="naive", **kwargs) -> ValueStd:
    """
    Compute an stationary K for the entire epidemic curve. This is function is
    just a façade to the concrete implementations listed bellow.

    {args}
        method ('OLS', 'naive'):
            The method name:
            * 'OLS': Uses ordinary least squares of log-new-cases.
            * 'Naive': Simply compute the mean and standard deviation from
              the smoothed-out curve or log-new-cases.

    See Also:
        naive_K
        OLS_K
    """
    return K_METHODS[method](curves, **kwargs)


#
# Time-dependent K(t)
#
@docstring(args=ARGS)
@method("Kt", "naive")
def naive_Kt(curves: Pandas, window=14) -> pd.DataFrame:
    """
    Return K(t) as the derivative of the logarithm of a smoothed-out series of
    new cases or deaths.

    {args}
    """
    a, b = window if isinstance(window, Sequence) else (window, window)
    daily = diff(cases(curves), prepend=0, smooth=a)
    data = diff(np.log(daily), smooth=b)

    return pd.DataFrame({"Kt": data})


@docstring(args=ARGS)
@method("Kt", ("ROLS", "RollingOLS"))
def rolling_OLS_Kt(curves, window=14) -> pd.DataFrame:
    """
    A Rolling window Ordinary Least Squares inference of the derivative of the
    logarithm of the number of cases.

    {args}
    """

    a, b = window if isinstance(window, Sequence) else (window, window)
    daily = diff(cases(curves), smooth=a)

    # We first make a OLS inference to extrapolate series to past
    Y = np.log(daily).values
    X = np.arange(len(Y))
    ols = sm.OLS(Y[:b], sm.add_constant(X[:b]), missing="drop")
    res = ols.fit()

    # We need at least c new observations to obtain a result without NaNs
    m = res.params[1]

    X_ = np.arange(X[0] - b, X[0])
    Y_ = m * (X_ - X[0]) + Y[0]

    X = np.concatenate([X_, X])
    Y = np.concatenate([Y_, Y])

    # Use Rolling OLS to obtain an inference to the growth ratio
    ols = RollingOLS(Y, sm.add_constant(X), window=b, missing="drop")
    res = ols.fit()

    Kt = res.params[b:, 1]
    low, high = res.conf_int()[b:, :, 1].T

    out = pd.DataFrame({"Kt": Kt, "Kt_low": low, "Kt_high": high}, index=curves.index)

    return out


#
# Static K
#
@docstring(args=ARGS)
@method("K", "naive")
def naive_K(curves, window=14) -> ValueStd:
    """
    Ordinary Least squares inference of data growth factor using the logarithm
    of the new number of cases.

    {args}
    """
    data = naive_Kt(curves, window)["Kt"]
    return ValueStd(data.mean(), data.std())


@docstring(args=ARGS)
@method("K", "OLS")
def ordinary_least_squares_K(curves, window=14) -> ValueStd:
    """
    Ordinary Least squares inference of data growth factor using the logarithm
    of the new number of cases.

    {args}
    """

    a, b = window if isinstance(window, Sequence) else (window, window)
    daily = diff(cases(curves), a)

    Y = smooth(np.log(daily), b)
    X = np.arange(len(Y))

    ols = sm.OLS(Y, sm.add_constant(X), missing="drop")
    res = ols.fit()

    return ValueStd(res.params[1], res.cov_HC0[1, 1])
