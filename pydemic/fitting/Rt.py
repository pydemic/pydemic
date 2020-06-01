import pandas as pd

from . import K
from .epidemic_curves import epidemic_curve
from .utils import cases
from .. import formulas
from ..diseases import disease as get_disease
from ..docs import docstring

ARGS = """Args:
    model ({'SIR', 'SEIR', 'SEAIR', etc}):
            Epidemic model used to compute R(t) from K(t)
    curves (pd.DataFrame):
        A dataframe with epidemic curves.
    window (int):
        Window size of triangular smoothing. Can be a single number or
        a 2-tuple with the window used to compute daily cases and the window
        used to smooth out the log-derivative.
    Re (bool):
        If True, return the effective reproduction number, instead of R0. Re
        does not consider the depletion of susceptibles and thus does not
        require to known the population size.
    population:
        Required parameter to obtain R(t), if model is passed as a string.
        This may be required if params is initialized from a disease object.
    params:
        Optional object holding simulation params. This argument is not necessary
        if model can be treated as Params or if a disease is given.
"""
METHODS_RT = {}
METHODS_R0 = {}
METHODS = {"Rt": METHODS_RT, "R0": METHODS_R0}


def method(which, names):
    """
    Decorator that register implementations to be used by the estimate_Rt()
    and estimate_R() functions.
    """
    names = [names] if isinstance(names, str) else names
    db = METHODS[which]

    def decorator(fn):
        for name in names:
            db[name] = fn
        return fn

    return decorator


@docstring(args=ARGS)
def estimate_Rt(model, curves: pd.DataFrame, method="RollingOLS", **kwargs) -> pd.DataFrame:
    """
    Estimate R(t) from epidemic curves and model.

    {args}

    Returns:
        A DataFrame with "Rt" and possibly other columns, depending on the
        method.

    See Also:
        naive_Rt
        rolling_OLS_Rt
    """
    return METHODS_RT[method](model, curves, **kwargs)


@docstring(args=ARGS)
@method("Rt", "naive")
def naive_Rt(model, curves: pd.DataFrame, window=14, **kwargs) -> pd.DataFrame:
    """
    Naive inference of R(t) from Epidemic curves using the naive_Kt() function.

    {args}


    See Also:
        :func:`pydemic.fitting.K.naive_Kt`
    """

    Kt = K.naive_Kt(curves, window=window)
    return Rt_from_Kt(model, curves, Kt, **kwargs)


@docstring(args=ARGS)
@method("Rt", ["RollingOLS", "ROLS"])
def rolling_OLS_Rt(model: str, curves: pd.DataFrame, window=14, **kwargs):
    """
    Compute R(t) from K(t) using the K.rolling_ols() function.

    {args}

    See Also:
        :func:`pydemic.fitting.K.rolling_OLS_Kt`
    """

    Kt = K.rolling_OLS_Kt(curves, window=window)
    return Rt_from_Kt(model, curves, Kt, **kwargs)


#
# Auxiliary functions
#
def Rt_from_Kt(model, curves, Kt, Re=False, **kwargs) -> pd.DataFrame:
    """
    Return Rt from Kt for model.

    Wraps a common logic for many functions in this module.
    """

    if isinstance(model, str):
        params = None
    else:
        params = model
        model = model.epidemic_model_name()

    params = kwargs.pop("params", params)
    if params is None:
        disease = get_disease()
        params = disease.params()

    re = pd.DataFrame(
        {col: formulas.R0_from_K(model, params, K=data, **kwargs) for col, data in Kt.iteritems()}
    )
    re.columns = ["Rt" + c[2:] if c.startswith("Kt") else c for c in re.columns]
    if Re:
        return re

    data = epidemic_curve(model, cases(curves), params)
    depletion = data["susceptible"] / data.sum(1)
    re /= depletion.values[:, None]
    return re
