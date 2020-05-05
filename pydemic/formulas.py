from numbers import Real

import numpy as np

from .params import select_param, ParamLike

FUNCTIONS = {"R0_from_K": {}, "K": {}, "R0": {}}
FUNCTIONS_R0_FROM_K = FUNCTIONS["R0_from_K"]
FUNCTIONS_K = FUNCTIONS["K"]
LN_2 = np.log(2)


def register(*args):
    """
    Decorator that register function to compute parameter for model.
    """

    def decorator(fn):
        if not args:
            param, _, model = fn.__name__.rpartition("_")
        else:
            param, model = args
        FUNCTIONS[param][model] = fn
        return fn

    return decorator


#
# Generic formulas
#
def R0_from_K(model, params=None, **kwargs) -> Real:
    """
    Compute R0 for the given model using the exponential growth factor K.

    Args:
        model ({'SIR', 'SEIR', 'SEAIR', etc}):
            A string naming the desired model.
        params:
            A params object as described in :func:`get_param`.

    Keyword Args:
        Extra epidemiological parameters that may be used in the formula. This
        overrides the values passed in params. It is safe to pass non-used
        parameters, and they will be simply ignored.

    Examples:
        >>> R0_from_K("SIR", K=0.5, gamma=0.5)
        2.0

    See Also:
        :func:`R0_from_K_SIR`
        :func:`R0_from_K_SEIR`
        :func:`R0_from_K_SEAIR`

    Returns:
        The value of R0 for that model.
    """
    return FUNCTIONS_R0_FROM_K[model](params, **kwargs)


def K(model, params=None, **kwargs) -> Real:
    """
    Compute the exponential growth factor K for the given model.

    Args:
        model ({'SIR', 'SEIR', 'SEAIR', etc}):
            A string naming the desired model.
        params:
            A params object as described in :func:`get_param`.

    Keyword Args:
        Extra epidemiological parameters that may be used in the formula. This
        overrides the values passed in params. It is safe to pass non-used
        parameters, and they will be simply ignored.

    Examples:
        >>> K("SIR", R0=2.0, gamma=0.5)
        0.5

    Returns:
        The value of K for that model.
    """
    return FUNCTIONS_K[model](params, **kwargs)


def doubling_time(model, params=None, **kwargs) -> Real:
    """
    Compute the doubling time for the given model. Negative results correspond
    to the halving time of decaying processes.

    Args:
        model ({'SIR', 'SEIR', 'SEAIR', etc}):
            A string naming the desired model.
        params:
            A params object as described in :func:`get_param`.

    Keyword Args:
        Extra epidemiological parameters that may be used in the formula. This
        overrides the values passed in params. It is safe to pass non-used
        parameters, and they will be simply ignored.

    Returns:
        The doubling (or halving) time for the process.
    """
    k = FUNCTIONS_K[model](params, **kwargs)
    if k == 0:
        return float("inf")
    else:
        return LN_2 / k


#
# SIR formulas
#
@register()
def R0_from_K_SIR(params: ParamLike = None, K=None, gamma=None, **kwargs) -> float:
    """
    Return R0 from the exponential growth factor K and the other model
    parameters.

    Params can be anything that has the epidemiological parameters
    as attributes or methods.
    """
    gamma = select_param("gamma", params, gamma)
    K = select_param("K", params, K)

    return 1.0 + K / gamma


@register()
def K_SIR(params: ParamLike = None, R0=None, gamma=None, **kwargs) -> float:
    """
    Return the exponential growth factor K from R0 and the other model
    parameters.

    Params can be anything that has the epidemiological parameters
    as attributes or methods.
    """
    gamma = select_param("gamma", params, gamma)
    R0 = select_param("R0", params, R0)

    return gamma * (R0 - 1)


#
# SEIR formulas
#
@register()
@register("R0_from_K", "SEAIR")
def R0_from_K_SEIR(params: ParamLike = None, K=None, gamma=None, sigma=None, **kwargs) -> float:
    gamma = select_param("gamma", params, gamma)
    sigma = select_param("sigma", params, sigma)
    K = select_param("K", params, K)

    return 1.0 + (gamma + sigma + K) * K / (gamma * sigma)


@register()
@register("K", "SEAIR")
def K_SEIR(params: ParamLike = None, R0=None, gamma=None, sigma=None, **kwargs) -> float:
    gamma = select_param("gamma", params, gamma)
    sigma = select_param("sigma", params, sigma)
    R0 = select_param("R0", params, R0)
    mu = sigma + gamma

    return 0.5 * mu * (np.sqrt(1 + 4 * (R0 - 1) * sigma * gamma / (mu * mu)) - 1)


#
# SEAIR formulas
#
R0_from_K_SEAIR = R0_from_K_SEIR
K_SEAIR = K_SEIR
