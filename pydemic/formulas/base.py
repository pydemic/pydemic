import inspect
from collections import defaultdict
from functools import wraps
from numbers import Real

import numpy as np

from ..params import get_param, ParamLike

# Function registry
FUNCTIONS_R0_FROM_K = {}
FUNCTIONS_K = {}
FUNCTIONS_R0 = {}
FUNCTIONS_BETA = {}
FUNCTIONS_STATE_FROM_CASES = {}
FUNCTIONS = defaultdict(
    dict,
    {
        "R0_from_K": FUNCTIONS_R0_FROM_K,
        "K": FUNCTIONS_K,
        "R0": FUNCTIONS_R0,
        "beta": FUNCTIONS_BETA,
        "state_from_cases": FUNCTIONS_STATE_FROM_CASES,
    },
)

# Constants
LN_2 = np.log(2)

# Aliases and alternate forms for the "formula" decorator.
ALTERNATE_FORMS = {
    # Inverted relations
    "infectious_period": ("gamma", lambda x: 1 / x),
    "incubation_period": ("sigma", lambda x: 1 / x),
    "doubling_time": ("K", lambda k: float("inf") if k == 0 else LN_2 / k),
    # Aliases
    "Qs": ("prob_symptoms", lambda x: x),
}


def formula(model=None, ignore=(), options=(), invalid="raise", positional=0, formula_name=None):
    """
    Decorator that register formula functions.

    Formula functions receive a simple function that computes a mathematical
    formula from numeric keyword arguments and return a more flexible interface
    which can accept arguments from a namespace, understand alternative
    representations and aliases for common arguments, etc.

    Args:
        model:
            Name of the epidemiological model the formula applies to. If a
            sequence is passed, register function for several different models.
        ignore:
            Sequence of arguments to ignore. This is useful if one wants to
            provide a consistent interface for different models. This way, a
            simple model can simply ignore parameters from a more sophisticated
            one (e.g.. SIR can ignore the incubation_period of a SEIR model).
        options:
            Additional parameters that are simply passed to the decorated
            function without any special handling.
        positional:
            The number of required positional arguments the function uses.
        invalid ('ignore', 'pass', 'raise'):
            Strategy to use when encounter invalid parameters: 'ignore', simply
            suppress them, 'pass', pass them to the decorated function and
            'raise' (default) raises an error.
        formula_name:
            If given, register function under a different name than the declared
            function name.
    """

    def decorator(fn):
        if invalid not in ("ignore", "pass", "raise"):
            raise ValueError(f'bad "invalid" strategy: {invalid!r}')

        signature = inspect.signature(fn)
        arguments = set(list(signature.parameters)[positional:]).difference(options)
        alternatives = {x for x, (k, _) in ALTERNATE_FORMS.items() if k in arguments}
        fn_name = formula_name or fn.__name__

        # Remove **starred_kwargs from arguments
        for arg in signature.parameters.values():
            if arg.kind == arg.VAR_KEYWORD:
                arguments.discard(arg.name)
        n_args = len(arguments)

        # Feed ignore set with the list of alternatives
        ignore_set = set(ignore)
        options_set = set(options)

        for k in ignore_set:
            try:
                k, _ = ALTERNATE_FORMS[k]
                ignore_set.add(k)
            except KeyError:
                continue

        @wraps(fn)
        def decorated(*args_, **kwargs):
            params: ParamLike = None

            options = {}
            args = args_[:positional]
            if len(args_) == positional:
                params = None
            elif len(args_) == positional + 1:
                params = args_[positional]
            else:
                n = positional
                m = len(args_)
                TypeError(f"Expect {n} or {n + 1} positional arguments, got {m}")

            args = args[:positional]

            # Remove all ignored parameters
            if ignore_set:
                for k in list(kwargs):
                    if k in ignore_set:
                        del kwargs[k]

            # Extract options
            for k in options_set.intersection(kwargs):
                options[k] = kwargs.pop(k)

            # Normalize all alternative values passed as keyword arguments.
            for k in alternatives.intersection(kwargs):
                v = kwargs.pop(k)
                k, transform = ALTERNATE_FORMS[k]
                kwargs[k] = transform(v)

            # Check for invalid keyword arguments
            invalid_set = set(kwargs) - arguments
            if invalid_set and invalid == "ignore":
                for k in invalid:
                    del kwargs[k]
            elif invalid_set and invalid == "raise":
                raise TypeError(f"invalid argument: {invalid_set.pop()}")
            elif invalid_set:
                options.update({k: kwargs.pop(k) for k in invalid_set})

            if len(kwargs) == n_args:
                return fn(*args, **kwargs, **options)

            # If we do not have enough arguments, try the positional value
            missing = arguments - kwargs.keys()
            if params is None:
                raise TypeError(f"missing required argument: {missing.pop()}")

            for k in missing:
                try:
                    kwargs[k] = get_param(k, params)
                except ValueError:
                    pass

            # Try alternate forms if not all keyword arguments were completed.
            if len(kwargs) < n_args:
                for argname in alternatives:
                    k, transform = ALTERNATE_FORMS[argname]
                    if k not in kwargs:
                        kwargs[k] = transform(get_param(argname, params))

            return fn(*args, **kwargs, **options)

        def register(model):
            """
            Register function to model or list of models.

            Return the decorated function.
            """

            lst = (model,) if isinstance(model, str) else tuple(model)
            decorated.models = frozenset([*lst, *decorated.models])
            for m in lst:
                FUNCTIONS[fn.__name__][m] = decorated
            return decorated

        # Save attributes to the decorated function
        decorated.formula = fn
        decorated.arguments = frozenset(arguments)
        decorated.alternate_arguments = frozenset(alternatives)
        decorated.register = register

        # Register formula
        model_lst = (model,) if isinstance(model, str) else tuple(model or ())
        for m in model_lst:
            FUNCTIONS[fn_name][m] = decorated
        decorated.models = frozenset(model_lst)

        return decorated

    return decorator


#
# Generic formulas
#
def R0(model, params=None, **kwargs) -> float:
    """
    Compute R0 for the given model using epidemiological parameters.

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
        >>> R0("SIR", beta=1.0, gamma=0.5)
        2.0

    See Also:
        :func:`sir.R0`
        :func:`seir.R0`
        :func:`seair.R0`

    Returns:
        The value of R0 for that model.
    """
    return FUNCTIONS_R0[model](params, **kwargs)


def R0_from_K(model, params=None, **kwargs) -> float:
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
        :func:`sir.R0_from_K`
        :func:`seir.R0_from_K`
        :func:`seair.R0_from_K`

    Returns:
        The value of R0 for that model.
    """
    return FUNCTIONS_R0_FROM_K[model](params, **kwargs)


def beta(model, params=None, **kwargs) -> float:
    """
    Compute beta from for the given model using epidemiological parameters.

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
        >>> beta("SIR", R0=2.0, gamma=0.5)
        1.0

    See Also:
        :func:`sir.beta`
        :func:`seir.beta`
        :func:`seair.beta`

    Returns:
        The value of beta for that model.
    """
    return FUNCTIONS_BETA[model](params, **kwargs)


def K(model, params=None, **kwargs) -> float:
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


def initial_state(model, cases, params=None, **kwargs) -> np.ndarray:
    """
    Compute the initial state for model using the given number of cases.

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
       An array with the estimated initial state.
    """
    return FUNCTIONS_STATE_FROM_CASES[model](params, cases=cases, **kwargs)


def get_function(model, name):
    """
    Return registered function.
    """
    return FUNCTIONS[name][model]
