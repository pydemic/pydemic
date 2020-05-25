import numpy as np

from .base import formula

IGNORE = ("sigma", "rho", "prob_symptoms")
sir_formula = lambda ignore=(): formula("SIR", ignore=(*ignore, *IGNORE))


@sir_formula()
def R0(beta, gamma):
    """
    R0 from model parameters.
    """
    return beta / gamma


@sir_formula()
def beta(R0, gamma):
    """
    R0 from model parameters.
    """
    return R0 * gamma


@sir_formula()
def R0_from_K(K, gamma) -> float:
    """
    Return R0 from the exponential growth factor K and the other model
    parameters.
    """
    return 1.0 + K / gamma


@sir_formula()
def K(R0, gamma) -> float:
    """
    Return the exponential growth factor K from R0 and the other model
    parameters.
    """
    return gamma * (R0 - 1)


#
# Initial conditions
#
@sir_formula()
def time_to_seed(cases, R0, gamma) -> float:
    """
    Simple exponential extrapolation for the time from the first seed to
    reaching the given number of cases using the SIR model.

    This is a decent approximation for the outset of an epidemic. If the seed
    consists of more than a single individual, divide the number of cases by
    the estimated seed size.

    This extrapolation simply assumes that the I compartment grows with
    ``I = I0 * exp(K * t)`` and the number of cases as ``int(beta * I(t), t) + seed``
    """
    if R0 <= 1:
        raise ValueError(f"R0 must be greater than one (got {R0})")
    seed = 1
    return np.log((cases * (R0 - 1) + seed) / (2 * R0 - 1)) / (R0 - 1) / gamma


@sir_formula(ignore=["gamma"])
def infectious_from_cases(cases, R0) -> float:
    """
    Initializes the "infectious" component of a SIR model from the current
    number of cases.

    This formula assumes a perfect exponential growth.
    """
    if R0 <= 1:
        raise ValueError(f"R0 must be greater than one (got {R0})")

    seed = 1
    return (cases * (R0 - 1) + seed) / (2 * R0 - 1)


@sir_formula(ignore=["gamma"])
def state_from_cases(population, cases, R0):
    """
    Initialize the 3 components of a SIR model from cases and population.
    """
    I = infectious_from_cases.formula(cases, R0)  # noqa: E754
    R = cases - I
    S = population - I - R
    return np.array([S, I, R])
