import numpy as np

from . import sir
from .base import formula

IGNORE = tuple(x for x in sir.IGNORE if x != "sigma")
seir_formula = lambda ignore=(): formula("SEIR", ignore=(*ignore, *IGNORE))

R0 = sir.R0.register("SEIR")
beta = sir.beta.register("SEIR")


@seir_formula()
def R0_from_K(K, gamma, sigma) -> float:
    """
    Return R0 from the exponential growth factor K and the other model
    parameters.
    """
    return 1.0 + (gamma + sigma + K) * K / (gamma * sigma)


@seir_formula()
def K(R0, gamma, sigma) -> float:
    """
    Return the exponential growth factor K from R0 and the other model
    parameters.
    """
    mu = sigma + gamma
    return 0.5 * mu * (np.sqrt(1 + 4 * (R0 - 1) * sigma * gamma / (mu * mu)) - 1)


#
# Initial conditions
#
@seir_formula()
def time_to_seed(cases, R0, gamma, sigma):
    """
    Simple exponential extrapolation for the time from the first seed to
    reaching the given number of cases using the SEIR/SEAIR model.

    This is a decent approximation for the outset of an epidemic. If the seed
    consists of more than a single individual, divide the number of cases by
    the estimated seed size.
    """
    if R0 <= 1:
        raise ValueError(f"R0 must be greater than one (got {R0})")

    seed = 1
    cases_ = (cases - seed) / seed
    k = K.formula(R0, gamma, sigma)
    beta = gamma * R0
    return 1 / k * np.log((cases_ * k + beta) / beta)


@seir_formula()
def infectious_from_cases(cases, R0, gamma, sigma):
    """
    Initializes the "infectious" component of a SEIR/SEAIR model from the current
    number of cases.

    This formula assumes a perfect exponential growth.
    """
    if R0 <= 1:
        raise ValueError(f"R0 must be greater than one (got {R0})")

    seed = 1
    k = K.formula(R0, gamma, sigma)
    beta = gamma * R0
    return ((cases - seed) * k + beta * seed) / beta


@seir_formula()
def exposed_from_cases(cases, R0, gamma, sigma):
    """
    Initializes the "exposed" component of a SEIR model from the current
    number of cases.

    This formula assumes a perfect exponential growth.
    """
    k = K.formula(R0, gamma, sigma)
    I0 = infectious_from_cases.formula(cases, R0, gamma, sigma)
    return I0 * (gamma * R0) / (k + sigma)


@seir_formula()
def state_from_cases(population, cases, R0, gamma, sigma):
    """
    Initialize the 4 components of a SEIR model from cases and population.
    """
    I = infectious_from_cases.formula(cases, R0, gamma, sigma)
    E = exposed_from_cases.formula(cases, R0, gamma, sigma)
    R = cases - I
    S = population - I - E - R
    return np.array([S, E, I, R])
