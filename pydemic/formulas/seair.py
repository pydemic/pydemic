import numpy as np

from . import seir
from .base import formula

IGNORE = tuple(x for x in seir.IGNORE if x not in ("rho", "prob_symptoms"))
seair_formula = lambda ignore=(): formula("SEAIR", ignore=(*ignore, *IGNORE))

R0_from_K = seir.R0_from_K.register("SEAIR")
K = seir.K.register("SEAIR")


@seair_formula(ignore=["sigma"])
def R0(beta, gamma, prob_symptoms, rho):
    """
    R0 from model parameters.
    """
    Qs = prob_symptoms
    return beta / gamma * (Qs + (1 - Qs) * rho)


@seair_formula(ignore=["sigma"])
def beta(R0, gamma, prob_symptoms, rho):
    """
    R0 from model parameters.
    """
    Qs = prob_symptoms
    return R0 * gamma / (Qs + (1 - Qs) * rho)


#
# Initial conditions
#
time_to_seed = seir.time_to_seed.register("SEAIR")
infectious_from_cases = seir.infectious_from_cases.register("SEAIR")


@seair_formula(ignore=["rho"])
def exposed_from_cases(cases, R0, gamma, sigma, prob_symptoms):
    """
    Initializes the "exposed" component of a SEAIR model from the current
    number of cases.

    This formula assumes a perfect exponential growth.
    """
    Qs = prob_symptoms
    k = K.formula(R0, gamma, sigma)
    I0 = infectious_from_cases.formula(cases, R0, gamma, sigma)
    return I0 * (gamma + k) / (sigma * Qs)


@seair_formula(ignore=["rho"])
def asymptomatic_from_cases(cases, R0, gamma, sigma, prob_symptoms):
    """
    Initializes the "asymptomatic" component of a SEAIR model from the current
    number of cases.

    This formula assumes a perfect exponential growth.
    """
    Qs = prob_symptoms
    I0 = infectious_from_cases.formula(cases, R0, gamma, sigma)
    return I0 * (1 - Qs) / Qs


@seair_formula(ignore=["rho"])
def state_from_cases(population, cases, R0, gamma, sigma, prob_symptoms):
    """
    Initialize the 5 components of a SEAIR model from cases and population.
    """

    I = infectious_from_cases.formula(cases, R0, gamma, sigma)
    E = exposed_from_cases.formula(cases, R0, gamma, sigma, prob_symptoms)
    A = asymptomatic_from_cases.formula(cases, R0, gamma, sigma, prob_symptoms)
    S = population - I - E
    R = cases - I
    R /= prob_symptoms
    return np.array([S, E, A, I, R])
