import numpy as np
import pandas as pd

from .smoothing import smoothed_diff
from ..formulas import formula, get_function

FORMULA_OPTS = {
    "positional": 1,
    "options": ("dt",),
    "invalid": "pass",
    "formula_name": "epidemic_curve",
}


@formula(positional=1, invalid="pass", options=("smooth",))
def infectious_curve(cases: pd.DataFrame, gamma=None, smooth=True, **kwargs):
    """
    Infer the infectious curve from the curve of cumulative cases.

    Return the "(I)nfectious" compartment from case data by propagating the
    number of new cases to the whole duration of its infectious period.

    This transformation is necessary because neither the cumulative cases nor the
    number of new cases per day correspond directly to any compartment in the
    traditional Differential equation models.

    Args:
        cases:
            Curve with the accumulative number of cases.
        gamma:
            Inverse of infectious period. Can be a number or an object that
            return parameters using the :func:`pydemic.params.get_param`
            function.
        smooth:
            If False, skip smoothing step.
    """
    cases = np.asarray(cases)
    N = len(cases)
    infectious = np.zeros(N)

    if smooth:
        new_cases = smoothed_diff(cases, **kwargs)
    else:
        new_cases = np.diff(cases, prepend=0.0)

    for i, x in enumerate(new_cases):
        infectious[i:] += x * np.exp(-gamma * np.arange(0, N - i))

    return infectious


def epidemic_curve(model, cases, *args, **kwargs):
    """
    Return the epidemic curve from cases for the given model.

    Args:
        model:
            A string like "SIR", "SEIR" or "SEAIR" identifying model type.
        cases:
            A dataframe of "cases" and "deaths" columns with cumulative values..

    Keyword Args:
        Keyword arguments are passed to the work function that implements
        each model.

    Returns:
        A dataframe that can be used to initialize model.

    See Also:
        :func:`sir_curves`
        :func:`seir_curves`
        :func:`seair_curves`
    """
    fn = get_function(model, "epidemic_curve")
    return fn(cases, *args, **kwargs)


@formula("SIR", **FORMULA_OPTS)
def sir_curves(cases, gamma, population, dt=1, **kwargs):
    """
    Infer SIR curves from experimental data.
    """

    N = len(cases)
    infectious = infectious_curve(cases, gamma=gamma, **kwargs)

    # Allocate data and initialize with values
    data = np.zeros((N, 3), dtype=float)
    iS, iI, iR = range(3)
    data[0, iS] = population
    data[:, iI] = infectious

    for i, row in enumerate(data[:-1], start=1):
        S, I, R = row
        data[i, iR] = R + gamma * I * dt
        data[i, iS] = population - data[i, 1:].sum()

    columns = ("susceptible", "infectious", "recovered")
    return pd.DataFrame(data, columns=columns, index=cases.index)


@formula("SEIR", **FORMULA_OPTS)
def seir_curves(cases, gamma, sigma, population, dt=1, **kwargs):
    """
    Infer SEIR curves from experimental data.
    """

    N = len(cases)
    infectious = infectious_curve(cases, gamma=gamma, **kwargs)

    # Allocate data and initialize with values
    data = np.zeros((N, 4), dtype=float)
    iS, iE, iI, iR = range(4)
    data[0, iS] = population
    data[:, iE] = infectious * (sigma / gamma)
    data[:, iI] = infectious

    for i, row in enumerate(data[:-1], start=1):
        S, E, I, R = row
        data[i, iR] = R + gamma * I * dt
        data[i, iS] = population - data[i, 1:].sum()

    columns = ("susceptible", "exposed", "infectious", "recovered")
    return pd.DataFrame(data, columns=columns, index=cases.index)


@formula("SEAIR", **FORMULA_OPTS)
def seair_curves(cases, gamma, sigma, prob_symptoms, rho, population, dt=1, **kwargs):
    """
    Infer SEAIR curves from experimental data.
    """

    N = len(cases)
    infectious = infectious_curve(cases, gamma=gamma, **kwargs)

    # Allocate data and initialize with values
    data = np.zeros((N, 5), dtype=float)
    iS, iE, iA, iI, iR = range(5)
    data[0, iS] = population
    data[:, iE] = infectious * (sigma / gamma / prob_symptoms)
    data[:, iA] = infectious * (1 - prob_symptoms) / prob_symptoms
    data[:, iI] = infectious

    for i, row in enumerate(data[:-1], start=1):
        S, E, A, I, R = row
        data[i, iR] = R + gamma * (I + A) * dt
        data[i, iS] = population - data[i, 1:].sum()

    columns = ("susceptible", "exposed", "asymptomatic", "infectious", "recovered")
    return pd.DataFrame(data, columns=columns, index=cases.index)
