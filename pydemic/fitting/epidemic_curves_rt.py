import numpy as np
import pandas as pd

from .epidemic_curves import infectious_curve
from .. import formulas
from ..formulas import formula

CURVE_OPTIONS = ("dt", "Rt_smooth", "smooth_days", "ret_Rt")


@formula("SIR", positional=1, options=CURVE_OPTIONS, invalid="pass")
def sir_curves_Rt(
    cases, gamma, R0, population, dt=1, Rt_smooth=0.2, smooth_days=3, ret_Rt=False, **kwargs
):
    """
    Infer SIR curves from experimental data.
    """
    N = len(cases)

    infectious = infectious_curve(cases, gamma, **kwargs)
    data = np.zeros((N, 3), dtype=float)
    data[0, 0] = population
    data[:, 1] = infectious

    Rt = [1.0] if R0 is None else [R0]
    beta = formulas.beta("SIR", R0=Rt[0], gamma=gamma)

    for i, row in enumerate(data[:-1], start=1):
        S, I, R = row

        if Rt_smooth != 1:
            e = 1e-9
            xs = infectious[i - min(i, smooth_days) : i + 1]
            growth = np.diff(np.log(xs + e)).mean()
            beta_new = (growth + gamma) * (population / S)
            beta = (1 - Rt_smooth) * beta_new + Rt_smooth * beta

        Rt.append(beta / gamma)
        data[i, 2] = R + gamma * I * dt
        data[i, 0] = population - data[i, 1:].sum()

    columns = ("susceptible", "infectious", "recovered")
    df = pd.DataFrame(data, columns=columns, index=cases.index)
    if ret_Rt:
        df["Rt"] = Rt
    return df


@formula("SEIR", positional=1, options=CURVE_OPTIONS, invalid="pass")
def seir_curves_Rt(
    cases, gamma, sigma, R0, population, dt=1, Rt_smooth=0.2, smooth_days=3, ret_Rt=False, **kwargs
):
    """
    Infer SEIR curves from experimental data.
    """
    N = len(cases)

    infectious = infectious_curve(cases, gamma=gamma, **kwargs)
    data = np.zeros((N, 4), dtype=float)
    data[0, 0] = population
    data[:, 1] = infectious * (sigma / gamma)
    data[:, 2] = infectious

    Rt = [1.0] if R0 is None else [R0]
    beta = formulas.beta("SEIR", R0=Rt[0], gamma=gamma, sigma=sigma)

    for i, row in enumerate(data[:-1], start=1):
        S, E, I, R = row

        if Rt_smooth != 1:
            e = 1e-9
            xs = infectious[i - min(i, smooth_days) : i + 1]
            growth = np.diff(np.log(xs + e)).mean()
            beta_new = ((growth + sigma) * E / (I + e)) * (population / S)
            beta = (1 - Rt_smooth) * beta_new + Rt_smooth * beta

        Rt.append(beta / gamma)

        if R0 is not None:
            force = beta * I * (S / population)
            data[i, 1] = E + (force - sigma * E) * dt

        data[i, 3] = R + gamma * I * dt
        data[i, 0] = population - data[i, 1:].sum()

    columns = ("susceptible", "exposed", "infectious", "recovered")
    df = pd.DataFrame(data, columns=columns, index=cases.index)
    if ret_Rt:
        df["Rt"] = Rt
    return df


@formula("SEAIR", positional=1, options=CURVE_OPTIONS, invalid="pass")
def seair_curves_Rt(
    cases,
    gamma,
    sigma,
    prob_symptoms,
    rho,
    R0,
    population,
    dt=1,
    Rt_smooth=1.0,
    smooth_days=3,
    ret_Rt=False,
    **kwargs
):
    """
    Infer SEAIR curves from experimental data.
    """

    N = len(cases)
    infectious = infectious_curve(cases, gamma=gamma, **kwargs)

    # Allocate data and initialize with values
    data = np.zeros((N, 5), dtype=float)
    S, E, A, I, R = range(5)
    data[0, S] = population
    data[:, E] = infectious * (sigma / gamma / prob_symptoms)
    data[:, A] = infectious * (1 - prob_symptoms) / prob_symptoms
    data[:, I] = infectious

    Rt = [1.0] if R0 is None else [R0]
    formula_args = dict(gamma=gamma, sigma=sigma, rho=rho, prob_symptoms=prob_symptoms)
    beta = formulas.beta("SEAIR", R0=Rt[0], **formula_args)

    for i, row in enumerate(data[:-1], start=1):
        S, E, A, I, R = row

        if Rt_smooth != 1:
            e = 1e-9
            xs = infectious[i - min(i, smooth_days) : i + 1]
            growth = np.diff(np.log(xs + e)).mean()
            beta_new = ((growth + sigma) * E / (I + rho * A + e)) * (population / S)
            beta = (1 - Rt_smooth) * beta_new + Rt_smooth * beta

        Rt.append(formulas.R0("SEAIR", beta=beta, **formula_args))

        if R0 is not None:
            force = beta * (S / population) * (I + rho * A)
            data[i, 1] = E + (force - sigma * E) * dt

        data[i, 4] = R + gamma * (I + A) * dt
        data[i, 0] = population - data[i, 1:].sum()

    columns = ("susceptible", "exposed", "asymptomatic", "infectious", "recovered")
    df = pd.DataFrame(data, columns=columns, index=cases.index)
    if ret_Rt:
        df["Rt"] = Rt
    return df
