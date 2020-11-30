import datetime
from abc import ABC
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from scipy import integrate
from sidekick import X
from sidekick import placeholder as _

from .model import Model
from ..formulas import K, R0_from_K
from ..packages import np, sk, integrate, pd
from ..utils import param_property, state_property, inverse_transform
from ..utils import param_transform
from ..params.model_params import SIRParams, SEIRParams, SEAIRParams

Param = Any
T = TypeVar("T")
NOT_GIVEN = object()
DAY = datetime.timedelta(days=1)


class AbstractSIR(Model, ABC):
    """
    Abstract base class for all SIR-based models.
    """

    class Meta:
        model_name = "SIR"
        params = SIRParams()
        data_aliases = {
            "S": "susceptible",
            "I": "infectious",
            "R": "recovered",
            "E": "infectious",
            "exposed": "infectious",
        }

    # Simulation state
    susceptible: float = state_property(0)
    exposed: float = sk.alias("infectious")  # an alias to infectious
    infectious: float = state_property(1)
    recovered: float = state_property(2)

    #
    # Accessors for parameters from other models
    #

    # Basic epidemiological parameters
    infectious_period: float = param_property("infectious_period", default=1.0)

    # SIR model assumes all cases are symptomatic
    prob_symptoms: float = 1.0
    Qs: float = sk.alias("prob_symptoms", mutable=True)

    # A null incubation period implies an infinite sigma. We instead assign
    # a very small value to avoid ZeroDivision errors
    incubation_period: float = 1e-50
    sigma: float = sk.alias("incubation_period", transform=(1 / X), prepare=(1 / X))

    # The rationale for Rho defaulting to 1 is that the difference between
    # regular and asymptomatic cases is simply a matter of notification
    rho: float = 1.0

    # Derived parameters and expressions
    gamma: float = inverse_transform("infectious_period")
    beta: float = sk.property(_.gamma * _.R0)
    K: float = sk.property(_.gamma * (_.R0 - 1))

    #
    # Model API
    #
    def extrapolate_cases(self, which="infectious"):
        """
        Extrapolates how many cumulative cases where necessary to form the
        current number of infectious.
        """
        I = self[which]
        return I.iloc[0] * (self.beta + 1e-50) / max(self.K, 0.1)

    def _initial_state(self):
        return np.array((self.population - 1, 1, 0), dtype=float)

    def _initial_infected(self):
        return self.extrapolate_cases()

    def _initial_cases(self, col="infectious"):
        return self.extrapolate_cases() * self.Qs

    #
    # Process data
    #
    def get_data_force(self, idx):
        I = self["infectious", idx]
        N = self["N", idx]
        return self.beta * (I / N)

    def get_data_resolved_cases(self, idx):
        I = self["infectious", idx]
        res = integrate.cumtrapz(I, self.times, initial=0.0)
        return pd.Series(res * self.gamma, index=I.index)

    def get_data_cases(self, idx):
        infections = self["force"] * self["susceptible"] * self.Qs
        res = integrate.cumtrapz(infections, self.times, initial=0.0)
        data = pd.Series(res + self.initial_cases, index=infections.index)
        return data[idx] if idx is not None else data

    def get_data_infected(self, idx):
        infections = self["force"] * self["susceptible"]
        res = integrate.cumtrapz(infections, self.times, initial=0.0)
        data = pd.Series(res + self.initial_infected, index=infections.index)
        return data[idx] if idx is not None else data


class AbstractSEIR(AbstractSIR):
    """
    Abstract base class for all SEIR-based models.
    """

    class Meta:
        model_name = "SEIR"
        params = SEIRParams()
        data_aliases = {"E": "exposed", "exposed": None}

    # Basic epidemiological parameters
    sigma = param_transform("incubation_period", (1 / X), (1 / X))
    incubation_period = param_property("incubation_period", default=1.0)

    # Derived expressions
    @property
    def K(self):
        return K("SEIR", gamma=self.gamma, sigma=self.sigma, R0=self.R0)

    @K.setter
    def K(self, value):
        self.R0 = R0_from_K("SEIR", gamma=self.gamma, sigma=self.sigma, K=value)

    # Simulation state
    susceptible = state_property(0)
    exposed = state_property(1)
    infectious = state_property(2)
    recovered = state_property(3)

    def _initial_state(self):
        return np.array((self.population - 1, 1, 0, 0), dtype=float)

    def _initial_infected(self):
        return super().extrapolate_cases("exposed")

    def get_data_cases(self, idx):
        # SEIR makes a distinction between "Infectious" and "Exposed". Different
        # diseases may have different clinical evolutions, but it is reasonable
        # to expect that in many situations, individuals only manifest symptoms
        # after becoming infectious.
        #
        # We therefore changed the definition of "cases" in SEIR to be simply
        # the number of elements that enter the "I" compartment.
        E = self["exposed"]
        res = integrate.cumtrapz(self.Qs * self.sigma * E, self.times, initial=0.0)
        data = pd.Series(res + self.initial_cases, index=E.index)
        return data[idx] if idx is not None else data


class AbstractSEAIR(AbstractSEIR, ABC):
    """
    Abstract base class for all SEIR-based models.
    """

    class Meta:
        model_name = "SEAIR"
        params = SEAIRParams()
        data_aliases = {"A": "asymptomatic"}

    # Basic epidemiological parameters
    rho = param_property("rho", default=1.0)
    prob_symptoms = param_property("prob_symptoms", default=1.0)
    Qs = sk.alias("prob_symptoms", mutable=True)

    # Derived expressions
    @property
    def beta(self):
        gamma = self.gamma
        R0 = self.R0
        rho = self.rho
        Qs = self.Qs
        return gamma * R0 / (Qs + (1 - Qs) * rho)

    # Simulation state
    susceptible = state_property(0)
    exposed = state_property(1)
    asymptomatic = state_property(2)
    infectious = state_property(3)
    recovered = state_property(4)

    def _initial_state(self):
        return np.array((self.population - 1, 1, 0, 0, 0), dtype=float)

    def get_data_force(self, idx):
        I = self["infectious", idx]
        A = self["asymptomatic", idx]
        N = self["N", idx]
        return self.beta * (I + self.rho * A) / N
