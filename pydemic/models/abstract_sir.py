from abc import ABC
from typing import Any

from sidekick import X
from sidekick import placeholder as _

from .model import Model
from ..packages import np, sk, integrate, pd
from ..utils import param_property, state_property, inverse_transform

Param = Any
NOT_GIVEN = object()


class AbstractSIR(Model, ABC):
    """
    Abstract base class for all SIR-based models.
    """

    class Meta:
        model_name = "SIR"
        data_aliases = {
            "S": "susceptible",
            "I": "infectious",
            "R": "recovered",
            "E": "infectious",
            "exposed": "infectious",
        }

    # Basic epidemiological parameters
    infectious_period: float = param_property("infectious_period", default=1.0)

    # Simulation state
    susceptible: float = state_property(0)
    exposed: float = sk.alias("infectious")  # an alias to infectious
    infectious: float = state_property(1)
    recovered: float = state_property(2)

    #
    # Accessors for parameters from other models
    #

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
    def _initial_state(self):
        return np.array((self.population - 1, 1, 0), dtype=float)

    def extrapolate_cases(self, which="infectious"):
        """
        Extrapolates how many cumulative cases where necessary to form the
        current number of infectious.
        """
        I = self[which]
        return I.iloc[0] * (self.beta + 1e-50) / max(self.K, 0.1)

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
