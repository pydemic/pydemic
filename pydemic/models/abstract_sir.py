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

    DATA_ALIASES = {"S": "susceptible", "I": "infectious", "R": "recovered"}
    model_name = "SIR"

    # Basic epidemiological parameters
    R0: float = param_property("R0", default=2.0)
    infectious_period: float = param_property("infectious_period", default=1.0)

    # Simulation state
    susceptible: float = state_property(0)
    exposed: float = state_property(1)  # an alias to infectious
    infectious: float = state_property(1)
    recovered: float = state_property(2)

    initial_cases: float = sk.lazy(lambda self: self._initial_cases())
    initial_infected: float = sk.lazy(lambda self: self._initial_infected())

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
    def get_data_N(self):
        return self.data.sum(1)

    def get_data_force(self):
        I = self["infectious"]
        N = self["N"]
        return self.beta * (I / N)

    def get_data_resolved_cases(self):
        I = self["infectious"]
        res = integrate.cumtrapz(I, self.times, initial=0.0)
        return pd.Series(res * self.gamma, index=I.index)

    def get_data_cases(self):
        infections = self["force"] * self["susceptible"] * self.Qs
        res = integrate.cumtrapz(infections, self.times, initial=0.0)
        return pd.Series(res + self.initial_cases, index=infections.index)

    def get_data_infected(self):
        infections = self["force"] * self["susceptible"]
        res = integrate.cumtrapz(infections, self.times, initial=0.0)
        return pd.Series(res + self.initial_infected, index=infections.index)
