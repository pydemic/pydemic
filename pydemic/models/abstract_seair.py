from abc import ABC

from .abstract_seir import AbstractSEIR as Base
from ..packages import sk, np
from ..utils import state_property, param_property


class AbstractSEAIR(Base, ABC):
    """
    Abstract base class for all SEIR-based models.
    """

    DATA_ALIASES = {
        "S": "susceptible",
        "E": "exposed",
        "A": "asymptomatic",
        "I": "infectious",
        "R": "recovered",
    }
    model_name = "SEAIR"

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

    def get_data_force(self):
        I = self["infectious"]
        A = self["asymptomatic"]
        N = self["N"]
        return self.beta * (I + self.rho * A) / N
