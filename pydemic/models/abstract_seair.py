from abc import ABC

from .abstract_seir import AbstractSEIR as Base
from ..packages import sk, np
from ..utils import state_property, param_property


class AbstractSEAIR(Base, ABC):
    """
    Abstract base class for all SEIR-based models.
    """

    DATA_ALIASES = {
        'S': 'susceptible',
        'E': 'exposed',
        'A': 'asymptomatic',
        'I': 'infectious',
        'R': 'recovered',
    }

    # Basic epidemiological parameters
    rho = param_property('rho')
    prob_symptomatic = param_property('prob_symptomatic')
    qs = sk.alias('prob_symptomatic')

    # Derived expressions
    @property
    def beta(self):
        gamma = self.gamma
        R0 = self.R0
        rho = self.rho
        qs = self.qs
        return gamma * R0 / (qs + (1 - qs) * rho)

    # Simulation state
    susceptible = state_property(0)
    exposed = state_property(1)
    asymptomatic = state_property(2)
    infectious = state_property(3)
    recovered = state_property(4)

    def set_ic(self, vector=(1e6 - 1, 1, 0, 0, 0), **kwargs):
        vector = np.array(vector, dtype=float)
        super().set_ic(vector, **kwargs)
