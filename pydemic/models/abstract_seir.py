import numpy as np

from .abstract_sir import AbstractSIR as Base
from .. import params
from ..utils import state_property, param_property


class AbstractSEIR(Base):
    """
    Abstract base class for all SEIR-based models.
    """

    DATA_ALIASES = {
        'S': 'susceptible',
        'E': 'exposed',
        'I': 'infectious',
        'R': 'recovered',
    }

    # Basic epidemiological parameters
    params = params.epidemic.DEFAULT
    sigma = param_property('sigma')
    incubation_period = param_property('incubation_period')

    # Derived expressions
    @property
    def K(self):
        g = self.gamma
        s = self.sigma
        R0 = self.R0
        return 0.5 * (s + g) * (np.sqrt(1 + 4 * (R0 - 1)) - 1)

    # Simulation state
    susceptible = state_property(0)
    exposed = state_property(1)
    infectious = state_property(2)
    recovered = state_property(3)

    def set_ic(self, vector=(1e6 - 1, 1, 0, 0), **kwargs):
        vector = np.array(vector, dtype=float)
        super().set_ic(vector, **kwargs)
