import numpy as np
import pandas as pd
from scipy import integrate
from sidekick import X

from .abstract_sir import AbstractSIR as Base
from .. import params
from ..formulas import K, R0_from_K
from ..utils import state_property, param_property, param_transform


class AbstractSEIR(Base):
    """
    Abstract base class for all SEIR-based models.
    """

    DATA_ALIASES = {"S": "susceptible", "E": "exposed", "I": "infectious", "R": "recovered"}
    model_name = "SEIR"

    # Basic epidemiological parameters
    params = params.epidemic.DEFAULT
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

    def get_data_cases(self):
        # SEIR makes a distinction between "Infectious" and "Exposed". Different
        # diseases may have different clinical evolutions, but it is reasonable
        # to expect that in many situations, individuals only manifest symptoms
        # after becoming infectious.
        #
        # We therefore changed the definition of "cases" in SEIR to be simply
        # the number of elements that enter the "I" compartment.
        E = self["exposed"]
        res = integrate.cumtrapz(self.Qs * self.sigma * E, self.times, initial=0.0)
        return pd.Series(res + self.initial_cases, index=E.index)
