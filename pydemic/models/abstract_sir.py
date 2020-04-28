from abc import ABC

from sidekick import placeholder as _

from pydemic.utils.properties import inverse_transform
from .model import Model
from .. import params
from ..packages import np, sk, integrate, pd
from ..utils import param_property, state_property


class AbstractSIR(Model, ABC):
    """
    Abstract base class for all SIR-based models.
    """

    DATA_ALIASES = {
        'S': 'susceptible',
        'I': 'infectious',
        'R': 'recovered',
    }

    # Basic epidemiological parameters
    params = params.epidemic.DEFAULT_SIR
    R0 = param_property('R0')
    infectious_period = param_property('infectious_period')

    # Derived parameters and expressions
    gamma = inverse_transform('infectious_period')
    beta = sk.property(_.gamma * _.R0)
    K = sk.property(_.gamma * (_.R0 - 1))

    # Simulation state
    susceptible = state_property(0)
    exposed = state_property(1)  # an alias to infectious
    infectious = state_property(1)
    recovered = state_property(2)

    def set_ic(self, vector=(1e6 - 1, 1, 0), **kwargs):
        vector = np.array(vector, dtype=float)
        super().set_ic(vector, **kwargs)

    def get_data_N(self):
        return self.data.sum(1)

    def get_data_force(self):
        I = self["infectious"]
        N = self["N"]
        return self.beta * (I / N)

    def get_data_resolved_cases(self):
        I = self["infectious"]
        res = integrate.cumtrapz(I, self.times, initial=I.iloc[0])
        return pd.Series(res * self.gamma, index=I.index)

    def get_data_cases(self):
        i0 = np.sum(self.data.iloc[0].infectious)
        infections = self["force"] * self["susceptible"]
        res = integrate.cumtrapz(infections, self.times, initial=i0)
        return pd.Series(res, index=infections.index)
