import numpy as np
from sidekick import api as sk, placeholder as _

from pydemic import formulas
from pydemic.params.fields import MainParameter, inverse
from pydemic.params.params import Params
from pydemic.utils import not_implemented


class EpidemicParams(Params):
    """
    Common epidemiological parameters
    """

    __slots__ = ()
    R0: float = MainParameter(default=2.0)
    K: float = sk.property(not_implemented)
    duplication_time: float = sk.alias(
        "K", transform=lambda K: np.log(2) / K, prepare=lambda tau: np.log(2) / tau
    )


class SIRParams(EpidemicParams):
    """
    Declare parameters for the SIR model.
    """

    __slots__ = ()

    infectious_period: float = MainParameter(default=1.0)

    # SIR model assumes all cases are symptomatic
    prob_symptoms: float = 1.0
    Qs: float = sk.alias("prob_symptoms")

    # A null incubation period implies an infinite sigma. We instead assign
    # a very small value to avoid ZeroDivision errors
    incubation_period: float = MainParameter(default=1e-50)
    sigma: float = inverse("incubation_period")

    # The rationale for Rho defaulting to 1 is that the difference between
    # regular and asymptomatic cases is simply a matter of notification
    rho: float = 1.0

    # Derived parameters and expressions
    gamma: float = inverse("infectious_period")
    beta: float = sk.property(_.gamma * _.R0)
    K: float = sk.property(_.gamma * (_.R0 - 1))


class SEIRParams(SIRParams):
    """
    Declare parameters for the SEIR model, extending SIR.
    """

    __slots__ = ()

    sigma: float = inverse("incubation_period")
    incubation_period: float = MainParameter(default=1.0)

    @property
    def K(self):
        return formulas.K("SEIR", gamma=self.gamma, sigma=self.sigma, R0=self.R0)

    @K.setter
    def K(self, value):
        self.R0 = formulas.R0_from_K("SEIR", gamma=self.gamma, sigma=self.sigma, K=value)


class SEAIRParams(SEIRParams):
    """
    Declare parameters for the SEAIR model, extending SEIR.
    """

    __slots__ = ()

    rho: float = MainParameter(default=1.0)
    prob_symptoms: float = MainParameter(default=1.0)
    Qs: float = sk.alias("prob_symptoms")

    # Derived expressions
    @property
    def beta(self):
        gamma = self.gamma
        R0 = self.R0
        rho = self.rho
        Qs = self.Qs
        return gamma * R0 / (Qs + (1 - Qs) * rho)
