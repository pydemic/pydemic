from types import SimpleNamespace

import sidekick as sk
from sidekick import placeholder as _

from .param import Params


class EpidemicParams(Params):
    R0: float
    infectious_period: float
    incubation_period: float = 0.0
    gamma = sk.property(1 / _.infectious_period)
    sigma = sk.property(1 / _.incubation_period)


#
# Epidemiological parameters
#
EPIDEMIC_DEFAULT = ep = EpidemicParams(
    "Default", R0=2.74, rho=0.55, prob_symptoms=0.14, incubation_period=3.69, infectious_period=3.47
)

epidemic = SimpleNamespace(
    DEFAULT=EPIDEMIC_DEFAULT,
    DEFAULT_SIR=EpidemicParams(
        "SIR Default", R0=ep.R0, infectious_period=ep.incubation_period + ep.infectious_period
    ),
)
