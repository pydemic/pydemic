from .abstract import AbstractSIR, AbstractSEIR, AbstractSEAIR
from .. import solver


class eSIR(AbstractSIR):
    """
    A simple Susceptible Infectious Recovered model linearized around the DFE.
    """

    class Meta:
        solver_model = solver.eSIRSolver


class SIR(AbstractSIR):
    """
    A simple Susceptible Infectious Recovered model.
    """

    class Meta:
        solver_model = solver.SIRSolver


class SEIR(AbstractSEIR):
    """
    A simple Susceptible Exposed Infectious Recovered model.
    """

    class Meta:
        solver_model = solver.SEIRSolver


class SEAIR(AbstractSEAIR):
    """
    A simple Susceptible Exposed Asymptomatic Infectious Recovered model.
    """

    class Meta:
        solver_model = solver.SEAIRSolver
