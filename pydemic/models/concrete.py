from .abstract import AbstractSIR, AbstractSEIR, AbstractSEAIR
from .. import solver


class eSIR(AbstractSIR):
    """
    A simple SIR model linearized around the DFE.
    """

    class Meta:
        solver_model = solver.eSIRSolver


class SIR(AbstractSIR):
    """
    A simple SIR model linearized around the DFE.
    """

    class Meta:
        solver_model = solver.SIRSolver


class SEIR(AbstractSEIR):
    """
    A simple SIR model linearized around the DFE.
    """

    class Meta:
        solver_model = solver.SEIRSolver


class SEAIR(AbstractSEAIR):
    """
    A simple SIR model linearized around the DFE.
    """

    class Meta:
        solver_model = solver.SEAIRSolver
