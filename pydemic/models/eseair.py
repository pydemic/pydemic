from .abstract_seair import AbstractSEAIR


class eSEAIR(AbstractSEAIR):
    """
    A simple SEAIR model linearized around the DFE.
    """

    def run_to_fill(self, data, ts):
        raise NotImplementedError
