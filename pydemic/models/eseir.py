from .abstract_seir import AbstractSEIR


class eSEIR(AbstractSEIR):
    """
    A simple SEIR model linearized around the DFE.
    """

    def run_to_fill(self, data, ts):
        raise NotImplementedError
