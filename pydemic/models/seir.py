import numpy as np
from ..packages import click

from .abstract_seir import AbstractSEIR
from .ode_model import ODEModel


class SEIR(ODEModel, AbstractSEIR):
    """
    A simple SIR model linearized around the DFE.
    """

    def diff(self, x, t):
        s, e, i, r = x
        n = s + e + i + r
        beta = self.beta
        gamma = self.gamma
        sigma = self.sigma

        return np.array(
            [
                -beta * s * (i / n),
                +beta * s * (i / n) - sigma * e,
                +sigma * e - gamma * i,
                +gamma * i,
            ]
        )


if __name__ == "__main__":

    @click.command()
    @click.option("--R0", "-r", default=2.74, help="Reproduction number")
    @click.option("--duration", "-t", default=90, type=int, help="Duration")
    @click.option("--log", default=False, type=bool, help="Use log-scale?")
    def cli(duration, r0, log):
        m = SEIR()
        m.R0 = r0
        m.run(duration)
        m.plot(logy=log, show=True)

    cli()
