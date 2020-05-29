import numpy as np

from .abstract_seair import AbstractSEAIR
from .ode_model import ODEModel
from ..packages import click


class SEAIR(ODEModel, AbstractSEAIR):
    """
    A simple SIR model linearized around the DFE.
    """

    def diff(self, x, t):
        s, e, a, i, r = x
        n = s + e + a + i + r
        beta = self.beta
        gamma = self.gamma
        sigma = self.sigma
        rho = self.rho
        Qs = self.Qs

        infections = beta * s * ((i + rho * a) / n)
        return np.array(
            [
                -infections,
                +infections - sigma * e,
                +(1 - Qs) * sigma * e - gamma * a,
                +Qs * sigma * e - gamma * i,
                +gamma * (i + a),
            ]
        )


if __name__ == "__main__":

    @click.command()
    @click.option("--R0", "-r", default=2.74, help="Reproduction number")
    @click.option("--duration", "-t", default=90, type=int, help="Duration")
    @click.option("--log", default=False, type=bool, help="Use log-scale?")
    def cli(duration, r0, log):
        m = SEAIR()
        m.R0 = r0
        m.run(duration)
        m.plot(logy=log, show=True)

    cli()
