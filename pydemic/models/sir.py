import numpy as np

from .abstract_sir import AbstractSIR
from .ode_model import ODEModel


class SIR(ODEModel, AbstractSIR):
    """
    A simple SIR model linearized around the DFE.
    """

    def diff(self, x, t):
        s, i, r = x
        n = s + i + r
        beta = self.beta
        gamma = self.gamma
        return np.array([-beta * s * (i / n), +beta * s * (i / n) - gamma * i, +gamma * i])


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--R0", "-r", default=2.74, help="Reproduction number")
    @click.option("--duration", "-t", default=90, type=int, help="Duration")
    @click.option("--linear", "-l", is_flag=True, help="Use log-scale?")
    def cli(duration, r0, linear):
        m = SIR(R0=r0)
        m.run(duration)
        m.plot(logy=not linear, show=True)

    cli()
