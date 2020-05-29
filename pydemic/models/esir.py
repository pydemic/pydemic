import numpy as np

from .abstract_sir import AbstractSIR


class eSIR(AbstractSIR):
    """
    A simple SIR model linearized around the DFE.
    """

    def run_to_fill(self, data, ts):
        ts = ts - self.time

        s0 = self.susceptible
        i0 = self.infectious
        r0 = self.recovered
        n = s0 + i0 + r0
        e = max(s0 / n, 0.0)

        R0 = self.R0
        gamma = self.gamma
        Ke = gamma * (e * R0 - 1)

        # Recovered and susceptible
        i = i0 * np.exp(Ke * ts)
        if abs(Ke) < 1e-6:
            x = Ke * ts
            factor = i0 * gamma * ts * (1 + x / 2 + x * x / 2)
        else:
            factor = (gamma / Ke) * np.maximum(i - i0, 0.0)
        r = np.minimum(r0 + factor, n)
        s = np.maximum(s0 - R0 * e * factor, 0.0)

        # Save data
        data[:, 0] = s
        data[:, 1] = i
        data[:, 2] = r


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--R0", "-r", default=2.74, help="Reproduction number")
    @click.option("--duration", "-t", default=90, type=int, help="Duration")
    @click.option("--steps", "-s", default=10, type=int, help="Number of steps")
    @click.option("--log", default=False, type=bool, help="Use log-scale?")
    def cli(duration, steps, r0, log):
        m = eSIR()
        m.R0 = r0

        while steps:
            step = round(duration / steps)
            m.run(step)

            steps -= 1
            duration -= step

        m.plot(logy=log, show=True)

    cli()
