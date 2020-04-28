from abc import ABC

import numpy as np

from .model import Model


class ODEModel(Model, ABC):
    """
    Base class for all models that uses ordinary differential equations.
    """

    integration_method = "RK4"
    sub_steps = 4

    def diff(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Derivative function for the ODE.

        Must be implemented in subclasses.
        """
        raise NotImplementedError

    def integration_step(self, x, t, dt, method=None):
        """
        A single RK4 iteration step.
        """
        method = method or self.integration_method
        if method == "RK4":
            k1 = self.diff(x, t)
            k2 = self.diff(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = self.diff(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = self.diff(x + 1.0 * dt * k3, t + 1.0 * dt)
            return x + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)
        else:
            raise ValueError(f"unknown integration method: {method!r}")

    def run_to_fill(self, data, times):
        x = self.state
        for i, t in enumerate(times):
            dt = (t - self.time) / self.sub_steps
            for j in range(self.sub_steps):
                x = self.integration_step(x, t, dt)
            self.time = t
            self.state = x
            data[i] = x
        return x
