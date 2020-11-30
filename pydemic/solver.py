from abc import ABC
from collections import ChainMap
from functools import lru_cache
from types import MappingProxyType
from typing import Union, Mapping, Sequence, Callable, Tuple, FrozenSet

import numpy as np
import pandas as pd

from .types import Numeric
from .formulas import sir, seir, seair

IC = Union[Mapping[str, Numeric], Sequence[Numeric]]
ParamsF: Callable[[Union[None, float]], Mapping[str, Numeric]]
empty = MappingProxyType({})
tuple_str = lambda obj: (obj,) if isinstance(obj, str) else tuple(obj)


class Solver(ABC):
    """
    A solver class takes an initial condition, a set of parameters and a time
    frame and finds the solution to a particular model for the duration of
    the time frame.
    """

    __slots__ = ("_static_params", "_dynamic_params")
    param_names: FrozenSet[str] = frozenset()
    variables: Tuple[str, ...] = ()
    shape: Tuple[int, ...] = ()
    defaults: Mapping[str, Numeric] = MappingProxyType({})
    ndim = property(lambda self: len(self.shape))

    def __init_subclass__(cls, variables=(), shape=None, defaults=None, params=()):
        super().__init_subclass__()

        params = set(params)
        defaults = dict(defaults or {})
        variables = variables[::-1]

        for sub in cls.__bases__:
            # Defaults are included with first seen taking precedence
            items = getattr(sub, "defaults", empty).items()
            defaults.update((k, v) for k, v in items if k not in defaults)

            # Variables are inherited, avoiding repetitions. First seen are
            # included in the end of the sequence
            new = getattr(sub, "variables", ())
            extra = reversed([v for v in new if v not in variables])
            if extra:
                variables = (*variables, *extra)

            # Params are also inherited. Repetitions are avoided naturally by
            # using sets
            params.update(getattr(sub, "_params", ()))

        cls.variables = tuple(variables[::-1])
        cls.shape = tuple([len(variables)] if shape is None else shape)
        cls.defaults = MappingProxyType(dict(defaults))
        cls.param_names = frozenset({*params, *cls.defaults})

    def __init__(self, params=None, **kwargs):
        invalid = set(kwargs).difference(self.param_names)
        if invalid:
            raise TypeError(f"invalid arguments: {invalid}")

        params = ChainMap(params or {}, kwargs, self.defaults)
        self._static_params = {}
        self._dynamic_params = {}

        for k in self.param_names:
            v = params[k]
            if callable(v):
                self._dynamic_params[k] = v
            else:
                self._static_params[k] = v

        if not self._dynamic_params:
            self._dynamic_params = None

    def __getstate__(self):
        return self._static_params, self._dynamic_params

    def __setstate__(self, state):
        self._static_params, self._dynamic_params = state

    def _normalize_ic(self, ic) -> np.ndarray:
        if isinstance(ic, Mapping):
            return np.array([ic[col] for col in self.variables])

        n = len(ic)
        m = len(self.variables)
        if n < m:
            # Fill missing values with zeros
            zeros = [0] * (m - n)
            return np.array([*ic, *zeros])
        return np.array(ic)

    def run(self, ic: IC, steps: int, t0: float = 0.0, dt: float = 1.0) -> pd.DataFrame:
        """
        Runs the model for the given duration and return an array with the
        results.

        Args:
            ic:
                Vector or dictionary with the initial condition.
            steps:
                Number of simulation steps.
            t0:
                Initial simulation time.
            dt:
                Fixed time step duration.
        """
        ts = t0 + dt * np.arange(steps + 1)
        data = np.zeros((steps + 1, *self.shape), dtype=float)
        data[0] = self._normalize_ic(ic)
        self.run_to_fill(data, ts)

        if self.ndim == 1:
            return pd.DataFrame(data, columns=self.variables, index=ts)
        else:
            raise NotImplementedError

    def run_to_fill(self, data, times) -> None:
        """
        Run simulation to fill pre-allocated array of data.
        """
        raise NotImplementedError("must be implemented in subclasses")

    def copy(self, **kwargs):
        """
        Return a copy of solver, possibly overwriting some parameters.
        """
        new = object.__new__(type(self))
        params, *rest = self.__getstate__()
        params = {**self.params, **kwargs}
        new.__setstate__((params, *rest))
        return new

    def params(self, t):
        """
        Get a mapping of parameters at time t.
        """

        if self._dynamic_params is None:
            return {**self._static_params}
        out = {k: fn(t) for k, fn in self._dynamic_params.items()}
        out.update(self._static_params)
        return out


class ODESolver(Solver, ABC):
    """
    Base class for all models that uses ordinary differential equations.
    """

    __slots__ = ("substeps", "method", "_fast_params")

    def __init__(self, params=None, substeps=4, method="RK4", **kwargs):
        super().__init__(params, **kwargs)
        self.substeps = substeps
        self.method = method
        if self._dynamic_params:
            self._fast_params = lru_cache(16)(self.params)
        else:
            params = self._static_params
            self._fast_params = lambda _: params

    def diff(self, params, x: np.ndarray, t: float) -> np.ndarray:
        """
        Derivative function for the ODE.
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    def integration_step(self, x, t, dt, method=None):
        """
        A single RK4 iteration step.
        """
        method = method or self.method
        if method == "RK4":
            t1 = t + 0.5 * dt
            t2 = t + dt

            p0 = self._fast_params(t)
            p1 = self._fast_params(t1)
            p2 = self._fast_params(t2)

            k1 = self.diff(p0, x, t)
            k2 = self.diff(p1, x + 0.5 * dt * k1, t)
            k3 = self.diff(p1, x + 0.5 * dt * k2, t1)
            k4 = self.diff(p2, x + 1.0 * dt * k3, t2)

            return x + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)
        elif method == "Euler":
            return x + self.diff(self._fast_params(t), x, t) * dt
        else:
            raise ValueError(f"unknown integration method: {method!r}")

    def run_to_fill(self, data, times: np.ndarray):
        x = data[0]
        t = times[0]

        for i, dt in enumerate(np.diff(times), 1):
            dt_ = dt / self.substeps
            t_ = t
            for _ in range(self.substeps):
                x = self.integration_step(x, t_, dt_)
                t_ += dt_
            t += dt
            data[i] = x
        return x


class eSIRSolver(
    Solver, variables=["susceptible", "infectious", "recovered"], defaults={"R0": 1.0, "gamma": 1.0}
):
    """
    A simple SIR model linearized around the DFE.
    """

    __slots__ = ()

    def run_to_fill(self, data, ts):
        # Only static params are accepted
        params = self.params(None)
        R0 = params["R0"]
        gamma = params["gamma"]

        # No inplace operation to avoid modifying argument
        ts = ts - ts[0]

        s0, i0, r0 = data[0]
        n = s0 + i0 + r0
        e = max(s0 / n, 0.0)
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


class SIRSolver(
    ODESolver,
    variables=["susceptible", "infectious", "recovered"],
    defaults={"R0": 1.0, "gamma": 1.0},
):
    """
    A simple SIR model.
    """

    __slots__ = ()
    _beta = staticmethod(sir.beta)

    def diff(self, params, x, t):
        R0 = params["R0"]
        gamma = params["gamma"]
        beta = R0 * gamma  # See formula

        s, i, r = x
        n = s + i + r
        return np.array([-beta * s * (i / n), +beta * s * (i / n) - gamma * i, +gamma * i])


class SEIRSolver(
    ODESolver,
    variables=["susceptible", "exposed", "infectious", "recovered"],
    defaults={"sigma": 1.0, **SIRSolver.defaults},
):
    """
    A simple SEIR solver.
    """

    __slots__ = ()
    _beta = staticmethod(seir.beta)

    def diff(self, params, x, t):
        R0 = params["R0"]
        gamma = params["gamma"]
        sigma = params["sigma"]
        beta = R0 * gamma  # See formula

        s, e, i, r = x
        n = s + e + i + r

        return np.array(
            [
                -beta * s * (i / n),
                +beta * s * (i / n) - sigma * e,
                +sigma * e - gamma * i,
                +gamma * i,
            ]
        )


class SEAIRSolver(
    ODESolver,
    variables=["susceptible", "exposed", "asymptomatic", "infectious", "recovered"],
    defaults={"prob_symptoms": 1.0, "rho": 1.0, **SEIRSolver.defaults},
):
    """
    A simple SIR model linearized around the DFE.
    """

    __slots__ = ()
    _beta = staticmethod(seair.beta.formula)

    def diff(self, params, x, t):
        R0 = params["R0"]
        gamma = params["gamma"]
        sigma = params["sigma"]
        rho = params["rho"]
        Qs = params["prob_symptoms"]
        beta = self._beta(R0, gamma, Qs, rho)

        s, e, a, i, r = x
        n = s + e + a + i + r
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


#
# Imperfect immunity
#
class SRSMixin(ODESolver, defaults={"immunity_period": float("inf")}):
    """
    Transforms ODE model in variant with imperfect immunity.
    It includes transitions R -> S.
    """

    __slots__ = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        names = ["susceptible", "recovered"]
        cls._sr_ids = list(map(cls.variables.index, names))

    def diff(self, params, x, t):
        s, _ = x[self._sr_ids]

        diff = super().diff(params, x, t)
        rate = (1 / params["immunity_period"]) * s
        diff[self._sr_ids] += [rate, -rate]
        return diff


class SIRSSolver(SRSMixin, SIRSolver):
    """
    SIR variant with imperfect immunity.
    """

    __slots__ = ()


class SEIRSSolver(SRSMixin, SEIRSolver):
    """
    SEIR variant with imperfect immunity.
    """

    __slots__ = ()


class SEAIRSSolver(SRSMixin, SEAIRSolver):
    """
    SEAIR variant with imperfect immunity.
    """

    __slots__ = ()


#
# Vaccination
#
class Vac1Mixin(
    ODESolver, defaults={"prob_immunity": 1.0, "immunization_period": 1.0, "immunization_rate": 0.0}
):
    """
    Mixin that adds a simple vaccination strategy to a model with
    susceptibles and recovered.

    It adds "vaccinated" and "protected" compartments.
    """

    __has_init_subclass = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__has_init_subclass:
            return

        unprotected = [v for v in cls.variables if v != "protected"]
        vaccinated = [v + "_v" for v in unprotected]
        cls.variables = (*cls.variables, *vaccinated)
        if "protected" not in cls.variables:
            cls.variables = (*cls.variables, "protected")

        names = ["protected"]
        cls._p_ids = list(map(cls.variables.index, names))
        cls._not_vaccinated_ids = list(map(cls.variables.index, unprotected))
        cls._vaccinated_ids = list(map(cls.variables.index, vaccinated))
        cls.__has_init_subclass = True

    def diff(self, params, x, t):
        v, p = x[self._vp_ids]
        n = x.sum()

        rate = params["immunization_rate"]
        Qv = params["prob_immunity"]
        gamma_v = 1.0 / params["immunization_period"]

        partial = super().diff(params, x[:-1], t)
        diff = np.array([*partial, 0.0])
        diff[self._srvp_ids] += [(1 - Qv) * gamma_v * v - vrate, vrate, Qv * gamma_v * v]

        return diff


class SIRVSolver(Vac1Mixin, SIRSolver):
    __slots__ = ()


class SIRSVSolver(Vac1Mixin, SIRSSolver):
    __slots__ = ()


# print(SIRSVSolver.variables)
# SIRSVSolver(R0=2, immunization_rate=0.05, gamma=0.15, immunization_period=20).run(
#     [1, 0.1], 100, dt=0.5).plot()
# from matplotlib.pyplot import show;
#
# show()
