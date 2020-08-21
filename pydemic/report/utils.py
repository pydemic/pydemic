import time
from collections import defaultdict
from functools import wraps

import mundi
from sidekick import api as sk


class RegionInfo:
    """
    Object that caches information about a region.
    """

    def _get_confirmed_curves(self, col):
        """
        Helper method for confirmed_cases/deaths attributes.
        """
        data = self.region.pydemic.epidemic_curve()
        self.confirmed_cases = data["cases"]
        self.confirmed_deaths = data["deaths"]
        return data[col]

    @sk.lazy
    def confirmed_cases(self):
        return self._get_confirmed_curves("cases")

    @sk.lazy
    def estimated_cases(self):
        raise NotImplementedError

    @sk.lazy
    def confirmed_deaths(self):
        return self._get_confirmed_curves("deaths")

    @sk.lazy
    def estimated_deaths(self):
        raise NotImplementedError

    def __init__(self, region):
        self.region = mundi.region(region)


class DefaultKeyDict(dict):
    """
    Dictionary that can compute default values from the keys.

    This is similar to collections.defaultdict, but the factory function receives
    the key as argument.
    """

    def __init__(self, default, data=()):
        super().__init__(data)
        self.default = default

    def __missing__(self, key):
        self[key] = value = self.default(key)
        return value


class Clock:
    """
    Simple clock that marks the number of executions and total runtime.
    """

    runtime = property(lambda self: sum(self._runs))
    n_runs = property(lambda self: len(self._runs))

    def __init__(self, clock=time.monotonic):
        self._runs = []
        self._clock = clock

    def run(self, *args, **kwargs):
        t0 = self._clock()
        func, *args = args
        result = func(*args, **kwargs)
        self._runs.append(self._clock() - t0)
        return result


class Benchmark(defaultdict):
    """
    A mapping of clock names to clocks.
    """

    def __init__(self):
        super().__init__(Clock)

    def runtime(self):
        return sum(c.runtime for c in self.values())

    def n_runs(self):
        return sum(c.n_runs for c in self.values())


def benchmarked_method(name=None, attr="_benchmarks"):
    """
    Benchmarked method clocks execution of methods

    Args:
        name:
            Clock name (extracted from method name, if not given)
        attr:
            Attribute that store all clocks in instance.
    """

    def decorator(func):
        key = func.__name__ if name is None else name

        @wraps(func)
        def decorated(self, *args, **kwargs):
            clock = getattr(self, attr)[key]
            return clock(func, self, *args, **kwargs)

        return decorated

    return decorator
