import numpy as np
import pandas as pd
import statsmodels.api as sm

from .smoothing import smoothed_diff


def time_dependent_K(cases: pd.Series, method="smoothing", **kwargs):
    """
    Compute a time-dependent exponential growth factor from the given times
    series of cases, deaths or some strongly correlated proxy.

    Args:
        cases:
            Time-series of cases.

    Returns:
        A DataFrame with ["value", "low", "high"] columns. The  main inference
        is in the "value" interval. If the chosen method does not compute
        confidence intervals, low and high will contain only NaNs.
    """
    compute = TimeDependentK(cases)
    return compute(method, **kwargs)


#
# Class-based facade to smoothing methods
#
class TimeDependentK:
    """
    A class that implements inference methods for time-dependent K as instance
    methods in the class.
    """

    def __init__(self, cases):
        self.cases = cases
        self.dates = cases.index
        self.data = cases.values

    def __call__(self, method, **kwargs):
        fn = getattr(self, "method_" + method)
        out = fn(**kwargs)
        return pd.Series(out, index=self.dates)

    def method_smoothing(self, **kwargs):
        res = naive_holt_smoothing(self.data, **kwargs)
        return res.fittedvalues


#
# Private implementations of smoothing methods.
# Those methods do not pre-process input or post-process outputs and generally
# work with raw numpy arrays.
#
def naive_holt_smoothing(ys, max_daily_growth=2, **kwargs):
    """
    A double-smoothing method.
    """

    max_factor = np.log(max_daily_growth)
    deltas = smoothed_diff(ys, **kwargs)
    dirty_ks = np.minimum(np.diff(np.log(deltas), prepend=0), max_factor)

    smoothing_level = kwargs.setdefault("smoothing_level", 0.1)
    holt = sm.tsa.Holt(dirty_ks, damped=True)
    return holt.fit(smoothing_level=smoothing_level)
