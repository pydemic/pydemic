import time
from collections import Counter
from contextlib import contextmanager
from functools import wraps

import pandas as pd

PERF_LOG = []
PERF_TRACK = True


@contextmanager
def timeit():
    """
    A timer context manager.

    Examples:
        >>> with timeit() as timer:
        ...     time.sleep(5)  # Do something expensive
        ... print('Start time:', timer.start)
        ... print('Elapsed time:', timer.delta)
        ... print('Keep track of time inside with block:', timer())
    """

    def timer():
        return time.time() - t0 if timer.delta is None else timer.delta

    t0 = time.time()
    timer.start = t0
    timer.delta = None
    try:
        yield timer
    finally:
        timer.delta = time.time() - t0


@contextmanager
def log_timing(key):
    """
    A timer that saves results as a key in PERF_LOG.
    """
    try:
        with timeit() as timer:
            yield
    finally:
        PERF_LOG.append((key, timer.delta))


def timed(fn):
    """
    Tracks timing of function execution.
    """
    if PERF_TRACK:

        @wraps(fn)
        def decorated(*args, **kwargs):
            with timeit() as timer:
                res = fn(*args, **kwargs)
            PERF_LOG.append((decorated, timer.delta))
            return res

        return decorated
    else:
        return fn


def show_perf_log():
    import streamlit as st

    n_calls = Counter()
    tot_time = Counter()
    for k, v in PERF_LOG:
        k = getattr(k, "__name__", k)
        n_calls[k] += 1
        tot_time[k] += v
    df = pd.DataFrame({"n_calls": pd.Series(n_calls), "tot_time": pd.Series(tot_time)})
    st.write(df)
