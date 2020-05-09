import numpy as np
import pandas as pd


def delayed(data, delay):
    """
    Transform data to be delayed by the given time delay.
    """
    ts = data.index
    ts_ = ts + delay
    return pd.Series(np.interp(ts, ts_, data), index=ts)


def delayed_with_discharge(data, dt1, dt2, positive=False):
    """
    Similar to :func:`delayed`, but includes a discharge time.

    Times dt1 and dt2 are additive, that is, elements wait a period dt1 to
    enter the delayed state, then wait dt2 to leave it. This function returns
    the difference between these two states.
    """
    ts = data.index

    if dt1 == 0:
        entry = data
    else:
        entry = np.interp(ts, ts + dt1, data)

    if dt2 == 0:
        discharge = entry
    else:
        discharge = np.interp(ts, ts + dt2, entry)

    diff = entry - discharge
    return pd.Series(np.maximum(diff, 0) if positive else diff, index=ts)
