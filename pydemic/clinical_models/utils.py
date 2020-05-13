import numpy as np
import pandas as pd


def delayed(data, delay, K=0):
    """
    Transform data to be delayed by the given time delay.
    """
    ts = data.index

    if K and data.iloc[0]:
        ts_, data = extend_data(data, ts, delay, K)
    else:
        ts_ = ts + delay

    return pd.Series(np.interp(ts, ts_, data), index=ts)


def delayed_with_discharge(data, dt1, dt2, K=0, positive=False):
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
        entry = np.interp(ts, *extend_data(data, ts, dt1, K))

    if dt2 == 0:
        discharge = entry
    else:
        discharge = np.interp(ts, *extend_data(entry, ts, dt2, K))

    diff = entry - discharge
    return pd.Series(np.maximum(diff, 0) if positive else diff, index=ts)


def extend_data(data, ts, delay, K):
    if K == "infer":
        K = np.log(data.iloc[1] / data.iloc[0])

    ts_pre = ts - delay
    ts_pre = ts_pre[ts_pre < 0]
    n = len(ts_pre)
    ts_ = np.concatenate([ts_pre + n, ts + delay])

    data_pre = data.iloc[0] * np.exp(K * ts_[:n] - K * ts_[n])
    data = np.concatenate([data_pre, data])
    return ts_, data
