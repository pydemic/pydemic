import numpy as np
import pandas as pd

from .. import utils

elementwise = lambda fn: lambda x: x.apply(fn)
astype = lambda fn: lambda x: x.astype(fn)

DATA_TRANSFORMS = {}
MODEL_TRANSFORMS = {}
TIME_CONVERSIONS = {"days": 1, "weeks": 7, "months": 365.25 / 12, "years": 365.25}
RATIOS_PER_POP = {"pp": 1, "ppc": 100, "p1k": 1e3, "p10k": 1e4, "p100k": 1e5, "p1m": 1e6}


#
# Transforms and utility functions
#
def peak_date(model, col):
    """
    Date with the largest value of col.
    """
    return model.to_date(peak_time(model, col))


def peak_time(model, col):
    """
    Time with the largest value of col.
    """
    idx = np.argmax(model[col])
    return model.times[idx]


def to_dataframe(model, col):
    """
    Force columns to be data frames, even when results are vectors
    """
    name_, _, _ = col.partition(":")
    series = model[col]
    if isinstance(series, pd.DataFrame):
        return series
    return pd.DataFrame({name_: series.values}, index=series.index)


def to_dates(model, col):
    """
    Convert column index to use dates.
    """
    data = model[col]
    data.index = model.to_dates(data.index)
    return data


def ratio_transform(factor):
    """
    Factory function for ratios per population transformers.
    """
    return lambda model, col: factor * model[col] / model.population


def time_transform(factor):
    """
    Divide time index by the given factor.
    """

    def fn(model, col):
        data = model[col]
        data.index = data.index / factor
        return data

    return fn


#
# Update transform dictionaries
#
DATA_TRANSFORMS.update(
    {
        #
        # Simple queries
        "initial": lambda x: x.iloc[0],
        "final": lambda x: x.iloc[-1],
        "first": lambda x: x.iloc[0],
        "last": lambda x: x.iloc[-1],
        "max": lambda x: x.max(),
        "min": lambda x: x.min(),
        #
        # Type conversions
        "np": lambda x: np.asarray(x),
        "int": astype(int),
        "float": astype(float),
        "str": astype(str),
        #
        # Elementwise transforms
        "round": elementwise(lambda x: round(x)),
        "round1": elementwise(lambda x: round(x, 1)),
        "round2": elementwise(lambda x: round(x, 2)),
        "round3": elementwise(lambda x: round(x, 3)),
        "human": elementwise(utils.fmt),
        "pcfmt": elementwise(utils.pc),
        "p1kfmt": elementwise(utils.pm),
        "p10kfmt": elementwise(utils.p10k),
        "p100kfmt": elementwise(utils.p100k),
    }
)
MODEL_TRANSFORMS.update(
    {
        "peak-date": peak_date,
        "peak-time": peak_time,
        "df": to_dataframe,
        "dates": to_dates,
        #
        # Ratios per population
        **{k: ratio_transform(v) for k, v in RATIOS_PER_POP.items()},
        #
        # Time intervals
        **{k: time_transform(v) for k, v in TIME_CONVERSIONS.items()},
    }
)
