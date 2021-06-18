from functools import lru_cache
from numbers import Number
from typing import Union, NamedTuple

import mundi
import numpy as np
import pandas as pd

import sidekick.api as sk
from .. import db
from .. import fitting as fit
from ..cache import ttl_cache
from ..params import get_param
from ..utils import trim_zeros, force_monotonic, slugify

QualDataT = Union[pd.DataFrame, "Dataset"]
QualValueT = Union[pd.Series, Number, "Dataset"]
DB_PATH = db.DATABASES / "diseases"


def dunder(name, reversed=False):
    rname = ("__" + name[3:]) if reversed else ("__r" + name[2:])

    def op(self: "Dataset", other):
        if isinstance(other, Dataset):
            other = other.data

        fn = getattr(self.data, name, None)
        if fn is not None:
            value = fn(other)
            if value is not NotImplemented:
                return Dataset(value)

        fn = getattr(other, rname, None)
        if fn is not None:
            value = fn(self.data)
            if value is not NotImplemented:
                return Dataset(value)

        return NotImplemented

    op.__name__ = op.__qualname__ = name
    return op


class Dataset(NamedTuple):
    """
    A qualified dataset. Includes raw data in the .data attribute and meta
    information
    """

    data: Union[pd.DataFrame, pd.Series, np.ndarray, Number]
    source: str = ""
    notes: str = ""

    __add__ = dunder("__add__")
    __sub__ = dunder("__sub__")
    __mul__ = dunder("__mul__")
    __truediv__ = dunder("__truediv__")
    __floordiv__ = dunder("__floordiv__")
    __pow__ = dunder("__pow__")
    __mod__ = dunder("__mod__")

    # Reversed methods
    __radd__ = dunder("__radd__", reversed=True)
    __rsub__ = dunder("__rsub__", reversed=True)
    __rmul__ = dunder("__rmul__", reversed=True)
    __rtruediv__ = dunder("__rtruediv__", reversed=True)
    __rfloordiv__ = dunder("__rfloordiv__", reversed=True)
    __rpow__ = dunder("__rpow__", reversed=True)
    __rmod__ = dunder("__rmod__", reversed=True)


def normalize_source(source):
    """
    Slugify and remove unnecessary suffixes from source reference.

    This is useful to transform a textual description of the source work to
    a valid filename in which it is installed.

    Examples:
        >>> normalize_source("Verity, et. al.")
        "verity"
    """
    return slugify(source, suffixes=("et-al",))


def lazy_stored_string(path):
    """
    A lazy string-like object that loads information from a file.
    """
    return sk.deferred(lazy_description, path)


def age_adjusted_average(ages, value):
    """
    Given an age distribution and a table with age-adjusted values, compute the
    mean value for population.

    This method automatically aligns ages and value if both tables are not
    aligned.
    """
    if isinstance(ages, str):
        ages = mundi.region(ages).age_distribution
    population = ages.values.sum()
    if set(ages.index).issuperset(value.index):
        value = value.reindex(ages.index, method="ffill")
    else:
        raise NotImplementedError
    return (value * ages / population).sum()


@lru_cache(1)
def world_age_distribution():
    """
    World age distribution computed by summing
    """
    countries = mundi.countries_dataframe()
    return countries.mundi["age_distribution"].sum(0)


@lru_cache(32)
def lazy_description(path):
    """
    Lazily load string from path.
    """
    with open(DB_PATH / path) as fd:
        return fd.read()


@lru_cache(32)
def read_table(path: str, key=None):
    """
    Read dataframe from path.
    """
    return db.read_table(path, key)


def set_age_distribution_default(dic, value=None, drop=False):
    """
    Set the ages_distribution key of dictionary to the given value or to the
    World's age distribution.
    """

    ages = dic.pop("age_distribution", None)

    if ages is None:
        ages = world_age_distribution() if value is None else value
        if isinstance(ages, str):
            ages = mundi.region(value).age_distribution
        elif not isinstance(ages, (pd.Series, pd.DataFrame)):
            ages = get_param("age_distribution", value)

    if not drop:
        dic["age_distribution"] = ages
    return ages


@ttl_cache("pydemic.disease", timeout=4 * 3600)
def epidemic_curve(
    cls,
    region,
    params=None,
    diff=False,
    smooth=False,
    real=False,
    keep_observed=False,
    window=14,
    trim="left",
    clean=True,
    **kwargs,
) -> pd.DataFrame:
    """
    Implements the epidemic_curve method of Disease classes.
    """
    data = cls._epidemic_curve(mundi.region(region), **kwargs)
    data.index.name = "date"
    data = trim_zeros(data, trim)

    if clean:
        data = force_monotonic(data)

    if real:
        from ..empirical.functions import estimate_real_cases

        method = "CFR" if real is True else real
        real = estimate_real_cases(data, params, method)

        if keep_observed:
            rename = {"cases": "cases_observed", "deaths": "deaths_observed"}
            data = pd.concat([real, data.rename(rename, axis=1)], axis=1)
        else:
            data = real

    if diff:
        values = np.diff(data, prepend=0, axis=0)
        data = pd.DataFrame(values, index=data.index, columns=data.columns)

    if smooth:
        columns = pd.MultiIndex.from_product([["observed", "smooth"], data.columns])
        data = pd.concat([data, fit.smooth(data, window)], axis=1)
        data.columns = columns

    return data
