from functools import wraps
from operator import methodcaller
from typing import List

import mundi
import pandas as pd

import sidekick.api as sk
from .. import diseases


def cached_method(func):
    """
    Cache results of simple methods that takes no arguments.
    """
    name = f"__{func.__name__}"

    @wraps(func)
    def cached(self, *args, **kwargs):
        if args or kwargs:
            return func(self, *args, **kwargs)

        try:
            return getattr(self, name)
        except AttributeError:
            result = func(self)
            setattr(self, name, result)
            return result

    return cached


def delegate_map(attr: str, constructor=None):
    """
    A delegate method that returns the result of mapping a transformation to
    each element of a sequence
    """

    def method(self, *args, **kwargs):
        fn = methodcaller(attr, *args, **kwargs)
        if constructor is None:
            make = type(self)
        else:
            make = constructor
        return make(fn(obj) for obj in self)

    return method


def delegate_mapping(attr: str, constructor=dict, items=methodcaller("items")):
    """
    A delegate method that returns the result of mapping a transformation to
    each element of a sequence
    """

    def method(self, *args, **kwargs):
        fn = methodcaller(attr, *args, **kwargs)
        return constructor((k, fn(obj)) for k, obj in items(self))

    return method


def concat_frame(items) -> pd.DataFrame:
    """
    Concatenate series into data frames.
    """

    keys: List[str] = []
    values: List[pd.Series] = []
    for k, v in items:
        keys.append(k)
        values.append(v)
    return pd.concat(values, keys=keys, axis=1)


@sk.curry(2)
def agg_frame(method, items):
    """
    Aggregate frame using given method.
    """
    if callable(method):
        fn = method
    else:
        fn = lambda df: getattr(df, method)(axis=1)
    return fn(pd.DataFrame(dict(items)))


def from_region(cls, transform, region, params=None, disease=None, **kwargs):
    """
    Initialize data extracting curve from region.
    """
    region = mundi.region(region)
    disease = diseases.disease(disease)
    data = disease.epidemic_curve(region)
    kwargs.setdefault("population", region.population)
    return cls(transform(data), params, **kwargs)
