import operator
from typing import TYPE_CHECKING, Iterable

import pandas as pd

if TYPE_CHECKING:
    from ..models import Model  # noqa: F401


def map_models(fn: callable, models: Iterable["Model"], key=operator.attrgetter("name")):
    """
    Map function to list of models and return a Series or DataFrame that merges
    the content of fn(model) the given column for each model.

    Args:
        fn:
            Function used to extract data from models.
        models:
            A sequence of models.
        key:
            A callable that receives model and return the corresponding column
            name. If not given, uses `model.name` as key.
    """

    if not isinstance(models, (tuple, list, set)):
        models = tuple(models)
    data = [fn(m) for m in models]
    names = [key(m) for m in models]
    types = set(map(type, data))

    # A sequence of columns should become a dataframe
    if types == {pd.Series}:
        return pd.concat(data, axis=1).set_axis(names, axis=1)

    elif types == {pd.DataFrame}:
        cols = ((name, col) for df, name in zip(data, names) for col in df.columns)
        cols = pd.MultiIndex.from_tuples(cols)
        return pd.concat(data, axis=1).set_axis(cols, axis=1)

    # Otherwise, we assume scalar values and create a simple series object
    return pd.Series(data, index=names)


def prepare_data(models: dict):
    """
    Last step in processing the results of a ModelGroup() getitem.

    Args:
        models:
            A mapping from keys to data.
    """

    data = list(models.values())
    types = set(map(type, data))
    names = list(models.keys())

    # A sequence of columns should become a dataframe
    if types == {pd.Series}:
        return pd.concat(data, axis=1).set_axis(names, axis=1)

    elif types == {pd.DataFrame}:
        cols = ((name, col) for df, name in zip(data, names) for col in df.columns)
        cols = pd.MultiIndex.from_tuples(cols)
        return pd.concat(data, axis=1).set_axis(cols, axis=1)

    # Otherwise, we assume scalar values and create a simple series object
    return pd.Series(data, index=names)


def map_method(*args, **kwargs):
    """
    Given a method name and a sequence of models, return a sequence of values
    of applying the extra positional and keyword arguments in each method.

    Attr:
        attr:
            Method name. Dotted python names are valid.
        models:
            A sequence of models.
        *args, **kwargs:
            Arguments to pass to m.<attr>(*args, **kwargs)
    """
    attr, data, *args = args
    attr, _, method = attr.partition(".")
    if method:
        data = map(operator.attrgetter(attr), data)
        return map(operator.methodcaller(method, *args, **kwargs), data)
    else:
        return map(operator.methodcaller(attr, *args, **kwargs), data)


def model_group_method(attr, out=None):
    """
    A method that applies
    """
    out = out or (lambda x: x)

    def method(self, *args, **kwargs):
        return out(map_method(attr, self.group, *args, **kwargs))

    method.__name__ = attr.rpartition(".")[-1]
    return method


def group(x):
    """
    Convert object to ModelGroup.
    """
    from .model_group import ModelGroup

    return x if isinstance(x, ModelGroup) else ModelGroup(x)
