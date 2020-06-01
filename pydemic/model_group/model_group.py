import operator
from collections import Sequence
from types import MappingProxyType
from typing import Iterable, Union, Type

import pandas as pd

import mundi
from mundi import Region
from pydemic.utils import extract_keys
from .model_list import ModelList
from .properties import ModelGroupClinical, ModelGroupInfo, ModelGroupResults
from .utils import map_models, prepare_data
from ..models import Model


class ModelGroup(Iterable):
    """
    A group of (usually) closely related model instances.
    """

    models: ModelList
    info: ModelGroupInfo = property(ModelGroupInfo)
    results: ModelGroupResults = property(ModelGroupResults)
    clinical: ModelGroupClinical = property(ModelGroupClinical)
    kind: Type[Model] = Model

    @property
    def dates(self):
        return self._data[0].dates

    @property
    def times(self):
        return self._data[0].times

    @property
    def names(self):
        return pd.Index(m.name for m in self)

    @classmethod
    def from_children(
        cls,
        region: Union[Region, str],
        model_cls: Type[Model],
        options=MappingProxyType({}),
        **kwargs,
    ) -> "ModelGroup":
        """
        Create a group from children of the given Region.
        """
        region: Region = mundi.region(region)
        children = region.children(**extract_keys(("deep", "type", "subtype", "which"), kwargs))

        name = kwargs.pop("name", "{region.name}")
        group = []
        if options:
            options = {mundi.region(k): v for k, v in options.items()}

        for child in children:
            opts = options.get(child, {})
            if isinstance(name, str):
                opts["name"] = name.format(region=child, **kwargs, **opts)
            group.append(model_cls(region=child, **kwargs, **opts))

        return ModelGroup(group)

    def __init__(self, models):
        self._data = list(models)
        self.models = ModelList(self, self._data)

    def __len__(self) -> int:
        return len(self.models)

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, item):
        return item in self._data

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._data[item]
        elif isinstance(item, slice):
            return ModelGroup(self._data[item])
        return prepare_data({m.name: m[item] for m in self._data})

    def __getstate__(self):
        return self._data

    def __setstate__(self, data):
        self._data = data
        self.models = ModelList(self, self._data)

    #
    # Model group API
    #
    def apply(self, method, *args, **kwargs) -> list:
        """
        Execute method on each model by name and return a list of results.
        """

        out = []
        for i, m in enumerate(self):
            try:
                fn = getattr(m, method)
            except AttributeError:
                raise AttributeError(f"{i}-th model does not have a {method} method")
            else:
                out.append(fn(*args, **kwargs))
        return out

    def apply_table(self, method, *args, **kwargs) -> pd.Series:
        """
        Similar to :method:`apply`, but returns a Series object.
        """
        data = self.apply(method, *args, **kwargs)
        return pd.Series(data, index=self.names)

    def attrs(self, attr):
        """
        Return a list of values for the given attribute on each model.

        If attr is a list, return a list of lists.
        """
        if isinstance(attr, str):
            return list(self.map(getattr, attr))
        else:
            attrs = tuple(attr)
            out = []
            for m in self:
                out.append([getattr(m, attr) for attr in attrs])
            return out

    def table(self, attr) -> Union[pd.Series, pd.DataFrame]:
        """
        Like :method:`attrs`,  but return a Series object.

        If attr is a list, return a DataFrame.
        """
        if isinstance(attr, str):
            return pd.Series(self.attrs(attr), index=self.names)
        else:
            attrs = tuple(attr)
            return pd.DataFrame(self.attrs(attrs), index=self.names, columns=attrs)

    def map(self, func, *args, **kwargs) -> Iterable:
        """
        Map function to all models and return a iterator of results.
        """
        for m in self:
            yield func(m, *args, **kwargs)

    def set_attr(self, attr, value):
        """
        Set attribute in models.
        """
        if not isinstance(value, str) and isinstance(value, Sequence):
            for m, v in zip(self, value):
                setattr(m, attr, v)
        else:
            for m in self:
                setattr(m, attr, value)

    #
    # Model API
    #
    def run(self, period):
        """
        Execute the run method of each model for the given period.
        """
        for m in self:
            m.run(period)
