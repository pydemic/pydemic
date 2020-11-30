import operator
from collections import MutableSequence
from collections import Sequence
from types import MappingProxyType
from typing import Iterable, Union, Type
from weakref import ref

import mundi
import pandas as pd
from mundi import Region

from .utils import map_models, map_method, prepare_data, model_group_method, model_group
from ..models import Model
from ..utils import extract_keys


class ModelGroup(Iterable):
    """
    A group of (usually) closely related model instances.
    """

    models: "ModelList"
    info: "ModelGroupInfo" = property(lambda self: ModelGroupInfo(self))
    results: "ModelGroupResults" = property(lambda self: ModelGroupResults(self))
    clinical: "ModelGroupClinical" = property(lambda self: ModelGroupClinical(self))
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


#
# Auxiliary classes
#
class ModelList(MutableSequence):
    """
    Implements the "models" attribute of a model group.

    It offers a list-like interface to the models included in a ModelGroup.
    """

    __slots__ = ("_data", "_group")

    group: "ModelGroup" = property(lambda self: self._group())
    _data: list

    def __init__(self, group: "ModelGroup", data: list):
        self._group = ref(group)
        self._data = data

    def __getitem__(self, idx):
        if isinstance(idx, str):
            for m in self._data:
                if m.name == idx:
                    return m
            else:
                raise KeyError(f"no model named {idx!r}")

        data = self._data[idx]
        if isinstance(data, list):
            cls = type(self._group)
            return cls(data)
        return data

    def __setitem__(self, idx, obj) -> None:
        old = self[idx]
        model_cls = self.group.kind
        if isinstance(old, type(self._group)):
            data = getattr(obj, "models", obj)
            if not all(isinstance(m, model_cls) for m in data):
                raise self._insert_type_error()
            self._data[idx] = data
        elif isinstance(obj, model_cls):
            self._data[idx] = obj
        else:
            raise self._insert_type_error()

    def __delitem__(self, i: int) -> None:
        del self._data[i]

    def __len__(self) -> int:
        return len(self._data)

    def insert(self, index: int, obj) -> None:
        if not isinstance(obj, self.group.kind):
            raise self._insert_type_error()
        self._data[index] = obj

    def _insert_type_error(self):
        return TypeError("can only insert model instances to model group")


#
# Properties for Model groups
#
class ModelGroupProp:
    """
    Base class for all model group properties.
    """

    group: "ModelGroup"
    prop_name: str = None
    MODEL_METHODS = frozenset()
    PROP_METHODS = frozenset()

    def __init__(self, group):
        self.group = group

    def __getitem__(self, item):
        models = self.iter_items(item)
        return map_models(operator.itemgetter(item), models)

    def __getattr__(self, item):
        if item in self.MODEL_METHODS:

            def method(*args, **kwargs):
                return ModelGroup(fn(*args, **kwargs) for fn in self.iter_attr(item))

            method.__name__ = item
            return method

        if item in self.PROP_METHODS:

            def method(*args, **kwargs):
                return [fn(*args, **kwargs) for fn in self.iter_attr(item)]

            method.__name__ = item
            return method

        return list(self.iter_attr(item))

    def __dir__(self):
        try:
            prop = next(self.iter_pros())
            prop_attrs = dir(prop)
        except IndexError:
            prop_attrs = ()
        return list({*super().__dir__(), *prop_attrs})

    def iter_props(self):
        """
        Iterate over each accessor.
        """
        prop_name = self.prop_name
        for m in self.group:
            yield getattr(m, prop_name)

    def iter_items(self, item):
        """
        Iterate over each accessor, selecting the given item.
        """
        for prop in self.iter_props():
            yield prop[item]

    def iter_attr(self, attr):
        """
        Iterate over each accessor, selecting the given attribute.
        """
        for prop in self.iter_props():
            yield getattr(prop, attr)

    def iter_method(self, *args, **kwargs):
        """
        Iterate over each accessor, executing the given method forwarding
        the passed arguments.
        """
        method, *args = args
        for fn in self.iter_attr(method):
            yield fn(*args, **kwargs)


class ModelGroupInfo(ModelGroupProp):
    """
    Implements the "info" attribute of model groups.
    """

    prop_name = "info"


class ModelGroupResults(ModelGroupProp):
    """
    Implements the "results" attribute of model groups.
    """

    prop_name = "results"


class ModelGroupClinical(ModelGroupProp):
    """
    Implements the "clinical" attribute of model groups.
    """

    prop_name = "clinical"
    MODEL_METHODS = {"clinical_model", "crude_model", "delay_model", "overflow_model"}

    def __call__(self, *args, **kwargs):
        return model_group(map_method("clinical", self.group, *args, **kwargs))

    crude_model = model_group_method("clinical.crude_model", out=model_group)
    delay_model = model_group_method("clinical.delay_model", out=model_group)
    overflow_model = model_group_method("clinical.overflow_model", out=model_group)
