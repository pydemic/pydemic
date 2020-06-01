"""
Implement interfaces for classes that expose lists of parameters.
"""
from abc import ABC
from numbers import Number
from typing import Union

import pandas as pd
import sidekick as sk
from sidekick import placeholder as _

from .params_info import ParamsInfo
from ..logging import log
from ..params import Param, param, get_param
from ..types import ImproperlyConfigured
from ..utils import extract_keys

param_ = param


class WithParamsMixin(ABC):
    """
    Basic interface for classes that have parameters
    """

    name: str
    params: ParamsInfo
    params_data: pd.DataFrame
    disease_params: object
    _params: dict

    # Cache information in the params_info object as instance attributes.
    __all_params: frozenset = sk.lazy(_.meta.params.all)
    __primary_params: frozenset = sk.lazy(_.meta.params.primary)
    __alternative_params: frozenset = sk.lazy(_.meta.params.alternative)

    def __init__(self, params=None, keywords=None):
        self._params = {}
        if params is not None:
            self.set_params(params)

        extra = self.__all_params.intersection(keywords)
        if extra:
            self.set_params(extract_keys(extra, keywords))

        for key in self.__primary_params - self._params.keys():
            self.set_param(key, init_param(key, self, self.disease_params))

        s1 = frozenset(self._params)
        s2 = self.__primary_params
        assert s1 == s2, f"Different param set: {s1} != {s2}"

    #
    # Parameters
    #
    def set_params(self, params=None, **kwargs):
        """
        Set a collection of params.
        """

        for k, v in kwargs:
            self.set_param(k, v)

        if params:
            NOT_GIVEN = object()
            for p in self.__primary_params:
                if p not in kwargs:
                    value = get_param(p, params, default=NOT_GIVEN)
                    if value is not NOT_GIVEN:
                        self._params[p] = param_(value)

    def set_param(self, name, value, *, pdf=None, ref=None):
        """
        Sets a parameter in the model, possibly assigning a distribution and
        reference.
        """
        if name in self.__primary_params:
            cls = type(self).__name__
            log.debug(f"{cls}.{name} = {value!r} ({self.name})")

            self._params[name] = param(value, pdf=pdf, ref=ref)
        elif name in self.__all_params:
            setattr(self, name, param(value).value)
        else:
            raise ValueError(f"{name} is an invalid param name")

    def get_param(self, name, param=False) -> Union[Number, Param]:
        """
        Return the parameter with given name.

        Args:
            name:
                Parameter name.
            param:
                If True, return a :cls:`Param` instance instead of a value.
        """
        if param:
            try:
                return self._params[name]
            except KeyError:
                return param_(self.get_param(name))
        try:
            return self._params[name].value
        except KeyError:
            pass
        if name in self.__all_params:
            return getattr(self, name)
        else:
            raise ValueError(f"invalid parameter name: {name!r}")


class WithDynamicParamsMixin(WithParamsMixin, ABC):
    """
    Like HasParams, but allow parameters to be set up as functions.
    """


def init_param(key, obj, disease):
    if disease is not None:
        try:
            return get_param(key, disease)
        except ValueError:
            pass
    try:
        return getattr(obj, key)
    except AttributeError:
        pass

    try:
        value = getattr(obj, "_" + key)
    except AttributeError:
        raise ImproperlyConfigured(
            f"""bad parameter: {key}

Classes must either provide explicit default values for parameters or
implement a _{key}() method used to initialize the parameter during instance
creation.

The initialization method may explicitly raise exceptions, if it is not
possible to provide a sane default value for the instance.
"""
        )
    else:
        if callable(value):
            return value()
        return value
