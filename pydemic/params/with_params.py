"""
Implement interfaces for classes that expose lists of parameters.
"""
from abc import ABC
from numbers import Number
from typing import Union

import pandas as pd
import sidekick as sk
from sidekick import placeholder as _

from .param import Params, param, Param
from .with_params_info import ParamsInfo
from ..utils import not_implemented

param_ = param


class WithParams(ABC):
    """
    Basic interface for classes that have parameters
    """

    params_info: ParamsInfo
    params_data: pd.DataFrame
    params: Params = sk.lazy(not_implemented)
    _params: dict

    # Cache information in the params_info object as instance attributes.
    __primary_params = sk.lazy(_.params_info.primary)
    __all_params = sk.lazy(_.params_info.all)
    __alternative_params = sk.lazy(_.params_info.alternative)

    def __init__(self, params=None):
        self._params = {}
        self.set_params(self.params)
        if params:
            self.set_params(params)
        s1 = set(self._params)
        s2 = set(self.params_info.primary)
        assert s1 == s2, f"Different param set: {s1} != {s2}"

    #
    # Parameters
    #
    def set_params(self, params=None, **kwargs):
        """
        Set a collection of params.
        """
        if params:
            for p in self.__primary_params:
                self._params[p] = kwargs.pop(p) if p in kwargs else params.param(p)

        for k, v in kwargs:
            self.set_param(k, v)

        name = type(self).__name__
        self.params = Params(name, **self._params)

    def set_param(self, name, value, *, pdf=None, ref=None):
        """
        Sets a parameter in the model, possibly assigning a distribution and
        reference.
        """
        if name in self.__primary_params:
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


class WithDynamicParams(WithParams, ABC):
    """
    Like HasParams, but allow parameters to be set up as functions.
    """
