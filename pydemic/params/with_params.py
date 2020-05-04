"""
Implement interfaces for classes that expose lists of parameters.
"""
from abc import ABC
from numbers import Number
from typing import Union

import pandas as pd
import sidekick as sk

from .param import Params, param as _param, Param
from .with_params_info import ParamsInfo
from ..utils import not_implemented


class WithParams(ABC):
    """
    Basic interface for classes that have parameters
    """

    params_info: ParamsInfo
    params_data: pd.DataFrame
    params: Params = sk.lazy(not_implemented)
    _params: dict

    def __init__(self, params=None):
        self._params = {}
        self.set_params(self.params)
        if params:
            self.set_params(params)
        s1 = set(self._params)
        s2 = set(self._meta.params.primary)
        assert s1 == s2, f"Different param set: {s1} != {s2}"

    #
    # Parameters
    #
    def set_params(self, params=None, **kwargs):
        """
        Set a collection of params.
        """
        if params:
            for p in self._meta.params.primary:
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
        if name in self._meta.params.primary:
            self._params[name] = _param(value, pdf=pdf, ref=ref)
        elif name in self._meta.params.derived:
            setattr(self, name, _param(value).value)
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
                param = self.get_param(name)
                return _param(param)
        try:
            return self._params[name].value
        except KeyError:
            pass
        if name in self._meta.params__derived:
            return getattr(self, name)
        else:
            raise ValueError(f"invalid parameter name: {name!r}")


class WithDynamicParams(WithParams, ABC):
    """
    Like HasParams, but allow parameters to be set up as functions.
    """
