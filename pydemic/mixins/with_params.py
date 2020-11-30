"""
Implement interfaces for classes that expose lists of parameters.
"""
from abc import ABC
from numbers import Number
from typing import Union, TYPE_CHECKING, Mapping

import pandas as pd

from ..logging import log
from ..types import Param, param

param_ = param
if TYPE_CHECKING:
    from ..models import Model


class WithParamsMixin(ABC):
    """
    Basic interface for classes that have parameters
    """

    name: str
    params: Mapping
    params_data: pd.DataFrame
    disease_params: object

    def __init__(self: "Model", params=None, keywords=None):
        self._params = self.meta.params.copy()
        params = params or {}

        for key, value in self.meta.params.items():
            if key in params:
                value = params[key]
            elif key in keywords:
                value = keywords.pop(key)
            elif key in self.disease_params:
                value = self.disease_params[key]
            else:
                continue
            self._params[key] = param(value)

    #
    # Parameters
    #
    def set_params(self, params=None, **kwargs):
        """
        Set a collection of params.
        """
        for k, v in kwargs.items():
            self.set_param(k, v)
        for k, v in params.items():
            if k not in kwargs:
                self.set_param(k, v)
        return self

    def set_param(self, name, value, *, pdf=None, ref=None):
        """
        Sets a parameter in the model, possibly assigning a distribution and
        reference.
        """
        if name not in self._params:
            raise ValueError(f"invalid parameter: {name}")
        cls = type(self).__name__
        log.debug(f"{cls}.{name} = {value!r} ({self.name})")

        self._params[name] = param(value, pdf=pdf, ref=ref)
        return self

    def get_param(self, name, full=False) -> Union[Number, Param]:
        """
        Return the parameter with given name.

        Args:
            name:
                Parameter name.
            full:
                If True, return a :class:`Param` instance instead of a value.
        """
        if full:
            try:
                return self._params[name]
            except KeyError:
                return param_(self.get_param(name))
        try:
            value = self._params[name]
            return getattr(value, "value", value)
        except KeyError:
            raise ValueError(f"invalid parameter name: {name!r}")
