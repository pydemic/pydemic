from numbers import Number
from types import SimpleNamespace
from typing import Any, Iterable, Tuple, Optional, Union, Callable

import numpy as np
import pandas as pd
from sidekick import api as sk

from .typing import Numeric
from ..math import numeric_derivative

ParamLike = Any


class ResultMixin:
    value: Any
    is_finite = property(lambda self: np.isfinite(self.value))

    __add__ = sk.delegate_to("value")
    __sub__ = sk.delegate_to("value")
    __mul__ = sk.delegate_to("value")
    __matmul__ = sk.delegate_to("value")
    __truediv__ = sk.delegate_to("value")
    __floordiv__ = sk.delegate_to("value")
    __mod__ = sk.delegate_to("value")
    __pow__ = sk.delegate_to("value")
    __and__ = sk.delegate_to("value")
    __lshift__ = sk.delegate_to("value")
    __rshift__ = sk.delegate_to("value")
    __or__ = sk.delegate_to("value")
    __xor__ = sk.delegate_to("value")

    __radd__ = sk.delegate_to("value")
    __rsub__ = sk.delegate_to("value")
    __rmul__ = sk.delegate_to("value")
    __rmatmul__ = sk.delegate_to("value")
    __rtruediv__ = sk.delegate_to("value")
    __rfloordiv__ = sk.delegate_to("value")
    __rmod__ = sk.delegate_to("value")
    __rpow__ = sk.delegate_to("value")
    __rand__ = sk.delegate_to("value")
    __rlshift__ = sk.delegate_to("value")
    __rrshift__ = sk.delegate_to("value")
    __ror__ = sk.delegate_to("value")
    __rxor__ = sk.delegate_to("value")

    __eq__ = sk.delegate_to("value")
    __ge__ = sk.delegate_to("value")
    __gt__ = sk.delegate_to("value")
    __le__ = sk.delegate_to("value")
    __lt__ = sk.delegate_to("value")
    __ne__ = sk.delegate_to("value")

    __abs__ = sk.delegate_to("value")
    __int__ = sk.delegate_to("value")
    __float__ = sk.delegate_to("value")
    __neg__ = sk.delegate_to("value")
    __pos__ = sk.delegate_to("value")
    __invert__ = sk.delegate_to("value")
    __index__ = sk.delegate_to("value")


class Result(ResultMixin, sk.Record):
    """
    A numeric result with additional information.
    """

    value: Any
    info: Any = None


class UserFloat(float):
    """
    Base class for all user-defined floats
    """

    __slots__ = ("ref", "pdf", "unit")

    # Expose the floating point value as a property. It makes it consistent
    # with other personalized values.
    value: float = property(float)

    #: Reference in the literature from where the parameter was extracted
    ref: Optional[str]

    #: Probability density function that generates random values for the parameter.
    pdf: Optional[Union[Callable, str]]

    #: Unit used for the parameter
    unit: Optional[str]

    def __new__(cls, data, *args, ref=None, pdf=None, unit=None, **kwargs):
        new = object.__new__(cls, data)
        new.ref = ref
        new.pdf = pdf
        new.unit = unit
        return new

    def __init__(self, data, *args, ref=None, pdf=None, unit=None, **kwargs):
        super().__init__(data)


class ValueCI(UserFloat):
    """
    Value with a confidence interval.
    """

    __slots__ = ("low", "high")
    value: float = property(float)
    low: Numeric
    high: Numeric

    def __init__(self, data, low=None, high=None, **kwargs):
        super().__init__(data, **kwargs)
        self.low = float(data if low is None else low)
        self.high = float(data if high is None else high)


class ValueStd(UserFloat):
    """
    A value that represents a number with its standard deviation.
    """

    __slots__ = ("std",)
    value: float = property(float)
    std: float

    @classmethod
    def mean(cls, iterable: Iterable[Tuple[float, float]], tol=1e-9) -> "ValueStd":
        """
        Merge several independent point estimates of (value, std) into a single
        estimate, weighting results by the inverse variance.

        Args:
            iterable:
                A sequence of (value, std) tuples.
            tol:
                A normalization term to avoid problem with null variances.
        """
        if isinstance(iterable, pd.DataFrame):
            iterable = iterable.values

        weights = 0.0
        cum_var = 0.0
        N = 0

        for (value, std) in iterable:
            var = std * std + tol
            weight = 1 / var
            weights += weight
            cum_var += weight * value
            N += 1

        return ValueStd(cum_var / weights, np.sqrt(cum_var / N))

    def __init__(self, data, std=0.0, **kwargs):
        super().__init__(data, **kwargs)
        self.std = float(std)

    def apply(self, func, derivative=None):
        """
        Transform value by function.
        """
        derivative = derivative or numeric_derivative(func)
        return ValueStd(func(self.value), abs(derivative(self.value)) * self.std)


class Param(ResultMixin, sk.Record):
    """
    Represents a parameter.
    """

    #: Value for the parameter or callable that generates the value
    data: Number

    #: Reference in the literature from where the parameter was extracted
    ref: Optional[str] = None

    #: Probability density function that generates random values for the parameter.
    pdf: Optional[Union[Callable, str]] = None

    @property
    def value(self):
        v = self.data
        return v() if callable(v) else v

    def __str__(self):
        suffix = []
        if self.ref:
            suffix.append(str(self.ref))
        if self.pdf and not isinstance(self.pdf, SimpleNamespace):
            suffix.append(str(self.pdf))
        suffix = ", ".join(suffix)
        return f"{self.value} ({suffix})" if suffix else str(self.value)
