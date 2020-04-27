from numbers import Number
from types import SimpleNamespace
from typing import NamedTuple, Optional, Callable, Union

import sidekick as sk


class Param(NamedTuple):
    """
    Represents a parameter
    """
    value: Number
    ref: Optional[str] = None
    pdf: Optional[Union[Callable, str]] = None

    def __str__(self):
        suffix = []
        if self.ref:
            suffix.append(str(self.ref))
        if self.pdf and not isinstance(self.pdf, SimpleNamespace):
            suffix.append(str(self.pdf))
        suffix = ', '.join(suffix)
        return f'{self.value} ({suffix})' if suffix else str(self.value)


class ParamMeta(type):
    @sk.lazy
    def getters(cls):
        getters = {}
        for k, v in vars(cls).items():
            if hasattr(v, 'fget'):
                getters[k] = v.fget
        return getters


class Params(metaclass=ParamMeta):
    """
    Represents a set of parameters.
    """
    __slots__ = ('name', 'refs', 'pdfs', '__dict__')

    def __init__(self, name=None, **kwargs):
        type(self).name.__set__(self, name or type(self).__name__)
        type(self).refs.__set__(self, {})
        type(self).pdfs.__set__(self, {})

        for k, v in kwargs.items():
            ref = rvs = None
            if isinstance(v, tuple):
                v, ref, rvs = v
            super().__setattr__(k, v)
            if ref is not None:
                self.refs[k] = ref
            if rvs is None:
                rvs = SimpleNamespace(rvs=cte(v))
            self.pdfs[k] = rvs

    def __setattr__(self, k, v):
        raise TypeError(
            f'Parameters are immutable. Use param.copy({k}={v!r}) to create a copy with '
            f'different values')

    def __getitem__(self, item):
        try:
            return self.__dict__[item]
        except KeyError:
            pass
        prop = type(self).getters[item]
        return prop(self)

    def __iter__(self):
        refs = self.refs.get
        rvs = self.pdfs.get
        for k, v in self.__dict__.items():
            yield k, Param(v, refs(k), rvs(k))

    def __str__(self):
        return self.summary()

    def __repr__(self):
        args = (f'{k}={v.value!r}' for k, v in self)
        args = ', '.join(args)
        cls = type(self).__name__
        return f'{cls}({self.name!r}, {args})'

    def copy(self, **kwargs):
        """
        Copy, possibly overwriting some values.
        """
        cls = type(self)
        for k, v in kwargs.items():
            if hasattr(self, k):
                kwargs[k] = v
            else:
                raise AttributeError(f'invalid attribute: {k}')

        kwargs = {**self.__dict__, **kwargs}
        return cls(self.name, **kwargs)

    def value(self, key, default=None) -> Number:
        """
        Return the value of the given parameter.
        """
        try:
            value = getattr(self, key)
            return value
        except KeyError:
            if default is None:
                raise
            return default

    def ref(self, key) -> Optional[str]:
        """
        Return the reference for the given parameter, if it exists.
        """
        return self.refs.get(key)

    def pdf(self, key):
        """
        Return the probability distribution function for the given param.
        """
        return self.pdfs.get(key)

    def param(self, key) -> Param:
        """
        Return a :cls:`Param` object describing the given parameter.
        """
        value = self.value(key)
        ref = self.ref(key)
        pdf = self.pdf(key)
        return Param(value, ref, pdf)

    def summary(self):
        """
        Print a summary of all parameters in the set.
        """
        lines = [f'Parameters ({self.name}):', *(f'  - {k}: {p}' for k, p in self)]
        return '\n'.join(lines)


# Return a function always return the same value
cte = lambda v: lambda: v


def param(value, ref=None, pdf=None) -> Param:
    """
    Declares a parameter with optional reference attribution and RVS
    attribution.
    """
    if isinstance(value, Param):
        ref = ref or value.ref
        pdf = pdf or value.pdf
        value = value.value

    return Param(value, ref, pdf)
