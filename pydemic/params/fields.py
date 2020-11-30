from sidekick import api as sk, placeholder as _
from sidekick.properties import _TransformingAlias, _MutableAlias

from ..params import Params

inverse = lambda name: sk.alias(name, transform=(1 / _), prepare=(1 / _))


class BaseField:
    """
    Base class for field classes.
    """

    is_param = True
    is_alias = False
    is_mutable = False
    is_field = True
    is_data_field = False

    def __get__(self, instance, owner):
        raise NotImplementedError

    def __set__(self, instance, value):
        raise NotImplementedError

    def __set_name__(self, owner, name):
        if self.name is None:
            self.name = name

    def transform_function(self, data):
        """
        Return a function that reads a mapping with primary field data and
        return the value of the field.
        """
        raise NotImplementedError


class Field(BaseField):
    """
    Field that refers to a raw data parameter.
    """

    is_data_field = True

    def __init__(self, name=None, default=None, description=None):
        self.name = name
        self.default = default
        self.description = description

    # noinspection PyProtectedMember
    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        if isinstance(obj, Params):
            return obj._data.get(self.name, self.default)
        return self.default

    # noinspection PyProtectedMember
    def __set__(self, obj, value):
        if not isinstance(obj, Params):
            raise TypeError("params are immutable")
        if obj._mutable:
            obj._data[self.name] = value

    def transform_function(self, data):
        return lambda data: data[self.name]


class TransformField(_TransformingAlias, BaseField):
    """
    Field that transforms another field.
    """

    def __init__(self, attr, fn, inv=None):
        super().__init__(attr, fn, inv)

    def transform_function(self, data):
        fn = self.transform
        return lambda data: fn(data[self.attr])


class InverseField(TransformField):
    """
    Field that performs an inverse transform in another field.
    """

    def __init__(self, attr, scale=1.0):
        super().__init__(attr, lambda x: scale / x, lambda x: scale / x)


class AliasField(_MutableAlias, BaseField):
    """
    A simple alias to another field
    """

    def __init__(self, attr):
        super().__init__(attr)

    def transform_function(self, data):
        return lambda data: data[self.attr]


class DerivedField(BaseField, property):
    """
    A derived field
    """

    @classmethod
    def from_function(cls, *args):
        def decorator(fn):
            return cls(fn, args, name=fn.__name__)

        return decorator

    def __init__(self, function, args, name=None):
        self.name = name
        self.args = args = tuple(args)
        self.function = fn = sk.to_callable(function)

        def fget(self):
            return fn(*(getattr(self, k) for k in args))

        super().__init__(fget)

    def __set_name__(self, owner, name):
        if self.name is None:
            self.name = name

    def transform_function(self, data):
        fn = self.function
        args = self.args
        return lambda data: fn(*(data[k] for k in args))
