from sidekick import api as sk, placeholder as _

from pydemic.params.params import Params

inverse = lambda name: sk.alias(name, transform=(1 / _), prepare=(1 / _))


class MainParameter:
    """
    Declares field as a main parameter
    """

    is_param = True
    is_alias = False
    is_mutable = False

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

    def __set_name__(self, owner, name):
        if self.name is None:
            self.name = name
