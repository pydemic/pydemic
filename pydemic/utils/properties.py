from functools import total_ordering

from ..packages import sk

NOT_GIVEN = object()


@total_ordering
class state_property(property):
    """
    A property that exposes a state component as rw.
    """

    name: str
    index: int
    ro: bool

    def __init__(self, i, ro=False, name=None):
        def fget(self):
            return self.state[i]

        def fset(self, value):
            self.state[i] = value

        args = (fget,) if ro else (fget, fset)
        self.index = i
        self.name = name
        self.ro = ro
        super().__init__(*args)

    def __set_name__(self, owner, name):
        if self.name is None:
            self.name = name

    def __gt__(self, other):
        if isinstance(other, state_property):
            return self.index > other.index
        return NotImplemented

    def __repr__(self):
        args = ", True" if self.ro else ""
        args += f", name={self.name!r}" if self.name else ""
        return f"state_property({self.index}{args})"

    def __str__(self):
        return self.name or super().__str__()


class param_property(property):
    """
    A property that simply mirrors a given param value.
    """

    is_param = True
    is_derived = False

    def __init__(self, name=None, ro=False, default=NOT_GIVEN):
        if callable(default):
            default = sk.extract_function(default)

        self.name = name
        self.default = default
        param = self

        def fget(self):
            try:
                return self._params[param.name].value
            except KeyError:
                if param.default is NOT_GIVEN:
                    name = type(self).__name__
                    raise AttributeError(f"{name} has no {param.name!r} attribute")
                value = param.default
                return value(self) if callable(value) else value

        def fset(self, value):
            self.set_param(param.name, value)

        args = (fget,) if ro else (fget, fset)
        super().__init__(*args)

    def __set_name__(self, owner, name):
        self.name = self.name or name


class param_transform(property):
    """
    A property that represents a transform into a parameter.
    """

    is_param = True
    is_derived = True

    def __init__(self, prop_name, read, write=None):
        read = sk.extract_function(read)

        if write is not None:
            write = sk.extract_function(write)

        def fget(self):
            try:
                value = self._params[prop_name].value
            except KeyError:
                value = self.get_param(prop_name)
            try:
                return read(value)
            except Exception as e:
                cls = type(e).__name__
                msg = f"error found when processing {prop_name!r}: {cls}{e}"
                raise ValueError(msg)

        def fset(self, value):
            value = write(value)
            self.set_param(prop_name, value)

        args = (fget, fset) if write else (fget,)
        super().__init__(*args)


def param_alias(prop, ro=False):
    """
    A simple alias to a parameter.
    """

    id_fn = lambda x: x
    return param_transform(prop, read=id_fn, write=None if ro else id_fn)


def inverse_transform(prop):
    """
    A param_transform for a parameter that is the inverse of the other.
    """

    return param_transform(prop, read=lambda x: 1 / x, write=lambda x: 1 / x)
