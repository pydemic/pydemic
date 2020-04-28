from ..packages import sk


class state_property(property):
    """
    A property that exposes a state component as rw.
    """

    def __init__(self, i, ro=False):
        def fget(self):
            return self.state[i]

        def fset(self, value):
            self.state[i] = value

        args = (fget,) if ro else (fget, fset)
        super().__init__(*args)


class param_property(property):
    """
    A property that simply mirrors a given param value.
    """

    is_param = True
    is_derived = False

    def __init__(self, name=None, ro=False):
        self.name = name
        prop = self

        def fget(self):
            return self._params[prop.name].value

        def fset(self, value):
            self.set_param(prop.name, value)

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

    def __init__(self, prop, read, write=None):
        read = sk.extract_function(read)
        if write is not None:
            write = sk.extract_function(write)

        def fget(self):
            try:
                value = self._params[prop].value
            except KeyError:
                value = self.get_param(prop)
            return read(value)

        def fset(self, value):
            value = write(value)
            self.set_param(prop, value)

        args = (fget,) if write else (fget, fset)
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
