import sys
from functools import wraps

from .base import Property


def method_as_function(method, wrapper=None):
    """
    Take a method from a property object and transform it in a independent
    function that takes the owner object as first input.
    """

    if wrapper is None:
        try:
            mod = sys.modules[method.__module__]
            path = method.__qualname__.partition(".")[0]
            wrapper = getattr(mod, path)
        except AttributeError:
            pass
        else:
            wrapper = lambda x: Property(x)

    @wraps(method)
    def fn(_object, *args, **kwargs):
        return method(wrapper(_object), *args, **kwargs)

    return fn


def function_as_method(fn):
    """
    Convert function that receives value to a method of a property accessor
    for that value.
    """

    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        return fn(self._object, *args, **kwargs)

    return wrapped
