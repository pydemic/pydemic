from collections import namedtuple

Result = namedtuple("Result", ["value", "info"])
ValueStd = namedtuple("ValueStd", ["value", "std"])
ValueCI = namedtuple("ValueCI", ["value", "low", "high"])


class ImproperlyConfigured(Exception):
    """
    Exception raised when trying to initialize object with improper configuration.
    """
