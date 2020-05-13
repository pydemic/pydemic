from collections import namedtuple

Result = namedtuple("Result", ["value", "info"])


class ImproperlyConfigured(Exception):
    """
    Exception raised when trying to initialize object with improper configuration.
    """
