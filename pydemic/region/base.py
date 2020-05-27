from mundi import Region


class RegionPropertyMeta(type):
    """
    Metaclass register cached methods.
    """

    def __new__(mcs, name, bases, ns):
        return super().__new__(mcs, name, bases, ns)


class RegionProperty(metaclass=RegionPropertyMeta):
    """
    Base class for all region properties.
    """

    __slots__ = ("region",)
    region: Region

    def __init__(self, region):
        self.region = region

    def __getstate__(self):
        return self.region

    def __setstate__(self, state):
        self.region = state


def cached(fn):
    """
    Mark methods as cached.
    """
    fn.is_cached = True
    return fn


def functools_lru_backend():
    pass


def joblib_backend():
    pass


def streamlit_backend():
    pass
