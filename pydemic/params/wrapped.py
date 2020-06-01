from ..utils import format_args


class WrappedParams:
    """
    A wrapper object that exposes a disease as a param namespace.
    """

    __slots__ = ("_wrapped", "_args", "_kwargs", "_cache", "_keys", "_blacklist")

    def __init__(self, wrapped, *args, params_blacklist=None, **kwargs):
        self._wrapped = wrapped
        self._args = args
        self._kwargs = kwargs
        self._cache = {}

        if params_blacklist is None:
            params_blacklist = getattr(wrapped, "PARAMS_BLACKLIST", None)
        if params_blacklist is None:
            params_blacklist = getattr(wrapped, "params_blacklist", ())
        self._blacklist = frozenset(params_blacklist)

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError(f"keys must be strings, got {item!r}")
        try:
            return self._fetch_param(item)
        except ValueError:
            raise KeyError(item)

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        try:
            return self._fetch_param(item)
        except ValueError:
            raise AttributeError(item)

    def __getstate__(self):
        return self._wrapped, self._args, self._kwargs

    def __setstate__(self, state):
        self._wrapped, self._args, self._args = state
        self._cache = {}

    def __repr__(self):
        args = format_args(self._wrapped, *self._args, **self._kwargs)
        return f"WrappedParams({args})"

    def __iter__(self):
        cls = type(self)
        blacklist = self._blacklist
        for attr in dir(self._wrapped):
            if attr.startswith("_") or hasattr(cls, attr) or attr in blacklist:
                continue

            value = getattr(self._wrapped, attr)
            if callable(value):
                yield attr

    def _fetch_param(self, name):
        """
        Obtain parameter with the given name and save result in cache.
        """
        try:
            return self._cache[name]
        except KeyError:
            try:
                value = getattr(self._wrapped, name)
            except AttributeError:
                raise ValueError(f"invalid parameter: {name!r}")
            if callable(value):
                value = value(*self._args, **self._kwargs)
            self._cache[name] = value
            return value
