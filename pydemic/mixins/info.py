from .base_attr import BaseAttr


class Info(BaseAttr):
    """
    Info objects store static information about a simulation.

    Not all information exposed by this object is necessarily relevant to a
    simulation model, but might be useful for later analysis or for
    convenience.

    Info attributes are organized in a dotted namespace.
    """

    __slots__ = ()
    _method_namespace = "info"

    @property
    def _cache(self):
        return self.model._info_cache

    def clear(self):
        self._cache.clear()
