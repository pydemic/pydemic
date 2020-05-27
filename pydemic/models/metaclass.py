from abc import ABCMeta

from ..mixins import Meta


class ModelMeta(ABCMeta):
    """
    Metaclass for model classes.
    """

    _meta: "Meta"

    def __init__(cls, name, bases, ns):
        meta = ns.pop("Meta", None)
        super().__init__(name, bases, ns)

        cls._meta = Meta.from_arguments(cls, meta)
