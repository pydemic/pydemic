from abc import ABCMeta

from ..mixins import Meta


class ModelMeta(ABCMeta):
    """
    Metaclass for model classes.
    """

    meta: "Meta"

    def __init__(cls, name, bases, ns):
        meta = ns.pop("Meta", None)
        super().__init__(name, bases, ns)

        cls.meta = Meta.from_arguments(cls, meta)
