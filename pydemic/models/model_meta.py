from abc import ABCMeta

from ..mixins import ParamsInfo, Meta


class ModelMeta(ABCMeta):
    """
    Metaclass for model classes.
    """

    DATA_ALIASES: dict
    _meta: "Meta"

    def __init__(cls, name, bases, ns):
        meta = ns.pop("Meta", None)
        super().__init__(name, bases, ns)
        cls._meta = Meta.from_arguments(cls, bases, meta)
        cls.params = ParamsInfo(cls)
