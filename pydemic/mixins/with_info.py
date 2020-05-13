from abc import ABC

from .info import Info


class WithInfoMixin(ABC):
    """
    Subclasses that adopt this mixin gain a "info" attribute that expose
    useful information about its owner.

    In the context of simulation models, it shows useful static information
    about the simulation.
    """

    info: Info

    def __init__(self):
        info_class = getattr(self, "info_class", None)
        if info_class is None and hasattr(self, "_meta"):
            info_class = getattr(self._meta, "info_class", Info)

        self.info = info_class(self)
